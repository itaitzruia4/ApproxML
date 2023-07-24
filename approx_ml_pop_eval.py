import random
import numpy as np
import pandas as pd
import subprocess
import os
import json

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.linear_model import SGDRegressor
from overrides import override

from eckity.evaluators.simple_individual_evaluator \
    import SimpleIndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual
from eckity.population import Population

from typing import List, Tuple
import utils


class ApproxMLPopulationEvaluator(PopulationEvaluator):
    """
    Fitness approximation population evaluator

    Parameters
    ----------
    should_approximate : callable, optional, default=None
        Whether the fitness should be approximated in the current generation.
        If None, always approximate.
    population_sample_size : int or float, optional, default=10
        number (or percentage) of individuals to sample and compute their
        fitness when approximating , by default 10%
    sample_strategy : {'random', 'cosine'}, optional, default='random'
        strategy to use when sampling individuals for fitness approximation
    gen_sample_step : int, optional, default=1
        how many generations should pass between samples
    scoring : callable, optional, default=mean_absolute_error
        evaluation metric for the model
    model_type : sklearn model, optional, default=SGDRegressor
        model to use for fitness approximation
    model_params : dict, optional, default=None
        model parameters
    gen_weight : callable, optional, default=lambda gen: gen + 1
        function to compute the weight of each generation in the training.
        The weight is used to give more importance to later generations.
    hide_fitness : bool, optional, default=False
        whether to return approximated fitness scores for sampled individuals
    ensemble : bool, optional, default=False
        whether to use an ensemble of models for fitness approximation
    n_folds : int, optional, default=None
        number of folds to use in cross validation
    handle_duplicates : {'ignore', 'first', 'last}, optional, default='ignore'
        how to handle duplicate individuals in the dataset
    use_slurm : bool, optional, default=False
        whether to use slurm for remote evaluation
    job_id : str, optional, default=None
        slurm job id
    split_experiment : bool, optional, default=False
        whether to split the experiment into two - one for evolution and one
        for fitness approximation.
        This is used for statistics.
    """

    def __init__(self,
                 should_approximate: callable = None,
                 population_sample_size=0.1,
                 sample_strategy='random',
                 gen_sample_step=1,
                 scoring=mean_absolute_error,
                 model_type=SGDRegressor,
                 model_params=None,
                 gen_weight=lambda gen: gen + 1,
                 hide_fitness=False,
                 ensemble=False,
                 n_folds=None,
                 handle_duplicates='ignore',
                 use_slurm=False,
                 job_id=None):
        super().__init__()
        self.approx_fitness_error = float('inf')
        self.population_sample_size = population_sample_size
        self.sample_strategy = sample_strategy
        self.gen_sample_step = gen_sample_step
        self.scoring = scoring
        self.hide_fitness = hide_fitness

        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model_type = model_type

        # ML model
        self.model = None

        # generation counter
        self.gen = 0

        # counters of real/approximate fitness scores
        self.fitness_count = 0
        self.approx_count = 0

        self.use_slurm = use_slurm
        self.job_id = job_id

        self.gen_population = None
        self.best_in_gen = None

        if should_approximate is None:
            def default_should_approximate(eval):
                return True
            should_approximate = default_should_approximate

        self.should_approximate = should_approximate
        self.is_approx = False

        # population dataset
        self.df = None
        
        # Process/Thread pool executor, used for parallel fitness evaluation
        self.executor = None

        # Ensemble of models
        self.ensemble = ensemble
        if self.ensemble:
            self.models = dict()

        # generation weighting function for sample weights
        self.gen_weight = gen_weight

        # number of folds to use in cross validation
        self.n_folds = n_folds

        # how to handle duplicate individuals in the dataset
        self.handle_duplicates = handle_duplicates

        # best individual of each generation
        self.best_individuals = []

    @override
    def _evaluate(self, population: Population) -> Individual:
        """
        Updates the fitness score of the given individuals,
        then returns the best individual

        Parameters
        ----------
        population:
            the population of the evolutionary experiment

        Returns
        -------
        Individual
            the individual with the best fitness out of the given individuals
        """
        super()._evaluate(population)

        self.gen_population = population
        self.is_approx = self.gen > 0 and self.should_approximate(self)

        for i, sub_population in enumerate(population.sub_populations):
            if self.is_approx:
                self._approximate_individuals(sub_population.individuals,
                                              sub_population.evaluator)
            else:
                # Compute fitness scores of the whole population
                fitnesses = self._evaluate_individuals(
                    sub_population.individuals,
                    sub_population.evaluator,
                    i
                )

                for ind, fitness_score in zip(sub_population.individuals,
                                              fitnesses):
                    ind.fitness.set_fitness(fitness_score)

                # Update population dataset
                vecs = [ind.get_vector() for ind in sub_population.individuals]
                self._update_dataset(vecs, fitnesses)

                # Update fitness evluation counter
                self.fitness_count += len(sub_population.individuals)

                if i == 0:
                    self.train_model()

        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals
        best_ind = self._get_best_individual(individuals)
        self.best_in_gen = best_ind
        self.best_individuals.append(best_ind)

        self.gen += 1
        return best_ind

    def _approximate_individuals(self,
                                 individuals: List[Individual],
                                 ind_eval: SimpleIndividualEvaluator) -> None:
        # Obtain fitness score predictions from ML model
        preds = self.predict(individuals)

        # Dictionary of (individual, true fitness score)
        inds2scores = {}
        sample_size = 0

        if self.gen > 0 and self.gen % self.gen_sample_step == 0:
            # Sample a subset of the population and compute their fitness
            sample_size = self.population_sample_size \
                if isinstance(self.population_sample_size, int) \
                else int(len(individuals) * self.population_sample_size)

            if sample_size > 0:
                # Sample a subset of individuals from the population
                # and compute their real fitness score
                sample_inds = self._sample_individuals(individuals,
                                                       sample_size)
                fitness_scores = self._evaluate_individuals(
                    sample_inds,
                    ind_eval
                )
                inds2scores.update(zip(sample_inds, fitness_scores))

                # update population dataset with sampled individuals
                vectors = [ind.get_vector() for ind in sample_inds]
                self._update_dataset(vectors, fitness_scores)

                # train the model with the updated dataset
                self.train_model()

        # Update fitness scores for the population
        for j, ind in enumerate(individuals):
            if not self.hide_fitness and ind in inds2scores:
                ind.fitness.set_fitness(inds2scores[ind])
            else:
                ind.fitness.set_fitness(preds[j])

        # Update approximation counter
        n_approximated = len(individuals) - sample_size
        self.approx_count += n_approximated
        # update fitness evaluations counter
        self.fitness_count += sample_size

    def _sample_individuals(self, individuals: List[Individual],
                            sample_size: int) -> List[Individual]:
        """
        Sample individuals from the population according to the given strategy

        Parameters
        ----------
        individuals: List[Individual]
            the population of the evolutionary experiment
        sample_size: int
            the number of individuals to sample
        """
        # Sample individuals with the lowest cosine similarity to the dataset
        if self.sample_strategy == 'cosine':
            ind_scores = zip(individuals, self._get_cosine_scores(individuals))
            sample_inds = [
                ind
                for ind, _
                in sorted(ind_scores, key=lambda x: x[1])[:sample_size]
            ]
        # Sample individuals randomly
        elif self.sample_strategy == 'random':
            sample_inds = random.sample(individuals, sample_size)
        return sample_inds
    
    def _get_cosine_scores(self,
                           individuals: List[Individual]
                           ) -> List[float]:
        """
        Compute the cosine similarity between each individual and the dataset

        Parameters
        ----------
        individuals : List[Individual]
            the population of the evolutionary experiment

        Returns
        -------
        List[float]
            list of cosine similarity scores
        """
        genotypes = self.df.iloc[:, :-2]
        # obtain the maximum similarity score in the dataset
        # for each individual
        similarity_scores = [
            genotypes.apply(lambda row:
                            utils.cosine_similarity(ind.vector, row),
                            axis=1, raw=True
                            ).max()
            for ind in individuals
        ]
        return similarity_scores

    def _get_best_individual(self,
                             individuals: List[Individual]) -> Individual:
        best_ind: Individual = individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness
        return best_ind

    def _evaluate_individuals(self,
                              individuals: List[Individual],
                              ind_eval: SimpleIndividualEvaluator,
                              sub_population_idx=0) -> List[float]:
        """
        Evaluate the fitness scores of a given individuals list

        Parameters
        ----------
        individuals : List[Individual]
            list of individuals

        Returns
        -------
        List[float]
            list of fitness scores, by the order of the individuals
        """

        if not self.use_slurm:
            eval_results = self.executor.map(ind_eval.evaluate_individual,
                                             individuals)
            fitness_scores = list(eval_results)
        
        else:
            fitness_scores = self._evaluate_slurm(individuals,
                                                  sub_population_idx)
        return fitness_scores

    def _evaluate_slurm(self,
                        individuals: List[Individual],
                        subpop_idx=0) -> List[float]:
        device = 'cpu'
        # Create a configuration file for the individuals
        config = {i: ind.get_vector()
                  for i, ind in enumerate(individuals, start=1)}

        # create folder for this experiment if it doesn't exist
        if not os.path.exists(f'configs/{device}/{self.job_id}'):
            os.mkdir(f'configs/{device}/{self.job_id}')
    
        with open(
            f'configs/{device}/{self.job_id}/{self.gen}_{subpop_idx}.json', 'w'
        ) as f:
            json.dump(config, f)

        # create folder for this experiment if it doesn't exist
        if not os.path.exists(f'jobs/{device}/{self.job_id}'):
            os.mkdir(f'jobs/{device}/{self.job_id}')

        # create a job script for this generation and submit it
        sbatch_str = utils.generate_sbatch_str(self.gen,
                                               len(individuals),
                                               device,
                                               self.job_id,
                                               subpop_idx)
        with open(
            f'jobs/{device}/{self.job_id}/sbatch_{self.gen}_{subpop_idx}.sh',
            'w'
        ) as f:
            f.write(sbatch_str)

        cmdline = [
            'sbatch',
            f'jobs/{device}/{self.job_id}/sbatch_{self.gen}_{subpop_idx}.sh'
        ]
        proc = subprocess.Popen(cmdline)

        fitness_scores = []

        try:
            proc.wait(timeout=utils.EVAL_TIMEOUT)

        except subprocess.TimeoutExpired as e:
            print(
                f'Job {device}/{self.job_id}/{self.gen}_{subpop_idx}',
                'timed out. error:', e
            )
            fitness_scores = [-np.inf] * len(individuals)
            return fitness_scores
        
        for i in range(1, len(individuals) + 1):
            try:
                # extract fitness from job .out file
                with open(
                    f'jobs/{device}/{self.job_id}/{self.gen}_{subpop_idx}_{i}.out', 'r'
                ) as f:
                    # ignore first line
                    f.readline()
                    # parse fitness score
                    fitness = float(f.readline())

            except (FileNotFoundError, ValueError) as e:
                print(
                    f'Job {device}/{self.job_id}/{self.gen}_{subpop_idx}_{i}',
                    'failed. error:', e
                )
                fitness = -np.inf

                error_file_path = f'jobs/{device}/{self.job_id}/job-{self.gen}_{subpop_idx}_{i}.out'
                if os.path.exists(error_file_path):
                    with open(error_file_path, 'r') as f:
                        print(f.read())

            # Remove individual output file
            try:
                
                os.remove(
                    f'jobs/{device}/{self.job_id}/{self.gen}_{subpop_idx}_{i}.out'
                )
            except FileNotFoundError as e:
                print(
                    f'File {device}/{self.job_id}/{self.gen}_{subpop_idx}_{i}',
                    'not found. Error:', e
                )
            
            fitness_scores.append(fitness)

        # Remove config file
        try:
            os.remove(
                f'configs/{device}/{self.job_id}/{self.gen}_{subpop_idx}.json'
            )
        except FileNotFoundError as e:
            print(
                f'File not found in job {device}_{self.job_id}_{self.gen}_{subpop_idx}:', e
            )

        return fitness_scores

    def _update_dataset(self, ind_vectors: List[List], fitnesses: List[float]):
        df = pd.DataFrame(np.array(ind_vectors))
        df['fitness'] = np.array(fitnesses)
        df['gen'] = self.gen

        # Filter out individuals with infinite fitness scores
        df = df[df['fitness'] != -np.inf]

        if self.df is None:
            self.df = df

        else:
            self.df = pd.concat([self.df, df], ignore_index=True, copy=False)
            n_features = self.df.shape[1] - 2

            # Handle rows with duplicate individuals:
            if self.handle_duplicates in ['last', 'first']:
                # If the same individual is evaluated multiple times,
                # keep the first/last evaluation.
                self.df.drop_duplicates(subset=range(n_features),
                                        keep=self.handle_duplicates,
                                        inplace=True)
            else:
                self.df.drop_duplicates(inplace=True)

    def train_model(self) -> None:
        """
        Fit the Machine Learning model incrementally.

        The model is trained to estimate the fitness score of an individual
        given its representation.

        Parameters
        ----------
        individuals : List[Individual]
            List of individuals in the sub-population
        fitnesses : List[float]
            Fitness scores of the individuals, respectively.
        """
        self.model = self.model_type(**self.model_params)

        # Add new model to the ensemble (if ensemble is enabled)
        if self.ensemble:
            self.models[self.gen] = self.model

        X, y = self.df.iloc[:, :-2].to_numpy(), self.df['fitness'].to_numpy()
        # Vector of generation number of each individual in the dataset
        #  (used for sample weights)
        w = self.gen_weight(self.df['gen'].to_numpy())

        # Perform KFold CV to estimate the fitness error of the model
        if self.n_folds is not None:
            kf = KFold(n_splits=self.n_folds, shuffle=True)
            scorer = make_scorer(self.scoring)
            cross_val_scores = cross_val_score(self.model,
                                               X, y, cv=kf,
                                               scoring=scorer)
            self.approx_fitness_error = np.mean(cross_val_scores)

        # Now fit the model on the whole training set
        self.model.fit(X, y, sample_weight=w)

    def predict(self, individuals: List[Individual]):
        """
        Perform fitness approximation of a given list of individuals.

        Parameters
        ----------
        individuals :   list of individuals
            Individuals in the sub-population

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        ind_vectors = [ind.get_vector() for ind in individuals]

        if self.ensemble:
            weights = [self.gen_weight(gen) for gen in self.models]
            preds = [model.predict(ind_vectors)
                     for model in self.models.values()]
            preds = np.average(preds, weights=weights, axis=0)

        else:
            preds = self.model.predict(ind_vectors)

        return preds
    
    def print_best_of_run(self, sender, data_dict) -> Tuple[Individual, float]:
        """
        Get the best individual of the run.

        Returns
        -------
        Individual
            Best individual of the run
        """
        population = data_dict['population']
        individual_evaluator = population.sub_populations[0].evaluator
        best_fitness_scores = self._evaluate_individuals(self.best_individuals,
                                                         individual_evaluator)
        best_ind_idx = np.argmax(best_fitness_scores)
        best_ind = self.best_individuals[best_ind_idx]
        best_fitness = best_fitness_scores[best_ind_idx]
        print('Best individual\n:', best_ind.vector)
        print('Best fitness:', best_fitness)
    
    def export_dataset(self, folder_path) -> None:
        """
        Export the dataset used to train the model to a CSV file.
        """
        self.df.to_csv(f'{folder_path}/{self.job_id}.csv', index=False)
