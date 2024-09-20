"""
Implementation of the paper Hybrid evolutionary algorithm with extreme machine
learning fitness function evaluation for two-stage capacitated facility
location problems by Guo et al. (2017).
"""

import random
from typing import Any

import numpy as np
import pandas as pd
from elm import ELM
from overrides import override

from approxml.sample import (
    BestSamplingStrategy,
    RandomSamplingStrategy,
    SamplingStrategy,
)
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual
from eckity.population import Population


class HEAFAPopulationEvaluator(PopulationEvaluator):
    def __init__(
        self,
        population_sample_size: float = 0.1,
        sample_strategy: SamplingStrategy = RandomSamplingStrategy(),
        gen_sample_step: int = 1,
        model_params: dict[str, Any] = None,
        handle_duplicates="ignore",
        job_id=None,
    ):
        super().__init__()
        self.approx_fitness_error = float("inf")
        self.population_sample_size = population_sample_size
        self.sample_strategy = sample_strategy
        self.gen_sample_step = gen_sample_step

        # ML model
        self.model = ELM(**model_params)

        # generation counter
        self.gen = 0

        # counters of real/approximate fitness scores
        self.fitness_count = 0
        self.approx_count = 0

        self.job_id = job_id

        self.gen_population = None
        self.prev_gen_individuals = []
        self.best_of_run = None
        self.best_fitness = None

        # population dataset
        self.df = None

        # Process/Thread pool executor, used for parallel fitness evaluation
        self.executor = None

        # how to handle duplicate individuals in the dataset
        self.handle_duplicates = handle_duplicates

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

        for i, sub_population in enumerate(population.sub_populations):
            if self.gen > 0:
                self._approximate_individuals(
                    sub_population.individuals, sub_population.evaluator
                )

            else:
                # Compute fitness scores of the whole population
                fitnesses = self.evaluate_individuals(
                    sub_population.individuals, sub_population.evaluator
                )

                for ind, fitness_score in zip(
                    sub_population.individuals, fitnesses
                ):
                    ind.fitness.set_fitness(fitness_score)

                # Update population dataset
                vecs = [ind.get_vector() for ind in sub_population.individuals]
                self._update_dataset(vecs, fitnesses)

                if i == 0:
                    self.train_model()

        # Update best individual of the current generation
        sub_population = population.sub_populations[0]
        best_ind = self._get_best_individual(sub_population.individuals)

        # Update best individual of the run
        if self.gen == 0:
            self.best_of_run = best_ind
            self.best_fitness = best_ind.get_pure_fitness()
        else:
            # Apply local search to the best individual
            best_ind = self._local_search(best_ind)
            ind_eval = population.sub_populations[0].evaluator
            best_of_gen_fitness = ind_eval.evaluate_individual(best_ind)
            self.fitness_count += 1

            if best_of_gen_fitness > self.best_fitness:
                self.best_of_run = best_ind

        # Selection operation II - Mu + Lambda
        population_size = sub_population.population_size
        combined_individuals = (
            sub_population.individuals + self.prev_gen_individuals
        )
        sub_population.individuals = sorted(
            combined_individuals,
            key=lambda ind: ind.get_pure_fitness(),
            reverse=sub_population.higher_is_better,
        )[:population_size]

        # restart strategy
        n_restarted = population_size // 10
        best_vec = sub_population.individuals[0].get_vector()
        worst_vec = sub_population.individuals[-1].get_vector()
        length = len(best_vec)
        n_shared = sum(
            [1 if best_vec[i] == worst_vec[i] else 0 for i in range(length)]
        )
        if n_shared >= 0.9 * length:
            new_individuals = sub_population.creators[0].create_individuals(
                n_restarted, sub_population.higher_is_better
            )

            # Evaluate fitness for new individuals
            new_fitnesses = self.evaluate_individuals(
                new_individuals, sub_population.evaluator
            )
            for ind, fitness_score in zip(new_individuals, new_fitnesses):
                ind.fitness.set_fitness(fitness_score)

            # Update population dataset with new individuals
            new_vecs = [ind.get_vector() for ind in new_individuals]
            self._update_dataset(new_vecs, new_fitnesses)

            sub_population.individuals[-n_restarted:] = new_individuals

        # clone since the individuals will be modified in the next generation
        # by in-place genetic operators
        self.prev_gen_individuals = [
            ind.clone() for ind in sub_population.individuals
        ]
        self.gen += 1
        return best_ind

    def _approximate_individuals(
        self,
        individuals: list[Individual],
        ind_eval: SimpleIndividualEvaluator,
    ) -> list[Individual]:
        # Obtain fitness score predictions from ML model
        preds = self.predict(individuals)

        # Dictionary of (individual, true fitness score)
        inds2scores = {}
        sample_size = 0
        sample_inds = []

        if self.gen > 0 and self.gen % self.gen_sample_step == 0:
            # Sample a subset of the population and compute their fitness
            sample_size = (
                self.population_sample_size
                if isinstance(self.population_sample_size, int)
                else int(len(individuals) * self.population_sample_size)
            )

            if sample_size > 0:
                # Sample a subset of individuals from the population
                # and compute their real fitness score
                sample_inds = self.sample_strategy.sample(
                    individuals, sample_size, preds
                )
                fitness_scores = self.evaluate_individuals(
                    sample_inds, ind_eval
                )
                inds2scores.update(zip(sample_inds, fitness_scores))

                # update population dataset with sampled individuals
                vectors = [ind.get_vector() for ind in sample_inds]
                self._update_dataset(vectors, fitness_scores)

                # train the model with the updated dataset
                self.train_model()

        # Update fitness scores for the population
        for j, ind in enumerate(individuals):
            if ind in inds2scores:
                ind.fitness.set_fitness(inds2scores[ind])
            else:
                ind.fitness.set_fitness(preds[j])

        # Update approximation counter
        n_approximated = len(individuals) - sample_size
        self.approx_count += n_approximated

        return sample_inds

    def _get_best_individual(
        self, individuals: list[Individual]
    ) -> Individual:
        best_ind: Individual = individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness
        return best_ind

    def evaluate_individuals(
        self,
        individuals: list[Individual],
        ind_eval: SimpleIndividualEvaluator,
    ) -> list[float]:
        """
        Evaluate the fitness scores of a given individuals list

        Parameters
        ----------
        individuals : list[Individual]
            list of individuals

        Returns
        -------
        list[float]
            list of fitness scores, by the order of the individuals
        """
        eval_results = self.executor.map(
            ind_eval.evaluate_individual, individuals
        )
        fitness_scores = list(eval_results)
        self.fitness_count += len(individuals)
        return fitness_scores

    def _update_dataset(
        self, ind_vectors: list[list[int]], fitnesses: list[float]
    ):
        df = pd.DataFrame(np.array(ind_vectors))
        df["fitness"] = np.array(fitnesses)

        # Filter out individuals with infinite fitness scores
        df = df[df["fitness"] != -np.inf]

        if self.df is None:
            self.df = df

        else:
            self.df = pd.concat([self.df, df], ignore_index=True, copy=False)
            n_features = self.df.shape[1] - 1

            # Handle rows with duplicate individuals:
            if self.handle_duplicates in ["last", "first"]:
                # If the same individual is evaluated multiple times,
                # keep the first/last evaluation.
                self.df.drop_duplicates(
                    subset=range(n_features),
                    keep=self.handle_duplicates,
                    inplace=True,
                )
            else:
                self.df.drop_duplicates(inplace=True)

    def train_model(self) -> None:
        """
        Fit the Machine Learning model incrementally.

        The model is trained to estimate the fitness score of an individual
        given its representation.

        Parameters
        ----------
        individuals : list[Individual]
            list of individuals in the sub-population
        fitnesses : list[float]
            Fitness scores of the individuals, respectively.
        """
        X, y = self.df.iloc[:, :-1].to_numpy(), self.df["fitness"].to_numpy()

        # Fit the model on the whole dataset
        self.model.fit(X, y)

    def predict(self, individuals: list[Individual]):
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
        X = np.array(ind_vectors)
        preds = self.model.predict(X)
        return preds

    def _local_search(self, best_ind):
        copies = []
        best_vector = best_ind.get_vector()
        bounds = set(range(best_ind.bounds[0], best_ind.bounds[1] + 1))
        for i in range(len(best_vector)):
            best_ind_copy = best_ind.clone()
            cell_values = bounds - set([best_vector[i]])
            best_ind_copy.set_cell_value(i, random.choice(list(cell_values)))
            copies.append(best_ind_copy)

        # Update fitness scores for copies obtained from LS
        preds = self.predict(copies)
        for j, ind in enumerate(copies):
            ind.fitness.set_fitness(preds[j])
            if ind.fitness.better_than(ind, best_ind.fitness, best_ind):
                best_ind = ind

        return best_ind

    def print_best_of_run(self):
        """
        Print the best individual of the run.
        """
        print(f"Best of run: {self.best_of_run.get_vector()}")
        print("best fitness:", self.best_fitness)

    def export_dataset(self, folder_path) -> None:
        """
        Export the dataset used to train the model to a CSV file.
        """
        self.df.to_csv(f"{folder_path}/{self.job_id}.csv", index=False)
