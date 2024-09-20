"""
Implementation of the paper Fitness Inheritance in Genetic Algorithms
by Smith et al. (1995).
"""

import numpy as np
import pandas as pd

from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from eckity.population import Population
from eckity.individual import Individual
from eckity.fitness.fitness import Fitness

from overrides import override
import random
from typing import List
import examples.utils as utils


class InheritancePopulationEvaluator(PopulationEvaluator):
    def __init__(
        self,
        weighted=False,
        population_sample_size=0.1,
        gen_sample_step=1,
        similarity_metric="cosine",
        handle_duplicates="ignore",
        sample_strategy="random",
        hide_fitness=False,
        job_id=None,
    ):
        super().__init__()
        self.weighted = weighted
        self.population_sample_size = population_sample_size
        self.gen_sample_step = gen_sample_step
        self.similarity_metric = similarity_metric

        self.hide_fitness = hide_fitness
        self.fitness_count = 0
        self.approx_count = 0

        self.handle_duplicates = handle_duplicates
        self.job_id = job_id
        self.sample_strategy = sample_strategy

        self.gen = 0

        # map individual IDs to their fitness score
        self.ids2fitness_genome = {}

        # for statistics
        self.df = None

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
        best_of_gen_candidates = []

        for i, sub_population in enumerate(population.sub_populations):
            if self.gen > 0:
                best_of_gen_candidates = self._approximate_individuals(
                    sub_population.individuals, sub_population.evaluator
                )
            else:
                # Compute fitness scores of the whole population
                fitnesses = self._evaluate_individuals(
                    sub_population.individuals, sub_population.evaluator, i
                )

                for ind, fitness_score in zip(
                    sub_population.individuals, fitnesses
                ):
                    ind.fitness.set_fitness(fitness_score)

                best_of_gen_candidates = sub_population.individuals

                # Update population dataset
                vecs = [ind.get_vector() for ind in sub_population.individuals]
                self._update_dataset(vecs, fitnesses)

                # Update fitness evluation counter
                self.fitness_count += len(sub_population.individuals)

        # Update best individual of the current generation
        if best_of_gen_candidates:
            best_ind = self._get_best_individual(best_of_gen_candidates)
        else:
            best_ind = None

        self.best_in_gen = best_ind

        self.gen += 1
        return best_ind

    def _get_best_individual(
        self, individuals: List[Individual]
    ) -> Individual:
        best_ind: Individual = individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness
        return best_ind

    def _approximate_individuals(
        self,
        individuals: List[Individual],
        ind_eval: SimpleIndividualEvaluator,
    ) -> None:
        # Obtain inherited fitness score
        inherited_fitness = self.inherit_fitness(individuals)

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
                sample_inds = self._sample_individuals(
                    individuals, sample_size
                )
                fitness_scores = self._evaluate_individuals(
                    sample_inds, ind_eval
                )
                inds2scores.update(zip(sample_inds, fitness_scores))

                # update population dataset with sampled individuals
                vectors = [ind.get_vector() for ind in sample_inds]
                self._update_dataset(vectors, fitness_scores)

        # Update fitness scores for the population
        for j, ind in enumerate(individuals):
            if not self.hide_fitness and ind in inds2scores:
                ind.fitness.set_fitness(inds2scores[ind])
            else:
                ind.fitness.set_fitness(inherited_fitness[j])

        # Update approximation counter
        n_approximated = len(individuals) - sample_size
        self.approx_count += n_approximated
        # update fitness evaluations counter
        self.fitness_count += sample_size

        return sample_inds

    def _sample_individuals(
        self, individuals: List[Individual], sample_size: int
    ) -> List[Individual]:
        """
        Sample individuals from the population according to the given strategy

        Parameters
        ----------
        individuals: List[Individual]
            the population of the evolutionary experiment
        sample_size: int
            the number of individuals to sample
        """
        # Sample individuals randomly
        if self.sample_strategy == "random":
            sample_inds = random.sample(individuals, sample_size)
        else:
            raise NotImplementedError(
                f"Sampling strategy {self.sample_strategy} not implemented"
            )
        return sample_inds

    def _update_dataset(self, ind_vectors: List[List], fitnesses: List[float]):
        df = pd.DataFrame(np.array(ind_vectors))
        df["fitness"] = np.array(fitnesses)
        df["gen"] = self.gen

        # Filter out individuals with infinite fitness scores
        df = df[df["fitness"] != -np.inf]

        if self.df is None:
            self.df = df

        else:
            self.df = pd.concat([self.df, df], ignore_index=True, copy=False)
            n_features = self.df.shape[1] - 2

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

    def inherit_fitness(self, individuals: List[Individual]) -> List[float]:
        """
        Computes the inherited fitness score for each individual in the
        population according to its parents

        Parameters
        ----------
        individuals: List[Individual]
            the population of the evolutionary experiment

        Returns
        -------
        List[float]
            the inherited fitness score for each individual in the population
        """
        if self.similarity_metric == "cosine":
            metric = utils.cosine_similarity
        else:
            raise NotImplementedError(
                f"Similarity metric {self.similarity_metric} not implemented"
            )

        fitness_scores = []
        for ind in individuals:
            if len(ind.parents) == 0:
                fitness_scores.append(self.ids2fitness_genome[ind.id][0])
                continue

            # Compute similarity between the individual and its parents
            # (only consider the first two parents)
            parents = ind.parents[:2]
            parent_fitness = [self.ids2fitness_genome[p][0] for p in parents]
            parent_vectors = [self.ids2fitness_genome[p][1] for p in parents]

            ind_vector = ind.get_vector()

            if self.weighted:
                # weighted average according to similarity
                similarities = [
                    metric(ind_vector, pvec) for pvec in parent_vectors
                ]
                # normalize similarities to sum to one
                similarities = [s / sum(similarities) for s in similarities]
                fitness = sum(
                    [s * f for s, f in zip(similarities, parent_fitness)]
                )
            else:
                # unweighted average
                fitness = sum(parent_fitness) / len(parent_fitness)

            fitness_scores.append(fitness)

        # empty the dictionary for memory efficiency
        # since its values won't be used in next generation
        del self.ids2fitness_genome
        self.ids2fitness_genome = {}

        return fitness_scores

    def update_fitness_and_genomes(self, sender, data_dict):
        individuals = sender.selected_individuals
        self.ids2fitness_genome.update(
            {
                ind.id: ((ind.fitness.get_pure_fitness(), ind.vector.copy()))
                for ind in individuals
            }
        )

    def _evaluate_individuals(
        self,
        individuals: List[Individual],
        ind_eval: SimpleIndividualEvaluator,
        sub_population_idx=0,
    ) -> List[float]:
        eval_results = self.executor.map(
            ind_eval.evaluate_individual, individuals
        )
        fitness_scores = list(eval_results)
        return fitness_scores

    def export_dataset(self, folder_path) -> None:
        """
        Export the dataset used to train the model to a CSV file.
        """
        self.df.to_csv(f"{folder_path}/{self.job_id}.csv", index=False)
