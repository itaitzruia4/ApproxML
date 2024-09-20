from random import random

from eckity.genetic_operators.genetic_operator import GeneticOperator


class CXReproduction(GeneticOperator):
    def __init__(self,
                 probability=1,
                 arity=2,
                 events=None,
                 p_min=0.5,
                 p_max=0.9):
        super().__init__(probability=probability, arity=arity, events=events)
        self.applied_individuals = None
        self.p_min = p_min
        self.p_max = p_max
        self.best_fitness = None
        self.avg_fitness = None
        self.parents_fitness = []
        

    def apply(self, individuals):
        self.adapt_probability()
        self.applied_individuals = individuals

        vectors = [ind.get_vector() for ind in individuals]
        offspring = self._reproduce_offspring(vectors)

        individuals[0].set_vector(offspring)

        return individuals[:1]

    def _reproduce_offspring(self, parents):
        offspring = [
            parents[0][i] if parents[0][i] == parents[1][i] else None
            for i in range(len(parents[0]))
        ]

        for i in range(len(offspring)):
            if offspring[i] is not None:
                # cell value is already assigned and same for both parents
                continue
            if random() < 0.5:
                # assign cell value from first parent
                offspring[i] = parents[0][i]
            else:
                # assign cell value from second parent
                offspring[i] = parents[1][i]
        return offspring

    def adapt_probability(self):
        fitness_scores = self.parents_fitness
        max_observed_fitness = max(fitness_scores)

        best_fitness, avg_fitness = self.best_fitness, self.avg_fitness

        self.probability = self.p_min + (self.p_max - self.p_min) * (
            (best_fitness - max_observed_fitness) / (best_fitness - avg_fitness)
        ) if max_observed_fitness > avg_fitness else self.p_max

    def set_fitness_scores(self, sender, data_dict):
        population = sender.gen_population
        individuals = population.sub_populations[0].individuals

        fitness_scores = [ind.get_pure_fitness() for ind in individuals]
        self.best_fitness = max(fitness_scores)
        self.avg_fitness = sum(fitness_scores) / len(fitness_scores)
