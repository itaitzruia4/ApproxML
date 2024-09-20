from eckity.genetic_operators.genetic_operator import GeneticOperator

from random import choices


class CellExchangeMutation(GeneticOperator):
    def __init__(self, probability=1, arity=1, events=None, p_min=0.01, p_max=0.2):
        super().__init__(probability=probability, arity=arity, events=events)
        self.applied_individuals = None
        self.p_min = p_min
        self.p_max = p_max
        self.best_fitness = None
        self.avg_fitness = None
        self.parents_fitness = []

    def apply(self, individuals):
        self.applied_individuals = individuals
        self.adapt_probability()

        for i in range(len(individuals)):
            ind = individuals[i]

            # select two random indices
            indices = choices(range(len(individuals[i].vector)), k=2)

            # swap the values at the two indices
            ind.vector[indices[0]], ind.vector[indices[1]] =\
                ind.vector[indices[1]], ind.vector[indices[0]]

        return individuals

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

