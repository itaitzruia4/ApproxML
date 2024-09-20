from overrides import override

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class HEAFACreator(GAVectorCreator):
    def __init__(self,
				 length=1,
				 gene_creator=None,
				 bounds=(0, 1),
                 vector_type=BitStringVector,
                 population_evaluator=None,
                 individual_evaluator=None,
				 events=None):
        super().__init__(length=length,
						 gene_creator=gene_creator,
						 bounds=bounds,
						 vector_type=vector_type,
						 events=events)
        self.population_evaluator = population_evaluator
        self.individual_evaluator = individual_evaluator
        self.initialized = False
        
    @override
    def create_individuals(self, n_individuals, higher_is_better):
        if self.initialized:
            # creation for population restart strategy
            return super().create_individuals(n_individuals, higher_is_better)
        
        # otherwise, initial creation of individuals

        # create twice as many individuals as population size
        individuals = super().create_individuals(n_individuals * 2,
                                                 higher_is_better)

        # evaluate the population
        fitness_scores = self.population_evaluator.evaluate_individuals(
            individuals,
            self.individual_evaluator
        )

        for ind, fitness in zip(individuals, fitness_scores):
            ind.fitness.set_fitness(fitness)

        # sort the individuals by fitness
        individuals.sort(key=lambda x: x.get_pure_fitness(),
                         reverse=higher_is_better)
        
        self.initialized = True
        
        # return the top n individuals
        return individuals[:n_individuals]
