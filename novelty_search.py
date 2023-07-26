import numpy as np

from overrides import override

from eckity.evaluators.individual_evaluator import IndividualEvaluator
from eckity.evaluators.simple_population_evaluator import SimplePopulationEvaluator
from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection


class NoveltySearchCreator(GAVectorCreator):
    def __init__(self,
                 operators_sequence,
                 length=1,
                 gene_creator=None,
                 bounds=(0.0, 1.0),
                 vector_type=BitStringVector,
                 fitness_type=SimpleFitness,
                 events=None,
                 k=15,
                 max_archive_size=1000):
        super().__init__(length=length, gene_creator=gene_creator,
                         bounds=bounds, vector_type=vector_type,
                         fitness_type=fitness_type, events=events)
        self.operators_sequence = operators_sequence
        self.individual_evaluator = NoveltySearchEvaluator(
            k=k,
            max_archive_size=max_archive_size
        )

    @override
    def create_individuals(self, n_individuals, higher_is_better):
        # create individuals with random vectors
        # higher_is_better = True because we are maximizing novelty
        individuals = super().create_individuals(n_individuals,
                                                 higher_is_better=True)

        # set original higher_is_better
        for ind in individuals:
            ind.fitness.higher_is_better = higher_is_better
        return individuals

    def novelty_search(self, individuals):
        evo_novelty = SimpleEvolution(
            Subpopulation(creators=None,
                          individuals=individuals,
                          population_size=100,
                          evaluator=self.individual_evaluator,
                          higher_is_better=True,
                          elitism_rate=0.0,
                          operators_sequence=self.operators_sequence,
                          selection_methods=[
                              # (selection method, selection probability) tuple
                              (TournamentSelection(tournament_size=4,
                                                   higher_is_better=True), 1)
                          ]),
            breeder=SimpleBreeder(),
            population_evaluator=SimplePopulationEvaluator(executor_method='submit'),
            max_generation=200,
            executor='process',
            max_workers=10
        )
        evo_novelty.evolve()
        individuals = evo_novelty.population.sub_populations[0].individuals
        return individuals


class NoveltySearchEvaluator(IndividualEvaluator):
    def __init__(self, k=15, max_archive_size=1000):
        self.archive = []
        self.k = k
        self.max_archive_size = max_archive_size

    def evaluate(self, individual, environment_individuals):
        idx = environment_individuals.index(individual)
        vectors = [ind.get_vector() for ind in environment_individuals]
        novelty_score = self.novelty_metric_genotypic(idx, individual, vectors)
        return novelty_score

    def insert_to_archive(self, individual, avg_nn):
        i = 0
        while i < len(self.archive):
            if avg_nn < self.archive[i][1]:
                break
            else:
                i += 1
        self.archive =\
            self.archive[:i] + [(individual, avg_nn)] + self.archive[i:]

    def add_to_archive(self, individual, avg_nn):
        # add point to archive if it is greater than minimum
        # archive is maintained sorted by avg_nn: [(point1, avg_nn1),  (point2, avg_nn2),... ]

        if len(self.archive) == 0:
            self.archive.append((individual, avg_nn))
        elif len(self.archive) < self.max_archive_size:
            self.insert_to_archive(individual, avg_nn)
        else:
            # archive is full
            if avg_nn > self.archive[0][1]:
                # greater than minimum
                del self.archive[0]
                self.insert_to_archive(individual, avg_nn)

    def novelty_metric_genotypic(self, i, individual, vectors):
        vec_i = np.array(vectors[i])
        dists = [np.linalg.norm(vec_i - np.array(vectors[j]))
                 for j in range(len(vectors)) if j != i]
        dists += [np.linalg.norm(vec_i - np.array(self.archive[j][0].vector))
                  for j in range(len(self.archive))]
        dists = sorted(dists)

        avg_nn = sum(dists[:self.k])/self.k     # average distance to nearest neighbors
        self.add_to_archive(individual, avg_nn)
        return avg_nn
