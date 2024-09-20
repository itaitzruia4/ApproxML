from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector

from ..operators.random_selection import RandomSelection


def test_random_selection():
    source_inds = [
        BitStringVector(vector=[0, 0, 1], fitness=SimpleFitness(0.1)),
        BitStringVector(vector=[0, 1, 0], fitness=SimpleFitness(0.2)),
    ]
    dest_inds = []
    selection = RandomSelection()
    selection.select(source_inds, dest_inds)
    assert len(dest_inds) == 1
    assert len(selection.selected_individuals) == 1

    selected_ind = selection.selected_individuals[0]
    assert selected_ind in source_inds
    assert selected_ind in dest_inds

    source_inds_vectors = [ind.get_vector() for ind in source_inds]
    assert selected_ind.vector in source_inds_vectors

    source_inds_fitness = [ind.get_pure_fitness() for ind in source_inds]
    assert selected_ind.get_pure_fitness() in source_inds_fitness
