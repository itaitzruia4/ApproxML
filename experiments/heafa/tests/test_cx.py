from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector

from ..operators.cx_reproduction import CXReproduction


def test_cx_reproduction():
    individuals = [
        BitStringVector(vector=[1, 1, 1, 0], fitness=SimpleFitness(0.1)),
        BitStringVector(vector=[0, 0, 1, 0], fitness=SimpleFitness(0.2)),
    ]
    cx = CXReproduction()
    cx.apply(individuals)
    assert len(cx.applied_individuals) == 2
    assert len(cx.points) == 1

    assert cx.applied_individuals[0].vector == [0, 1, 0]
    assert cx.applied_individuals[1].vector == [0, 0, 1]
