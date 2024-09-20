import numpy as np
import pandas as pd
import pytest
from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.float_vector import FloatVector


class RandomIndividualEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(ind):
        return np.random.rand()


@pytest.fixture
def individuals():
    return [
        FloatVector(fitness=SimpleFitness(), cells=list(np.random.rand(3)))
        for _ in range(10)
    ]


@pytest.fixture
def df():
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    df = pd.DataFrame(X)
    df["fitness"] = y
    df["gen"] = 0
    return df


@pytest.fixture
def rnd_ind_eval():
    return RandomIndividualEvaluator()
