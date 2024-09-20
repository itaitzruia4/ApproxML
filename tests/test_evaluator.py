import numpy as np
import pandas as pd

from approxml import ApproxMLPopulationEvaluator


def test_approximate_individuals(self, df, individuals, rnd_ind_eval):
    eval = ApproxMLPopulationEvaluator(sample_rate=0)
    eval.df = df
    old_approx_count = eval.approx_count
    eval._approximate_individuals(individuals, rnd_ind_eval)
    new_approx_count = eval.approx_count
    assert new_approx_count == old_approx_count + len(individuals)


def test_evaluate_individuals(self, individuals, rnd_ind_eval):
    eval = ApproxMLPopulationEvaluator()
    old_fitness_count = eval.fitness_count
    old_dataset_size = eval.df.shape[0]
    eval.evaluate_individuals(individuals, rnd_ind_eval)
    new_fitness_count = eval.fitness_count
    new_dataset_size = eval.df.shape[0]
    assert new_fitness_count == old_fitness_count + len(individuals)
    assert new_dataset_size == old_dataset_size + len(individuals)
