import random
from typing import List

import numpy as np
from eckity.individual import Individual

from ..approx_ml_pop_eval import ApproxMLPopulationEvaluator
from .sampling_strat import SamplingStrategy


class RandomSamplingStrategy(SamplingStrategy):
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        evaluator: ApproxMLPopulationEvaluator,
    ):
        return random.sample(individuals, sample_size)
