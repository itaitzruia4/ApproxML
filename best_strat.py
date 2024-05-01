from typing import List
from eckity.individual import Individual
from approx_ml_pop_eval import ApproxMLPopulationEvaluator

import numpy as np
from sampling_strat import SamplingStrategy


class BestSamplingStrategy(SamplingStrategy):
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        evaluator: ApproxMLPopulationEvaluator,
    ):
        ind_preds = zip(individuals, preds)
        sorted_inds = [
            ind for ind, _ in sorted(
                ind_preds, key=lambda x: x[1], reverse=True
            )
        ]
        sample_inds = sorted_inds[:sample_size]
        return sample_inds
