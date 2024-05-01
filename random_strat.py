from typing import List
from eckity.individual import Individual

import numpy as np
from sampling_strat import SamplingStrategy


class RandomSamplingStrategy(SamplingStrategy):
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        evaluator   # no type annotation to avoid circular imports
    ):
        X, _ = evaluator.get_X_y()
        cosine_scores = self._get_cosine_scores(individuals, X)
        ind_scores = zip(individuals, cosine_scores)

        # sample the individuals with the lowest cosine score
        # e.g. individuals that are least similar to the dataset
        sample_inds = [
            ind for ind, _ in sorted(
                ind_scores, key=lambda x: x[1]
            )[:sample_size]
        ]
        return sample_inds
