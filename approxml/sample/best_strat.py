from typing import List

import numpy as np
import pandas as pd
from eckity.individual import Individual

from .sampling_strat import SamplingStrategy


class BestSamplingStrategy(SamplingStrategy):
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        population_dataset: pd.DataFrame,
    ):
        ind_preds = zip(individuals, preds)
        sorted_inds = [
            ind
            for ind, _ in sorted(ind_preds, key=lambda x: x[1], reverse=True)
        ]
        sample_inds = sorted_inds[:sample_size]
        return sample_inds
