import random
from typing import List

import numpy as np
import pandas as pd

from eckity.individual import Individual

from .sampling_strat import SamplingStrategy


class RandomSamplingStrategy(SamplingStrategy):
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        population_dataset: pd.DataFrame,
    ):
        return random.sample(individuals, sample_size)
