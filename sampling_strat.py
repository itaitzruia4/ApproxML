from abc import ABC, abstractmethod

import numpy as np
from typing import List
from eckity.individual import Individual


class SamplingStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        evaluator  # no type annotation to avoid circular imports
    ) -> List[Individual]:
        """
        Sample individuals from the population according to the given strategy

        Parameters
        ----------
        individuals: List[Individual]
            current generation individuals
        sample_size: int
            number of individuals to sample
        preds : np.ndarray
            ML model predictions on current generation
        evaluator : ApproxMLPopulationEvaluator
            for any additional data on the population, dataset etc.

        Returns
        -------
        List[Individual]
            sampled individuals, list of length `sample_size`
        """
        raise NotImplementedError()
