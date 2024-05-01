from sklearn.metrics.pairwise import cosine_similarity

from typing import List
from eckity.individual import Individual
from approx_ml_pop_eval import ApproxMLPopulationEvaluator

import numpy as np
from sampling_strat import SamplingStrategy


class CosSimSamplingStrategy(SamplingStrategy):
    def __init__(self) -> None:
        pass

    def sample(
        self,
        individuals: List[Individual],
        sample_size: int,
        preds: np.ndarray,
        evaluator: ApproxMLPopulationEvaluator
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

    def _get_cosine_scores(
        self,
        individuals: List[Individual],
        X: np.ndarray
    ) -> List[float]:
        """
        Compute the cosine similarity between each individual and the dataset

        Parameters
        ----------
        individuals : List[Individual]
            the population of the evolutionary experiment

        Returns
        -------
        List[float]
            list of cosine similarity scores
        """
        # matrix of size (sample_size, n_features)
        ind_vectors = [ind.vector for ind in individuals]

        # matrix of size (n_samples, sample_size)
        similarity_scores = cosine_similarity(X, ind_vectors)

        # obtain the maximum similarity score in the dataset
        # for each individual
        return np.max(similarity_scores, axis=0)
