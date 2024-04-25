from approx_ml_pop_eval import ApproxMLPopulationEvaluator
import numpy as np
import utils

from sklearn.metrics.pairwise import cosine_similarity


class CosSimSwitchCondition:
    """
    Switch condition based on the cosine similarity between the current
    generation and the population dataset.

    Parameters
    ----------
    threshold : float
        Threshold for the maximum average cosine similarity
        between the current generation and the population dataset
    switch_once : bool
        If True, the switch will only occur once, and the condition will
        always return True afterwards
    """

    def __init__(self, threshold=0.5, switch_once=False):
        super().__init__()
        self.threshold = threshold

        self.switch_once = switch_once
        if switch_once:
            self.switched = False

        self.history = []

    def should_approximate(self, evaluator: ApproxMLPopulationEvaluator):
        # If switch_once is enabled and the switch already occured,
        # always approximate
        if self.switch_once and self.switched:
            return True

        individuals = evaluator.gen_population.sub_populations[0].individuals

        # Get the evaluator dataset with all the gens except the current one
        df = evaluator.df[evaluator.df["gen"] < evaluator.gen]

        genotypes = df.iloc[:, :-2]
        ind_vectors = [ind.vector for ind in individuals]

        similarity_scores = cosine_similarity(genotypes, ind_vectors)

        # compute the average of the similarity between the current
        # generation and their most similar rows in the dataset
        max_scores = np.max(similarity_scores, axis=0)
        avg_score = np.mean(max_scores)
        result = avg_score >= self.threshold

        self.history.append(avg_score)

        if self.switch_once:
            self.switched = result

        return result
