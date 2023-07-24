from approx_ml_pop_eval import ApproxMLPopulationEvaluator
import numpy as np
import utils


class CosSimSwitchCondition:
    '''
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
    '''
    def __init__(self, threshold=0.5, switch_once=False):
        super().__init__()
        self.threshold = threshold

        self.switch_once = switch_once
        if switch_once:
            self.switched = False

    def should_approximate(self, evaluator: ApproxMLPopulationEvaluator):
        # If switch_once is enabled and the switch already occured,
        # always approximate
        if self.switch_once and self.switched:
            return True
        
        individuals = evaluator.gen_population.sub_populations[0].individuals
        
        # Get the evaluator dataset with all the gens except the current one
        df = evaluator.df[evaluator.df['gen'] < evaluator.gen]

        genotypes = df.iloc[:, :-2]
        scores = [
            genotypes.apply(lambda row: utils.cosine_similarity(ind.vector, row), axis=1, raw=True).max()
            for ind in individuals
        ]

        avg_score = np.mean(scores)
        result = avg_score >= self.threshold

        if self.switch_once:
            self.switched = result

        return result


