from typing import Callable

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from typing_extensions import override

from approxml.approx_ml_pop_eval import ApproxMLPopulationEvaluator


class CVErrorSwitchCondition:
    def __init__(
        self,
        threshold: float = 0.1,
        switch_once: bool = False,
        n_splits: int = 5,
        scoring: Callable = mean_squared_error,
    ):
        super().__init__()
        self.threshold = threshold
        self.scoring = scoring
        self.n_splits = n_splits

        self.switch_once = switch_once
        if switch_once:
            self.switched = False

        # used for statistics
        self.history = []

    @override
    def should_approximate(self, evaluator: ApproxMLPopulationEvaluator):
        if self.switch_once and self.switched:
            return True

        X, y = evaluator.get_X_y()
        fit_params = evaluator.get_fit_params()

        # Perform KFold CV to estimate the fitness error of the model
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        scorer = make_scorer(self.scoring)
        cross_val_scores = cross_val_score(
            evaluator.model, X, y, cv=kf, fit_params=fit_params, scoring=scorer
        )
        cv_error = cross_val_scores.mean()
        self.history.append(cv_error)
        return cv_error < self.threshold
