from abc import ABC, abstractmethod

from ..approx_ml_pop_eval import ApproxMLPopulationEvaluator


class SwitchCondition(ABC):
    def __init__(self, threshold, switch_once) -> None:
        super().__init__()
        self.threshold = threshold

        self.switch_once = switch_once
        if switch_once:
            self.switched = False

        # used for statistics
        self.history = []

    @abstractmethod
    def should_approximate(self, evaluator: ApproxMLPopulationEvaluator) -> bool:
        """
        Check if the switch condition is satisfied

        Parameters
        ----------
        evaluator: ApproxMLPopulationEvaluator
            Population evaluator object

        Returns
        -------
        bool
            whether the switch condition is satisfied
        """
        raise NotImplementedError()
