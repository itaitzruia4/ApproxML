
from random import choices
from eckity.genetic_operators.selections.selection_method import SelectionMethod


class RandomSelection(SelectionMethod):
    '''
    Random selection operator. Selects individuals at random from the population.
    '''
    def __init__(self, arity=1, events=None):
        super().__init__(arity=arity, events=events)

    def select(self, source_inds, dest_inds):
        # sample n_selected pairs of individuals from source individuals
        selected = [ind.clone()
                    for ind in choices(source_inds, k=self.arity)]

        dest_inds.extend(selected)
        self.selected_individuals = selected

        return selected
