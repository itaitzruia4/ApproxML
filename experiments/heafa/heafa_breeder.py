from overrides import override

from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection


class HEAFABreeder(SimpleBreeder):
    '''
    Breeder for HEAFA algorithm.

    This breeder first applies elitism selection to the population, then
    selects pairs of individuals from the population at random and applies
    the operators to them.
    '''
    @override
    def apply_breed(self, population):
        for subpopulation in population.sub_populations:
            nextgen_population = []

            num_elites = subpopulation.n_elite
            if num_elites > 0:
                elitism_sel = ElitismSelection(
                    num_elites=num_elites,
                    higher_is_better=subpopulation.higher_is_better
                )
                elitism_sel.apply_operator((subpopulation.individuals,
                                            nextgen_population))

            assert all(ind.fitness._is_evaluated
                       for ind in subpopulation.individuals)

            population_size = subpopulation.population_size
            self.selected_individuals = []
            selection_method = subpopulation.get_selection_methods()[0][0]
            operators = subpopulation.get_operators_sequence()
            while len(nextgen_population) < population_size - num_elites:
                # select individuals from the population
                parents = selection_method.select(subpopulation.individuals,
                                                  nextgen_population)

                for operator in operators:
                    operator.parents_fitness = [
                        p.get_pure_fitness() for p in parents
                    ]

                # then runs all operators on selected parents
                for operator in operators:
                    offspring = operator.apply_operator(parents)

                    # remove duplicates
                    for off in offspring:
                        off_vec = off.get_vector()
                        for ind in nextgen_population:
                            if ind.get_vector() == off_vec:
                                break
                        nextgen_population.append(off)

                self.selected_individuals.extend(parents)

            subpopulation.individuals = nextgen_population

    @override
    def _apply_operators(self, operator_seq, individuals_to_apply_on):
        for operator in operator_seq:
            offspring = super()._apply_operators([operator], individuals_to_apply_on)

            # remove duplicates
            offspring = list(set(offspring))
