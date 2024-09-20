import sys

import numpy as np
import utils
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
    IntVectorNPointMutation,
)
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.subpopulation import Subpopulation
from plot_statistics import PlotStatistics

from problems.blackjack.blackjack_evaluator import BlackjackEvaluator
from problems.frozen_lake.frozen_lake_evaluator import FrozenLakeEvaluator
from problems.monster_cliff_walking.monstercliff_evaluator import (
    MonsterCliffWalkingEvaluator,
)


def main():
    """
    Basic setup.
    """

    if len(sys.argv) < 3:
        print("Usage: python3 evo_experiment.py <job_id> <problem>")
        exit(1)

    problem = sys.argv[2]

    if problem == "blackjack":
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        creator = GABitStringVectorCreator(length=length, bounds=(0, 1))
        ind_eval = BlackjackEvaluator(n_episodes=100_000)
        mutation = BitStringVectorNFlipMutation(
            probability=0.3, n=length // 10
        )
        generations = 200

    elif problem == "frozenlake":
        length = utils.FROZEN_LAKE_STATES
        creator = GAIntVectorCreator(length=length, bounds=(0, 3))
        ind_eval = FrozenLakeEvaluator(total_episodes=2000)
        mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        generations = 50

    elif problem == "monstercliff":
        length = utils.MONSTER_CLIFF_STATES
        ind_eval = MonsterCliffWalkingEvaluator(total_episodes=1000)
        creator = GAIntVectorCreator(length=length, bounds=(0, 3))
        mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        generations = 100

    else:
        raise ValueError("Invalid problem " + problem)

    evo = SimpleEvolution(
        Subpopulation(
            creators=creator,
            population_size=100,
            evaluator=ind_eval,
            higher_is_better=True,
            elitism_rate=0.0,
            operators_sequence=[
                VectorKPointsCrossover(probability=0.7, k=2),
                mutation,
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=4, higher_is_better=True
                    ),
                    1,
                )
            ],
        ),
        breeder=SimpleBreeder(),
        max_generation=generations,
        executor="process",
        max_workers=10,
        statistics=PlotStatistics(),
    )
    evo.evolve()

    try:
        statistics = evo.statistics[0]
        statistics.plot_statistics()
    except Exception as e:
        print("Failed to print statistics. Error:", e)

    best_ind = evo.best_of_run_
    print("Best individual\n:", best_ind.vector)
    print("Best fitness:", best_ind.get_pure_fitness())


if __name__ == "__main__":
    main()
