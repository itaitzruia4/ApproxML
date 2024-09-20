import numpy as np
import sys

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.creators.ga_creators.bit_string_vector_creator \
    import GABitStringVectorCreator
from eckity.genetic_operators.selections.tournament_selection \
    import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover \
    import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation \
    import BitStringVectorNFlipMutation, IntVectorNPointMutation

from inheritance_population_evaluator import InheritancePopulationEvaluator

from plot_statistics import PlotStatistics
import examples.utils as utils

from problems.blackjack.blackjack_evaluator import BlackjackEvaluator
from problems.frozen_lake.frozen_lake_evaluator import FrozenLakeEvaluator
from monstercliff_evaluator import MonsterCliffWalkingEvaluator


def main():
    """
    Basic setup.
    """
    if len(sys.argv) < 4:
        print('Usage: python3 evoml_experiment.py <job_id> <problem> \
              <sample_rate>')
        exit(1)

    job_id = sys.argv[1]
    problem = sys.argv[2]
    sample_rate = float(sys.argv[3])

    sample_strategy = 'random'
    handle_duplicates = 'ignore'

    weighted = 'weighted' in sys.argv

    if problem == 'blackjack':
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        ind_eval = BlackjackEvaluator(n_episodes=100_000)
        creator = GABitStringVectorCreator(length=length,
                                           bounds=(0, 1),
                                           update_parents=True)
        mutation = BitStringVectorNFlipMutation(probability=0.3, n=length//10)
        generations = 200

    elif problem == 'frozenlake':
        length = utils.FROZEN_LAKE_STATES
        ind_eval = FrozenLakeEvaluator(total_episodes=2000)
        creator = GAIntVectorCreator(length=length,
                                     bounds=(0, 3),
                                     update_parents=True)
        mutation = IntVectorNPointMutation(probability=0.3, n=length//10)
        generations = 50

    elif problem == 'monstercliff':
        length = utils.MONSTER_CLIFF_STATES
        ind_eval = MonsterCliffWalkingEvaluator(total_episodes=1000)
        creator = GAIntVectorCreator(length=length,
                                     bounds=(0, 3),
                                     update_parents=True)
        mutation = IntVectorNPointMutation(probability=0.3, n=length//10)
        generations = 100

    else:
        raise ValueError('Invalid problem ' + problem)

    selection = TournamentSelection(tournament_size=4,
                                    higher_is_better=True)

    operators_sequence = [
        VectorKPointsCrossover(probability=0.7, k=2),
        mutation,
    ]

    evoml = SimpleEvolution(
        Subpopulation(creators=creator,
                      population_size=100,
                      evaluator=ind_eval,
                      higher_is_better=True,
                      elitism_rate=0.0,
                      operators_sequence=operators_sequence,
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (selection, 1)
                      ]),
        breeder=SimpleBreeder(),
        population_evaluator=InheritancePopulationEvaluator(
                weighted=weighted,
                population_sample_size=sample_rate,
                gen_sample_step=1,
                sample_strategy=sample_strategy,
                handle_duplicates=handle_duplicates,
                job_id=job_id),
        executor='process',
        max_workers=10,
        max_generation=generations,
        statistics=PlotStatistics(),
    )
    pop_eval = evoml.population_evaluator

    # register population evaluator to update fitness scores post-selection
    selection.register('after_operator', pop_eval.update_fitness_and_genomes)

    evoml.evolve()

    best_ind = evoml.best_of_run_
    print('Best fitness:', best_ind.get_pure_fitness())

    try:
        statistics = evoml.statistics[0]
        statistics.plot_statistics()
    except Exception as e:
        print('Failed to print statistics. Error:', e)

    try:
        print('dataset samples:', pop_eval.df.shape[0])
        pop_eval.export_dataset(utils.DATASET_PATH)
    except Exception as e:
        print('Failed to export dataset. Error:', e)

    print('fitness computations:', pop_eval.fitness_count)
    print('approximations:', pop_eval.approx_count)


if __name__ == "__main__":
    main()
