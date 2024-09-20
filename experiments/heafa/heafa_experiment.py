import sys

import numpy as np
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector
from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
    IntVectorNPointMutation,
)
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.subpopulation import Subpopulation

import examples.utils as utils
from approxml.approx_ml_pop_eval import ApproxMLPopulationEvaluator
from sampling_strategies.best_strat import BestSamplingStrategy
from problems.blackjack.blackjack_evaluator import BlackjackEvaluator
from cell_exchange_mutation import CellExchangeMutation
from cx_reproduction import CXReproduction
from heafa.elm import ELM
from problems.frozen_lake.frozen_lake_evaluator import FrozenLakeEvaluator
from heafa_breeder import HEAFABreeder
from heafa_creator import HEAFACreator
from heafa_population_evaluator import HEAFAPopulationEvaluator
from monstercliff_evaluator import MonsterCliffWalkingEvaluator
from plot_statistics import PlotStatistics
from heafa.random_selection import RandomSelection
from random_strat import RandomSamplingStrategy


def main():
    """
    Basic setup.
    """
    if len(sys.argv) < 4:
        print(
            "Usage: python3 heafa_experiment.py <job_id> <problem> \
              <sample_rate>"
        )
        exit(1)

    job_id = sys.argv[1]
    problem = sys.argv[2]
    sample_rate = float(sys.argv[3])

    use_our_config = "our" in sys.argv or "ours" in sys.argv

    sampling_strat = (
        RandomSamplingStrategy() if use_our_config else BestSamplingStrategy()
    )
    handle_duplicates = "ignore"

    if problem == "blackjack":
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        ind_eval = BlackjackEvaluator(n_episodes=100_000)
        generations = 200
        if use_our_config:
            creator = GABitStringVectorCreator(length=length, bounds=(0, 1))
            mutation = BitStringVectorNFlipMutation(probability=0.3, n=length // 10)
        else:
            creator = HEAFACreator(
                length=length, bounds=(0, 1), vector_type=BitStringVector
            )
            mutation = CellExchangeMutation(probability=0.2)

    elif problem == "frozenlake":
        length = utils.FROZEN_LAKE_STATES
        ind_eval = FrozenLakeEvaluator(total_episodes=2000)
        generations = 50
        if use_our_config:
            creator = GAIntVectorCreator(length=length, bounds=(0, 3))
            mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        else:
            creator = HEAFACreator(length=length, bounds=(0, 3), vector_type=IntVector)
            mutation = CellExchangeMutation(probability=0.2)

    elif problem == "monstercliff":
        length = utils.MONSTER_CLIFF_STATES
        ind_eval = MonsterCliffWalkingEvaluator(total_episodes=1000)
        generations = 100
        if use_our_config:
            creator = GAIntVectorCreator(length=length, bounds=(0, 3))
            mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        else:
            creator = HEAFACreator(length=length, bounds=(0, 3), vector_type=IntVector)
            mutation = CellExchangeMutation(probability=0.2)

    else:
        raise ValueError("Invalid problem " + problem)

    crossover = (
        VectorKPointsCrossover(probability=0.7, k=2)
        if use_our_config
        else CXReproduction(probability=0.9)
    )
    operators_sequence = [crossover, mutation]

    model_params = {
        "input_size": creator.length,
        "hidden_size": 100
    }

    breeder = SimpleBreeder() if use_our_config else HEAFABreeder()
    selection = (
        TournamentSelection(tournament_size=4)
        if use_our_config
        else RandomSelection(arity=2)
    )

    if use_our_config:
        pop_eval = ApproxMLPopulationEvaluator(
            sample_rate=sample_rate,
            gen_sample_step=1,
            sampling_strategy=sampling_strat,
            model_type=ELM,
            model_params=model_params,
            should_approximate=lambda _: True,
            handle_duplicates=handle_duplicates,
            job_id=job_id,
        )
    else:
        pop_eval = HEAFAPopulationEvaluator(
            population_sample_size=sample_rate,
            model_params=model_params,
            gen_sample_step=1,
            sample_strategy=sampling_strat,
            handle_duplicates=handle_duplicates,
            job_id=job_id,
        )

    evoml = SimpleEvolution(
        Subpopulation(
            creators=creator,
            population_size=100,
            evaluator=ind_eval,
            higher_is_better=True,
            elitism_rate=0.0,
            operators_sequence=operators_sequence,
            selection_methods=[
                # (selection method, selection probability) tuple
                (selection, 1)
            ],
        ),
        breeder=breeder,
        population_evaluator=pop_eval,
        executor="process",
        max_workers=10,
        max_generation=generations,
        statistics=PlotStatistics(),
    )

    if not use_our_config:
        creator.population_evaluator = pop_eval
        creator.individual_evaluator = ind_eval

        # update initial population
        pop_eval.gen_population = evoml.population

        for gen_oper in operators_sequence:
            pop_eval.register("after_operator", gen_oper.set_fitness_scores)

    evoml.evolve()

    if use_our_config:
        best_ind = evoml.best_of_run_
        print("Best fitness:", best_ind.get_pure_fitness())
    else:
        pop_eval.print_best_of_run()

    try:
        statistics = evoml.statistics[0]
        statistics.plot_statistics()
    except Exception as e:
        print("Failed to print statistics. Error:", e)

    try:
        print("dataset samples:", pop_eval.df.shape[0])
        pop_eval.export_dataset(utils.DATASET_PATH)
    except Exception as e:
        print("Failed to export dataset. Error:", e)

    print("fitness computations:", pop_eval.fitness_count)
    print("approximations:", pop_eval.approx_count)


if __name__ == "__main__":
    main()
