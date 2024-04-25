import numpy as np
import sys

from sklearn.linear_model import Ridge, Lasso

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
    IntVectorNPointMutation,
)

from approx_ml_pop_eval import ApproxMLPopulationEvaluator
from cos_sim_switch_cond import CosSimSwitchCondition

from plot_statistics import PlotStatistics
import utils
from novelty_search import NoveltySearchCreator

from blackjack_evaluator import BlackjackEvaluator
from frozen_lake_evaluator import FrozenLakeEvaluator
from cliffwalking_evaluator import CliffWalkingEvaluator
from monstercliff_evaluator import MonsterCliffWalkingEvaluator


def main():
    """
    Basic setup.
    """
    if len(sys.argv) < 8:
        print(
            "Usage: python3 evoml_experiment.py <job_id> <problem> \
              <model> <switch_method> <threshold> <sample_strategy> <sample_rate>"
        )
        exit(1)

    job_id = sys.argv[1]
    problem = sys.argv[2]
    model = sys.argv[3]
    switch_method = sys.argv[4]
    threshold = float(sys.argv[5])
    sample_strategy = sys.argv[6]
    sample_rate = float(sys.argv[7])

    gen_weight = utils.sqrt_gen_weight

    norm_func = lambda y: np.log(-y)
    inv_norm_func = lambda y: -np.exp(y)

    novelty = "novelty" in sys.argv
    n_folds = 5 if switch_method == "error" else None
    handle_duplicates = "ignore"

    if problem == "blackjack":
        length = np.prod(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)
        ind_eval = BlackjackEvaluator(n_episodes=100_000)
        creator = GABitStringVectorCreator(length=length, bounds=(0, 1))
        mutation = BitStringVectorNFlipMutation(probability=0.3, n=length // 10)
        generations = 200

    elif problem == "frozenlake":
        length = utils.FROZEN_LAKE_STATES
        ind_eval = FrozenLakeEvaluator(total_episodes=2000)
        creator = GAIntVectorCreator(length=length, bounds=(0, 3))
        mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        generations = 50

    elif problem == "cliffwalking":
        length = utils.NUM_CLIFF_WALKING_STATES
        ind_eval = CliffWalkingEvaluator()
        creator = GAIntVectorCreator(length=length, bounds=(0, 3))
        mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        generations = 100

    elif problem == "monstercliff":
        length = utils.MONSTER_CLIFF_STATES
        ind_eval = MonsterCliffWalkingEvaluator(total_episodes=1000)
        creator = GAIntVectorCreator(length=length, bounds=(0, 3))
        mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
        generations = 100

    else:
        raise ValueError("Invalid problem " + problem)

    operators_sequence = [
        VectorKPointsCrossover(probability=0.7, k=2),
        mutation,
    ]

    if novelty:
        novelty_creator = NoveltySearchCreator(
            operators_sequence=operators_sequence,
            length=creator.length,
            bounds=creator.bounds,
            vector_type=creator.type,
            fitness_type=creator.fitness_type,
            k=20,
            max_archive_size=500,
        )
        del creator
        creator = novelty_creator

    # set model type and params
    if model == "ridge":
        model_type = Ridge
    elif model == "lasso":
        model_type = Lasso
    else:
        raise ValueError("Invalid model " + model)

    model_params = (
        {"alpha": 0.3, "max_iter": 3000}
        if model == "ridge"
        else (
            {"alpha": 0.65, "max_iter": 1000}
            if model == "lasso" and problem == "frozenlake"
            else (
                {"alpha": 0.6, "max_iter": 2000}
                if model == "ridge" and problem == "monstercliff"
                else {}  # default
            )
        )
    )

    # set switch condition
    if switch_method == "cosine":
        cos_sim = CosSimSwitchCondition(threshold=threshold, switch_once=False)

        def should_approximate(eval):
            return cos_sim.should_approximate(eval)

    elif switch_method == "error":

        def should_approximate(eval):
            return eval.approx_fitness_error < threshold

    elif switch_method == "false":

        def should_approximate(eval):
            return False

    else:
        raise ValueError("Invalid switch method " + switch_method)

    if sample_strategy not in ("random", "cosine"):
        raise ValueError("Invalid sampling strategy", sample_strategy)

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
                (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
            ],
        ),
        breeder=SimpleBreeder(),
        population_evaluator=ApproxMLPopulationEvaluator(
            population_sample_size=sample_rate,
            gen_sample_step=1,
            sample_strategy=sample_strategy,
            model_type=model_type,
            model_params=model_params,
            n_folds=n_folds,
            gen_weight=gen_weight,
            norm_func=norm_func,
            inv_norm_func=inv_norm_func,
            should_approximate=should_approximate,
            handle_duplicates=handle_duplicates,
            job_id=job_id,
        ),
        executor="process",
        max_workers=10,
        max_generation=generations,
        statistics=PlotStatistics(),
    )
    pop_eval = evoml.population_evaluator
    evoml.evolve()

    best_ind = evoml.best_of_run_
    print("Best fitness:", best_ind.get_pure_fitness())

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

    if switch_method == "cosine":
        print("cosine scores:", cos_sim.history)


if __name__ == "__main__":
    main()
