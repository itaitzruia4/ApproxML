'''
Origin: Solving Blackjack with Q-Learning
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
'''


from __future__ import annotations
from typing import Tuple, SupportsFloat
import numpy as np
import gymnasium as gym

import pickle
import sys
import json

import utils
import multiprocessing
from blackjack_individual import BlackjackIndividual

from eckity.evaluators.simple_individual_evaluator \
    import SimpleIndividualEvaluator
from eckity.fitness.simple_fitness import SimpleFitness


class BlackjackEvaluator(SimpleIndividualEvaluator):
    def __init__(self,
                 arity=1,
                 events=None,
                 event_names=None,
                 n_episodes=100_000,
                 use_rewards=False):
        super().__init__(arity, events, event_names)
        self.n_episodes = n_episodes
        self.use_rewards = use_rewards
        if use_rewards:
            self.ids2rewards = multiprocessing.Manager().dict()

        # Dump this instance into a pickle file, which will later be used for evaluation
        with open('blackjack_evaluator.pkl', 'wb') as f:
            pickle.dump(self, f)

    def evaluate_individual(self, individual):
        vector = individual.get_vector()
        q_values = np.reshape(vector, utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)

        env = gym.make("Blackjack-v1", sab=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        
        rewards = np.zeros(utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE) if self.use_rewards else None

        for episode in range(self.n_episodes):
            obs, info = env.reset()
            done = False

            # play one episode
            while not done:
                # fix obs values to match genetic encoding of an individual
                obs = (int(obs[0]) - utils.MIN_PLAYER_SUM,
                       int(obs[1]) - utils.MIN_DEALER_CARD,
                       int(obs[2]))

                # always draw if the player sum is less than 12
                action = 1 if obs[0] < 0 else self.get_action(obs, env, q_values)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the rewards
                if self.use_rewards:
                    self.update(obs, action, reward, rewards)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

        env.close()
        if self.use_rewards:
            rewards = rewards.flatten()
            self.ids2rewards[individual.id] = rewards
        return np.mean(np.array(env.return_queue).flatten())

    def get_action(self, obs: Tuple[int, int, bool], env: gym.Env, q_values: np.ndarray) -> int:
        """
        Returns the best action.
        """
        return int(q_values[obs])

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: SupportsFloat,
        ind_rewards: np.ndarray
    ):
        ind_rewards[obs] += reward


def main():
    if len(sys.argv) < 3:
        print('Usage: python blackjack_evaluator.py <path_to_config_file> <individual_index>')
        sys.exit(1)

    config_path = sys.argv[1]
    idx = sys.argv[2]

    # Initialize the evaluator
    with open('blackjack_evaluator.pkl', 'rb') as f:
        blackjack_evaluator = pickle.load(f)

    # Parse the given individual, then evaluate it
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    vector = data[idx]

    ind = BlackjackIndividual(SimpleFitness(), length=len(vector), bounds=(0, 1))
    ind.set_vector(vector)
    fitness = blackjack_evaluator.evaluate_individual(ind)

    # Write the fitness to stdout
    print(fitness, flush=True)
    print(list(ind.get_rewards()))

if __name__ == '__main__':
    main()

