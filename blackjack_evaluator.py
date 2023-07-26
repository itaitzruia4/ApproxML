'''
Origin: Solving Blackjack with Q-Learning
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
'''


from __future__ import annotations
from typing import Tuple, SupportsFloat
import numpy as np
import gymnasium as gym

import pickle

import utils
import multiprocessing

from eckity.evaluators.simple_individual_evaluator \
    import SimpleIndividualEvaluator
from eckity.fitness.simple_fitness import SimpleFitness


class BlackjackEvaluator(SimpleIndividualEvaluator):
    def __init__(self,
                 arity=1,
                 events=None,
                 event_names=None,
                 n_episodes=100_000):
        super().__init__(arity, events, event_names)
        self.n_episodes = n_episodes

    def evaluate_individual(self, individual):
        vector = individual.get_vector()
        q_values = np.reshape(vector, utils.BLACKJACK_STATE_ACTION_SPACE_SHAPE)

        env = gym.make("Blackjack-v1", sab=True)
        env = gym.wrappers.RecordEpisodeStatistics(env,
                                                   deque_size=self.n_episodes)

        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            done = False

            # play one episode
            while not done:
                # fix obs values to match genetic encoding of an individual
                obs = (int(obs[0]) - utils.MIN_PLAYER_SUM,
                       int(obs[1]) - utils.MIN_DEALER_CARD,
                       int(obs[2]))

                # always draw if the player sum is less than 12
                action = 1 if obs[0] < 0 else self.get_action(obs,
                                                              env,
                                                              q_values)
                next_obs, _, terminated, truncated, _ = env.step(action)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

        env.close()
        return np.mean(np.array(env.return_queue).flatten())

    def get_action(self, obs: Tuple[int, int, bool], env: gym.Env, q_values: np.ndarray) -> int:
        """
        Returns the best action.
        """
        return int(q_values[obs])
