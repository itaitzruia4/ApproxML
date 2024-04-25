"""
Origin: Monster Cliff Walking
https://github.com/Sebastian-Griesbach/MonsterCliffWalking
"""

import gymnasium as gym
from typing import List
import numpy as np
import math

from monstercliffwalking import unpair
import utils

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

CLIFF_REWARD = -100
MONSTER_REWARD = -1000
GOAL_POS = (3, 11)


class MonsterCliffWalkingEvaluator(SimpleIndividualEvaluator):
    def __init__(self, arity=1, events=None, event_names=None, total_episodes=1000):
        super().__init__(arity, events, event_names)
        self.total_episodes = total_episodes

        gym.register(
            id="MonsterCliffWalking-v0",
            entry_point="monstercliffwalking:MonsterCliffWalkingEnv",
        )

        self.env = gym.make("MonsterCliffWalking-v0")

    def evaluate_individual(self, individual):
        vector = individual.get_vector().copy()

        score_sum = 0
        for episode in range(self.total_episodes):
            score = 0
            obs = self.env.reset()[0]
            done = False
            n_steps = 0

            while not done:
                action = self.choose_action(obs, vector)

                # Take the action and observe the outcome state and reward
                obs, reward, terminated, truncated, _ = self.env.step(action)
                n_steps += 1

                reward = reward if reward != MONSTER_REWARD else -50
                score += reward

                done = terminated or truncated or n_steps == 1000

            score_sum += score

        return score_sum / self.total_episodes

    def choose_action(self, obs: int, vector: List[int]) -> int:
        """
        Returns the best action.
        """
        number = self.env.unwrapped.state_to_szudzik[obs]
        state = unpair(number, n=2)
        idx = np.ravel_multi_index(state, utils.MONSTER_CLIFF_SPACE_SHAPE)
        return int(vector[idx])

    def state_distance(self, state):
        """
        Calculate the euclidian distance between the final player position and the goal

        Parameters
        ----------
        state : int
            final state of the agent
        """
        # convert state from int to (x, y) format
        number = self.env.unwrapped.state_to_szudzik[state]
        player_state, _ = unpair(number, n=2)
        pos = np.unravel_index(player_state, utils.CLIFF_WALKING_MAP_SHAPE)
        return math.dist(pos, GOAL_POS)

    def terminate(self):
        self.env.close()
