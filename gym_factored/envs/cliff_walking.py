import numpy as np
import sys
from contextlib import closing
from io import StringIO
from gym_factored.envs.base import DiscreteEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingCostEnv(DiscreteEnv):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (4, 12)
        self.start_state_index = self.encode(3, 0)
        ns = int(np.prod(self.shape))
        na = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities, rewards and costs
        transitions = {}
        for s in range(ns):
            position = self.decode(s)
            transitions[s] = {a: [] for a in range(na)}
            transitions[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            transitions[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            transitions[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            transitions[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(ns)
        isd[self.start_state_index] = 1.0

        super().__init__(ns, na, transitions, isd)


    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """

        prob = 1.0
        new_position = np.array(current) + np.array(delta)
        (row, col) = new_position = self._limit_coordinates(new_position).astype(int)

        info = {
            'cost': self._get_cost(row, col)
        }
        if self._cliff[tuple(new_position)]:
            new_state = self.start_state_index
            is_done = False
            reward = -100
        else:
            new_state = self.encode(*new_position)
            terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            is_done = tuple(new_position) == terminal_state
            reward = -1
        return [(prob, new_state, reward, is_done, info)]


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = self.decode(s)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def decode(self, s):
        return np.unravel_index(s, self.shape)

    def encode(self, *values):
        return np.ravel_multi_index(tuple(values), self.shape)

    def _get_cost(self, row, col):
        if row in [0, self.shape[0] - 1]:
            return 0
        if col in [0, self.shape[1] - 1]:
            return 0
        return row
