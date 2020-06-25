import numpy as np
from gym_factored.envs.base import DiscreteEnv

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


class Chain2DEnv(DiscreteEnv):
    """
    this is a simple chain env using the DiscreteEnv class with two dimensions

    for a more general implementation of a chain env checkout gym/envs/toy_text/nchain.py

    """

    def __init__(self):
        self.size = 4
        self.ns = ns = 16
        na = 4  # left and right
        terminal_states = self.terminal_states = self.get_terminal_states()

        t = self.get_transition_function(na, ns)

        isd = np.zeros(ns)
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            if not terminal_states[s]:
                isd[s] = 1
            for a in range(na):
                info = {'cost': self.get_cost(s, a)}
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    if transition_prob > 0:
                        reward = self.get_reward(s, a, new_state)
                        done = terminal_states[new_state]
                        p[s][a].append((transition_prob, new_state, reward, done, info))
        isd /= isd.sum()
        DiscreteEnv.__init__(self, ns, na, p, isd)

    def get_reward(self, s, a, new_state):
        if self.terminal_states[s]:
            return 0
        x, y = list(self.decode(new_state))
        if a in [UP, DOWN]:
            if y in [0, self.size-1]:
                return 1
        if a in [LEFT, RIGHT]:
            if x == 0:
                if y == 1:
                    return 2
                if y == 2:
                    return 3
            if x == self.size-1:
                if y == 1:
                    return 10
                if y == 2:
                    return -10
        return 0

    def get_cost(self, s, a):
        x, _ = list(self.decode(s))
        cost = {
            RIGHT: [0, 2],
            LEFT: [1, 0],
            UP: [0, 0],
            DOWN: [0, 0],
        }
        if self.terminal_states[s]:
            return 0
        else:
            return cost[a][x - 1]


    def get_terminal_states(self):
        terminal_states = np.zeros(self.ns, dtype=bool)
        for i in range(4):
            terminal_states[self.encode(0, i)] = True
            terminal_states[self.encode(self.size - 1, i)] = True
            terminal_states[self.encode(i, 0)] = True
            terminal_states[self.encode(i, self.size - 1)] = True
        return terminal_states

    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))
        for s in range(1, ns-1):
            t[s, LEFT, self.left(s)] = 1
            t[s, RIGHT, self.right(s)] = 1
            t[s, UP, self.up(s)] = 1
            t[s, DOWN, self.down(s)] = 1
        for s in np.arange(self.ns)[self.terminal_states]:
            t[s, :, s] = 1
        return t

    def render(self, mode='human'):
        pass

    def encode(self, x, y):
        return y * self.size + x

    def decode(self, i):
        out = [i % self.size]
        i = i // self.size
        out.append(i)
        assert 0 <= i < self.size
        return out

    def left(self, s):
        x, y = list(self.decode(s))
        return self.encode(max(x - 1, 0), y)

    def right(self, s):
        x, y = list(self.decode(s))
        return self.encode(min(x + 1, self.size - 1), y)

    def up(self, s):
        x, y = list(self.decode(s))
        return self.encode(x, max(y - 1, 0))

    def down(self, s):
        x, y = list(self.decode(s))
        return self.encode(x, min(y + 1, self.size - 1))


class SlipperyChain2DEnv(Chain2DEnv):
    def get_transition_function(self, na, ns):
        suc_prob = 0.9
        fail_prob = 1 - suc_prob
        t = np.zeros((ns, na, ns))
        for s in range(1, ns-1):
            t[s, LEFT, self.left(s)] = suc_prob
            t[s, LEFT, self.right(s)] = fail_prob

            t[s, RIGHT, self.right(s)] = suc_prob
            t[s, RIGHT, self.left(s)] = fail_prob

            t[s, DOWN, self.down(s)] = suc_prob
            t[s, DOWN, self.up(s)] = fail_prob

            t[s, UP, self.up(s)] = suc_prob
            t[s, UP, self.down(s)] = fail_prob

        for s in np.arange(self.ns)[self.terminal_states]:
            t[s, :, s] = 1

        return t


class NonAbsorbingChain2DEnv(Chain2DEnv):
    def get_terminal_states(self):
        return np.zeros(self.ns, dtype=bool)
