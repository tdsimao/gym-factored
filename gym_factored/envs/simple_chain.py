import numpy as np
from gym_factored.envs.base import DiscreteEnv

LEFT = 0
RIGHT = 1


class ChainEnv(DiscreteEnv):
    """
    this is a simple chain env using the DiscreteEnv class

    for a more general implementation of a chain env checkout gym/envs/toy_text/nchain.py

    """

    def __init__(self):
        ns = 4
        na = 2  # left and right
        terminal_states = np.zeros(ns, dtype=bool)
        terminal_states[0] = True
        terminal_states[ns-1] = True

        t = self.get_transition_function(na, ns)

        isd = np.zeros(ns)
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            if not terminal_states[s]:
                isd[s] = 1
            for a in range(na):
                info = {'cost': int(a == RIGHT)}
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    if transition_prob > 0:
                        if new_state == ns-1:
                            reward = 10
                        elif new_state == 0:
                            reward = 1
                        else:
                            reward = 0
                        done = terminal_states[new_state]
                        p[s][a].append((transition_prob, new_state, reward, done, info))
        isd /= isd.sum()
        DiscreteEnv.__init__(self, ns, na, p, isd)

    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))
        for s in range(1, ns-1):
            t[s, LEFT, left(s)] = 1
            t[s, RIGHT, right(s, ns)] = 1
        t[0, :, 0] = 1
        t[ns - 1, :, ns - 1] = 1
        return t

    def render(self, mode='human'):
        pass


def left(s):
    return max(s - 1, 0)


def right(s, ns):
    return min(s + 1, ns - 1)


class SlipperyChainEnv(ChainEnv):
    def get_transition_function(self, na, ns):
        suc_prob = 0.9
        fail_prob = 1 - suc_prob
        t = np.zeros((ns, na, ns))
        for s in range(1, ns-1):
            t[s, LEFT, left(s)] = suc_prob
            t[s, LEFT, right(s, ns)] = fail_prob
            t[s, RIGHT, right(s, ns)] = suc_prob
            t[s, RIGHT, left(s)] = fail_prob
        t[0, :, 0] = 1
        t[ns - 1, :, ns - 1] = 1
        return t


class NonAbsorbingChainEnv(ChainEnv):
    def get_transition_function(self, na, ns):
        t = super().get_transition_function(na, ns)
        t[0, 1, 0] = 0
        t[0, 1, 1] = 1
        t[ns - 1, 0, ns - 1] = 0
        t[ns - 1, 0, ns - 2] = 1
        return t
