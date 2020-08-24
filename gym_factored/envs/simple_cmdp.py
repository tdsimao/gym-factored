import numpy as np
from gym_factored.envs.base import DiscreteEnv

STAY = 0
MOVE = 1


class CMDPEnv(DiscreteEnv):
    """
    A simple CMDP environment inspired by
     Zheng, L. & Ratliff, L.. (2020). Constrained Upper Confidence Reinforcement Learning.
     Proceedings of the 2nd Conference on Learning for Dynamics and Control, in PMLR 120:620-629
    """

    def __init__(self, ns=3):
        self.ns = ns
        na = 2
        terminal_states = self.get_terminal_states()

        t = self.get_transition_function(na, ns)

        isd = np.full(ns, fill_value=1./ns)
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            for a in range(na):
                info = {'cost': int(a == MOVE)}
                reward = int((a == MOVE) * (s + 1))
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    if transition_prob > 0:
                        done = terminal_states[new_state]
                        p[s][a].append((transition_prob, new_state, reward, done, info))
        DiscreteEnv.__init__(self, ns, na, p, isd)

    def get_terminal_states(self):
        return np.zeros(self.ns, dtype=bool)

    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))
        states = np.arange(ns)
        next_states = np.roll(states, -1)
        t[states, STAY, states] = 1
        t[states, MOVE, next_states] = 1
        return t

    def render(self, mode='human'):
        pass

    def encode(self, x):
        return x

    def decode(self, i):
        return reversed([i])

