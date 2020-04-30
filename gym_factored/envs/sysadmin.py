from gym.envs.toy_text import discrete
import numpy as np
import sys
from six import StringIO

PROB_MACHINE_FAIL = 0.05


class SysAdminEnv(discrete.DiscreteEnv):
    """
    The SysAdmin Problem
        Guestrin, C.; Koller, D.; Parr, R.; and Venkataraman, S. 2003.
        Efficient Solution Algorithms for Factored MDPs.
        Journal of Artificial Intelligence Research 19:399–468.
    """

    def __init__(self, size):
        self.nM = size
        n_m = size
        n_s = 2 ** n_m
        n_a = n_m + 1
        isd = np.zeros(n_s)
        isd[n_s-1] = 1
        P = {s: {a: [] for a in range(n_a)} for s in range(n_s)}
        for s in range(n_s):
            machines_on = list(self.decode(s))
            for a in range(n_a):
                prob_on = np.zeros(n_m)
                for m in range(n_m):
                    if a == m:
                        prob_on[m] = 1
                    elif not machines_on[m]:
                        prob_on[m] = 0
                    else:
                        fail_prob = PROB_MACHINE_FAIL
                        for neighbor in self.neighbor_of(m):
                            if not machines_on[neighbor]:
                                fail_prob += 0.3
                        prob_on[m] = max(1 - fail_prob, 0)
                total_p = 0
                for ns in range(n_s):
                    reward = sum(machines_on) - (a < n_m)
                    p = 1
                    next_machines_on = list(self.decode(ns))
                    for m, status in enumerate(next_machines_on):
                        if status:
                            p *= prob_on[m]
                        else:
                            p *= (1-prob_on[m])
                    if p > 0:
                        total_p += p
                        P[s][a].append((p, ns, reward, False))
                assert abs(1 - total_p) < 0.0000001
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, n_s, n_a, P, isd)

    def neighbor_of(self, m):
        return [(m-1) % self.nM, (m+1) % self.nM]

    def encode(self, *statuses):
        i = 0
        assert self.nM == len(statuses)
        for m in range(self.nM):
            if statuses[m]:
                i += 2 ** m
        return i

    def decode(self, i):
        out = bin(i)[2:].rjust(self.nM, '0')
        return map(int, reversed(out))

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        machines_on = list(self.decode(self.s))

        def ul(x):
            return "O" if x else "X"

        out = "".join([ul(x) for x in machines_on])

        outfile.write(out+"\n")
        if self.lastaction is not None:
            action_line = "".join("^" if i == self.lastaction else " " for i in range(self.nM + 1))
        else:
            action_line = ""
        outfile.write(action_line + "\n")

        if mode != 'human':
            return outfile
