import sys
import numpy as np
from six import StringIO

from gym_factored.envs.base import DiscreteEnv


# actions
A = 0
B = 1
RESET = 2


class CostChainEnv(DiscreteEnv):
    """
    A difficult CMDP environment
    """

    def __init__(self, prob_y_zero=0.1, n=3):
        self.domains = [range(n), [0, 1]]
        self.ns = ns = 6
        na = 3

        isd = np.zeros(2 * n)
        isd[0:2] = [0.5, 0.5]
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            x, y = list(self.decode(s))
            for a in range(na):
                if a == RESET:
                    done = False
                    info = {'cost': 0}
                    p[s][a].append((1, self.encode(0, y), 0, done, info))
                    continue
                if x == 0:
                    if prob_y_zero > 0:
                        p[s][a].append((prob_y_zero, self.encode(1, 0), 0, False, {}))
                    if 1 - prob_y_zero:
                        p[s][a].append((1 - prob_y_zero, self.encode(1, 1), 0, False, {}))
                elif x == 1:
                    cost = int(a == A)
                    info = {'cost': cost}
                    done = True
                    if y == 0:
                        p[s][a].append((1, self.encode(2, y), int(a == B), done, info))
                    else:
                        p[s][a].append((1, self.encode(2, y), int(a == A), done, info))
                else:
                    p[s][a].append((1, s, 0, True, {}))

        DiscreteEnv.__init__(self, ns, na, p, isd, domains=self.domains)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        res = [["»" for _ in self.domains[0]] for _ in self.domains[1]]
        x, y = self.decode(self.s)
        res[y][x] = "*"
        for line in res:
            outfile.write("".join(line) + "\n")
        outfile.write("last action: {}\n".format(self.lastaction) if self.lastaction is not None else "")
        if mode != 'human':
            return outfile
