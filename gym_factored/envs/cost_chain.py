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
    A chain with costs
    """

    def __init__(self, prob_y_zero=0.1, n=3):
        self.domains = [range(n + 1), [0, 1]]
        self.ns = ns = 2 * (n + 1)
        na = 3

        isd = np.zeros(ns)
        isd[0:2] = [0.5, 0.5]
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            x, y = list(self.decode(s))
            for a in range(na):
                if x == n-1:
                    suc = self.encode(x + 1, y)
                    reward = 1
                    done = True
                    cost = 0
                    p[s][a].append((1, suc, reward, done, {'cost': cost}))
                elif x == n:
                    suc = s
                    reward = 0
                    done = True
                    cost = 0
                    p[s][a].append((1, suc, reward, done, {'cost': cost}))
                elif x == n-1:
                    suc = s
                    reward = 1
                    done = True
                    cost = 0
                    p[s][a].append((1, suc, reward, done, {'cost': cost}))
                elif a == RESET:
                    suc = self.encode(0, y)
                    reward = 0.01
                    done = False
                    cost = 0
                    p[s][a].append((1, suc, reward, done, {'cost': cost}))
                elif not (x % 2):
                    reward = 0
                    cost = 0
                    done = False
                    if prob_y_zero > 0:
                        p[s][a].append((prob_y_zero, self.encode(x+1, 0), reward, done, {'cost': cost}))
                    if prob_y_zero < 1:
                        p[s][a].append((1 - prob_y_zero, self.encode(x+1, 1), reward, done, {'cost': cost}))
                elif x % 2:
                    cost = int(a == A)
                    done = False
                    if y == 0:
                        reward = int(a == B) * 0.01
                    else:
                        reward = int(a == A) * 0.01
                    suc = self.encode(x + 1, y)
                    p[s][a].append((1, suc, reward, done, {'cost': cost}))
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
