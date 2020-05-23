from gym.envs.toy_text.discrete import DiscreteEnv
from gym.envs.toy_text.discrete import categorical_sample
import sys
import numpy as np
from six import StringIO

FORWARD = 0
BACKWARD = 1
FALL = 2
ACTIONS = ['FORWARD', 'BACKWARD', 'FALL']


class BridgeEnv(DiscreteEnv):
    """
    The bridge problem.
        Fatemi, M.; Sharma, S.; Van Seijen, H.; and Kahou, S. E. 2019.
        Dead-ends and secure exploration in reinforcement learning.
        Proceedings of the 36th International Conference on Machine Learning, 1873–1881. PMLR.
    """

    def __init__(self, bridge_len, max_swimming_len):
        self.bridge_len = bridge_len
        self.max_swimming_len = max_swimming_len
        nS = self.bridge_len * self.max_swimming_len
        isd = np.zeros(nS)
        isd[0] = 1
        nA = 3
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for state in range(nS):
            pos_x, swim_step = list(self.decode(state))
            for a in range(nA):
                if swim_step == 0:
                    if a == FORWARD:
                        new_x = min(pos_x + 1, self.bridge_len - 1)
                        s = self.encode(new_x, swim_step)
                        if new_x == self.bridge_len - 1:
                            P[state][a].append((0.99, s, 100, True, {"suc": True}))
                        else:
                            P[state][a].append((0.99, s, 0, False, {}))
                        s = self.encode(pos_x, 1)
                        P[state][a].append((0.01, s, 0, False, {}))
                    elif a == BACKWARD:
                        s = self.encode(max(pos_x - 1, 0), swim_step)
                        P[state][a].append((1.0, s, 0, False, {}))
                    else:
                        s = self.encode(pos_x, 1)
                        P[state][a].append((1.0, s, 0, False, {}))
                elif swim_step < 2:
                    s = self.encode(pos_x, swim_step + 1)
                    P[state][a].append((1.0, s, 0, False, {}))
                else:
                    prob_drawn = 1.0/(self.max_swimming_len - swim_step)
                    s = self.encode(pos_x, min(swim_step, max_swimming_len))
                    P[state][a].append((prob_drawn, s, -1, True, {'fail': True}))

                    if prob_drawn < 1:
                        s = self.encode(pos_x, min(swim_step + 1, max_swimming_len))
                        P[state][a].append((1.0 - prob_drawn, s, 0, False, {}))

                for p, _, _, d, info in P[state][a]:
                    info["prob"] = p
                    info["suc"] = info.get("suc", False)
                    info["fail"] = info.get("fail", False)
        DiscreteEnv.__init__(self, nS, nA, P, isd)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, info = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, info)

    def encode(self, pos_x, swim_step):
        return pos_x * self.max_swimming_len + swim_step

    def decode(self, i):
        out = [i % self.max_swimming_len]
        i = i // self.max_swimming_len
        out.append(i)
        assert 0 <= i < self.bridge_len
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        bridge_river_map = [["=" for _ in range(self.bridge_len)]]
        pos_x, swim_steps = self.decode(self.s)
        for i in range(self.max_swimming_len):
            bridge_river_map.append(["≈" for _ in range(self.bridge_len)])
        bridge_river_map[swim_steps][pos_x] = "*"
        for line in bridge_river_map:
            outfile.write("".join(line) + "\n")
        outfile.write("last action: {}\n".format(ACTIONS[self.lastaction]) if self.lastaction is not None else "")
        if mode != 'human':
            return outfile
