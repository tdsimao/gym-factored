from gym.envs.toy_text import discrete


class DiscreteEnv(discrete.DiscreteEnv):
    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, info = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, info)
