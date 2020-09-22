from gym.envs.toy_text import discrete


class DiscreteEnv(discrete.DiscreteEnv):
    def __init__(self, *args, domains=None, **kwargs):
        self.domains = domains
        super().__init__(*args, **kwargs)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, info = transitions[i]
        self.s = s
        self.lastaction = a
        return s, r, d, info

    def encode(self, *args):
        res = 0
        for value, domain in zip(args, self.domains):
            res *= len(domain)
            res += value
        return res

    def decode(self, i):
        out = [i % len(self.domains[1])]
        i = i // len(self.domains[1])
        out.append(i)
        assert 0 <= i < len(self.domains[0])
        return reversed(out)

    def render(self, mode='human'):
        super().render(mode)
