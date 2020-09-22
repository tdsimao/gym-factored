from gym_factored.envs.base import DiscreteEnv
import numpy as np
import sys
from six import StringIO


class StockTradingEnv(DiscreteEnv):
    """
    The StockTrading Problem
        Strehl, A. L.; Diuk, C.; and Littman, M. L. 2007.
        Efficient Structure Learning in Factored-State MDPs.
        Proceedings of the Twenty-Second AAAI Conference on Artificial Intelligence, 645–650. AAAI Press.
    """

    def __init__(self, number_of_sectors, number_of_stocks_per_sector):
        self.number_of_sectors = number_of_sectors
        self.stocks_per_sector = number_of_stocks_per_sector
        self.number_of_factors = self.number_of_sectors * (self.stocks_per_sector + 1)
        self.nS = nS = 2 ** self.number_of_factors
        self.nA = nA = 2 * self.number_of_sectors + 1
        isd = np.zeros(nS)
        isd[0] = 1  # starts owing nothing and no stock is rising
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            state_factors = list(self.decode(s))

            # state factors pattern
            #
            # i=self.number_of_sectors
            # j=self.number_of_stocks_per_sector
            #
            # (own_sector_1, stock_1.1, stock_1.2, ..., stock_1.j,
            #  own_sector_2, stock_2.1, stock_2.2, ..., stock_2.j,
            # ...
            #  own_sector_i, stock_i.1, stock_i.2, ..., stock_i.j)

            for a in range(nA):
                prob_true = np.zeros(self.number_of_sectors * (self.stocks_per_sector + 1))
                reward = 0
                for sector in range(self.number_of_sectors):

                    own_sector = sector * (self.stocks_per_sector + 1)
                    if a == sector:
                        # buy sector
                        prob_true[own_sector] = 1
                    elif a == sector + self.number_of_sectors:
                        # sell sector
                        prob_true[own_sector] = 0
                    else:
                        # do nothing or exchange another sector
                        # prob(own_sector|do_nothing) = own_sector
                        prob_true[own_sector] = int(state_factors[own_sector])

                    sectors_stocks = self.stocks_from_sector(sector)
                    number_of_rising_stocks_in_sector = sum(state_factors[stock] for stock in sectors_stocks)
                    if state_factors[own_sector]:
                        reward += number_of_rising_stocks_in_sector
                        reward -= (len(sectors_stocks) - number_of_rising_stocks_in_sector)

                    for stock in sectors_stocks:
                        prob_true[stock] = 0.1 + (0.8 * number_of_rising_stocks_in_sector) / len(sectors_stocks)
                total_p = 0
                for ns in range(nS):
                    next_state_factors = list(self.decode(ns))
                    p = 1
                    for factor, value in enumerate(next_state_factors):
                        if value:
                            p *= prob_true[factor]
                        else:
                            p *= (1 - prob_true[factor])
                        if p == 0:
                            break
                    if p > 0:
                        total_p += p
                        P[s][a].append((p, ns, reward, False, {}))
                assert abs(1 - total_p) < 0.0000001
        isd /= isd.sum()
        DiscreteEnv.__init__(self, nS, nA, P, isd)

    def stocks_from_sector(self, sector):
        sectors_stocks = range(sector * (self.stocks_per_sector + 1) + 1, (sector + 1) * (self.stocks_per_sector + 1))
        return sectors_stocks

    def encode(self, *statuses):
        i = 0
        assert self.number_of_factors == len(statuses)
        for factor in range(self.number_of_factors):
            if statuses[factor]:
                i += 2 ** factor
        return i

    def decode(self, s):
        assert 0 <= s < self.nS
        out = bin(s)[2:].rjust(self.number_of_factors, '0')
        return map(int, reversed(out))

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        state_factors = list(self.decode(self.s))

        def ul(x):
            return "O" if x else "X"

        state_line = "".join([ul(x) for x in state_factors])
        outfile.write(state_line + "\n")
        if self.lastaction is not None:
            action_line = "".join(
                "^" if i == self.lastaction * (self.stocks_per_sector + 1) else
                "v" if i == (self.lastaction - self.number_of_sectors) * (self.stocks_per_sector + 1) else
                " " for i in range(self.number_of_factors)
            )
        else:
            action_line = ""
        outfile.write(action_line + "\n")

        if mode != 'human':
            return outfile
