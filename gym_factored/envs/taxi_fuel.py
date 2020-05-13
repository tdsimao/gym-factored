"""
code from [RLBase repo][https://github.com/canmanietp/RLBase] by @canmanietp with minor changes.
"""
import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np


MAPS = {
    "5x5": [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ],
    "6x6": [
        "+-----------+",
        "|R: | : :G: |",
        "| : | : : : |",
        "| : : : : : |",
        "| | : | : : |",
        "|Y| : |B: : |",
        "| | : | : : |",
        "+-----------+",
    ],
    "7x7": [
        "+-------------+",
        "|R: | : :G: : |",
        "| : | : : : : |",
        "| : : : : : : |",
        "| | : | : : : |",
        "|Y| : |B: : : |",
        "| | : | : : : |",
        "| | : | : : : |",
        "+-------------+",
    ]
}


class TaxiFuelEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow).
    The taxi starts off at a random square and the passenger is at one of the designated locations.
    The taxi must:
        - drive to the passenger's location
        - pick up the passenger
        - drive to the passenger's destination (another one of the four designated locations), and
        - drop off the passenger.
    Once the passenger is dropped off, the episode ends.

    In the TaxiFuel variation, the taxi also has a fuel level that decreases after each action.
    Running out of fuel gives a reward of -20 and terminates the episode.
    To avoid this problem, the agent has an extra action that lets it refuel in a specific location.

    Observations:
    In the default map (5x5), there are 7000 discrete states since there are:
        - 25 taxi positions
        - 5 possible locations of the passenger (including the case when the passenger is the taxi)
        - 4 destination locations
        - 14 fuel levels


    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    - 6: refuel

    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger.
    There is a reward of -10 for executing actions "pickup", "dropoff" or refuel illegally.


    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, fuel_capacity=14, map_name="5x5"):
        assert map_name in MAPS.keys(), "invalid map_name.\nValid names: {}".format(", ".join(MAPS.keys()))
        self.desc = np.asarray(MAPS[map_name], dtype='c')

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.fuel_capacity = fuel_capacity
        self.fuel_location = (3, 2)

        nR, nC = map_name.split('x')
        self.nR, self.nC = nR, nC = int(nR), int(nC)
        min_starting_fuel = 2 + (nR - 3) + (nC - 3)

        nS = nR * nC * (len(locs) + 1) * len(locs) * fuel_capacity
        isd = np.zeros(nS)
        nA = 7
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for state in range(nS):
            row, col, pass_idx, dest_idx, fuel = self.decode(state)
            if pass_idx < 4 and pass_idx != dest_idx and fuel >= min_starting_fuel:
                isd[state] += 1
            for a in range(nA):
                new_row, new_col, new_pass_idx, new_fuel = row, col, pass_idx, fuel
                reward = -1
                done = False
                taxiloc = (row, col)
                new_fuel -= 1

                if a == 0:
                    new_row = min(row + 1, nR - 1)
                elif a == 1:
                    new_row = max(row - 1, 0)
                elif a == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                    new_col = min(col + 1, nC - 1)
                elif a == 3 and self.desc[1 + row, 2 * col] == b":":
                    new_col = max(col - 1, 0)
                elif a == 4:  # pickup
                    if pass_idx < 4 and taxiloc == locs[pass_idx]:
                        new_pass_idx = 4
                    else:
                        reward = -10
                elif a == 5:  # dropoff
                    if (taxiloc == locs[dest_idx]) and pass_idx == 4:
                        new_pass_idx = dest_idx
                        done = True
                        reward = 20
                    elif (taxiloc in locs) and pass_idx == 4:
                        new_pass_idx = locs.index(taxiloc)
                    else:
                        reward = -10
                elif a == 6:  # refuel
                    if taxiloc == self.fuel_location:
                        new_fuel = self.fuel_capacity - 1
                    else:
                        reward = -10

                if new_fuel <= 0:
                    new_fuel = 0
                    reward = -20
                    done = True

                newstate = self.encode(new_row, new_col, new_pass_idx, dest_idx, new_fuel)
                P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def step(self, a):
        state, reward, done, info = super().step(a)
        info['suc'] = self.goal_state(state)
        info['fail'] = self.out_of_fuel(state)
        return state, reward, done, info

    def goal_state(self, state):
        dec_state = list(self.decode(state))
        return dec_state[2] == dec_state[3]

    def out_of_fuel(self, state):
        dec_state = list(self.decode(state))
        return dec_state[4] == 0

    def encode(self, taxirow, taxicol, passloc, destidx, fuel):
        i = taxirow
        i *= self.nC
        i += taxicol
        i *= (len(self.locs) + 1)
        i += passloc
        i *= (len(self.locs))
        i += destidx
        i *= self.fuel_capacity
        i += fuel
        return i

    def decode(self, i):
        out = [i % self.fuel_capacity]
        i = i // self.fuel_capacity
        out.append(i % len(self.locs))
        i = i // len(self.locs)
        out.append(i % (len(self.locs) + 1))
        i = i // (len(self.locs) + 1)
        out.append(i % self.nC)
        i = i // self.nC
        out.append(i)
        assert 0 <= i < self.nR
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx, fuelidx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        out[1 + self.fuel_location[0]][2 * self.fuel_location[1] + 1] = "F"

        if passidx < 4:
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(out[1 + taxirow][2 * taxicol + 1], 'yellow',
                                                               highlight=True)
            pi, pj = self.locs[passidx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(ul(out[1 + taxirow][2 * taxicol + 1]), 'green',
                                                               highlight=True)

        di, dj = self.locs[destidx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.lastaction is not None:
            action_names = ["South", "North", "East", "West", "Pickup", "Dropoff", "Refuel"]
            outfile.write("{}  ({})\n".format(fuelidx, action_names[self.lastaction]))
        else:
            outfile.write("\n\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

