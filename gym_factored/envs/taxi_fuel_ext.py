from gym_factored.envs.taxi_fuel import TaxiFuelEnv


class TaxiFuelExtEnv(TaxiFuelEnv):
    """
    This is an extension of the taxi-fuel environment where the agents only gets the final reward when refueling
     after having dropped-off the passenger.
    """

    def __init__(self, fuel_capacity=14, map_name="5x5"):
        super(TaxiFuelExtEnv, self).__init__(fuel_capacity, map_name)

    def get_transition(self, a, state):
        row, col, pass_idx, dest_idx, fuel = self.decode(state)
        new_row, new_col, new_pass_idx, new_fuel = row, col, pass_idx, fuel
        reward = -1
        done = False
        taxiloc = (row, col)
        new_fuel -= 1
        if a == 0:
            new_row = min(row + 1, self.nR - 1)
        elif a == 1:
            new_row = max(row - 1, 0)
        elif a == 2 and self.desc[1 + row, 2 * col + 2] == b":":
            new_col = min(col + 1, self.nC - 1)
        elif a == 3 and self.desc[1 + row, 2 * col] == b":":
            new_col = max(col - 1, 0)
        elif a == 4:  # pickup
            if pass_idx < 4 and taxiloc == self.locs[pass_idx]:
                new_pass_idx = 4
            else:
                reward = -10
        elif a == 5:  # dropoff
            if (taxiloc in self.locs) and pass_idx == 4:
                new_pass_idx = self.locs.index(taxiloc)
            else:
                reward = -10
        elif a == 6:  # refuel
            if taxiloc == self.fuel_location:
                new_fuel = self.fuel_capacity - 1
                if pass_idx == dest_idx:
                    done = True
                    reward = 20
            else:
                reward = -10
        if new_fuel <= 0:
            new_fuel = 0
            reward = -20
            done = True
        newstate = self.encode(new_row, new_col, new_pass_idx, dest_idx, new_fuel)
        return done, newstate, reward

    def goal_state(self, state):
        dec_state = list(self.decode(state))
        return (dec_state[2] == dec_state[3]) and dec_state[4] == (self.fuel_capacity - 1)
