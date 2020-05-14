import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text.discrete import categorical_sample
import sys
import numpy as np
from six import StringIO

FORWARD = 0
BACKWARD = 1
FALL = 2
ACTIONS = ['FORWARD', 'BACKWARD', 'FALL']


class BridgeEnv(gym.Env):
    """
    The bridge problem.
        Fatemi, M.; Sharma, S.; Van Seijen, H.; and Kahou, S. E. 2019.
        Dead-ends and secure exploration in reinforcement learning.
        Proceedings of the 36th International Conference on Machine Learning, 1873–1881. PMLR.
    """

    def __init__(self, bridge_len, max_swimming_len):
        self.bridge_len = bridge_len
        self.max_swimming_len = max_swimming_len

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(bridge_len * max_swimming_len)
        self.actions_probs = {
            FORWARD: [.99, 0., .01],
            BACKWARD: [0, 1, 0],
            FALL: [0, 0, 1]
        }
        self.rewards = {
            'positive': 100.0,
            'negative': -1.0,
            'step': 0.0
        }

        self.pos_x = 0
        self.swim_steps = 0
        self.swimming_limit = 0

        self.last_action = None
        self.np_random = None
        self.reset()

    def reset(self):
        self.last_action = None
        self.pos_x = 0
        self.swim_steps = 0
        self.swimming_limit = np.random.randint(2, self.max_swimming_len)
        return self._get_observation()

    def step(self, action):
        # assert action in self.action_space, 'Illegal action.'
        self.last_action = action
        reward, done = self._move(action)
        info = {
            'suc': self.pos_x == (self.bridge_len - 1),
            'fail': self.swim_steps == self.swimming_limit
        }
        return self._get_observation(), reward, done, info

    def _move(self, action):
        reward = self.rewards['step']
        done = False
        if self.swim_steps == 0:
            # from behaviour-prob of the taken action
            if action == FORWARD:
                action_effect = categorical_sample(self.actions_probs[action], self.np_random)
            else:
                action_effect = self.actions_probs[action].index(1)

            if action_effect == FORWARD:
                self.pos_x += 1
                if self.pos_x == (self.bridge_len - 1):
                    reward = self.rewards['positive']
                    done = True
            elif action_effect == BACKWARD:
                if self.pos_x != 0:
                    self.pos_x -= 1
            elif action_effect == FALL:
                self.swim_steps = 1
        else:
            if self.swim_steps == self.swimming_limit:
                reward = self.rewards['negative']
                done = True
            else:
                self.swim_steps += 1
        return reward, done

    def _get_observation(self):
        return self.encode(self.pos_x, self.swim_steps)

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
        for i in range(self.max_swimming_len):
            bridge_river_map.append(["≈" for _ in range(self.bridge_len)])
        bridge_river_map[self.swim_steps][self.pos_x] = "*"
        for line in bridge_river_map:
            outfile.write("".join(line) + "\n")
        outfile.write("last action: {}\n".format(ACTIONS[self.last_action]) if self.last_action is not None else "")
        if mode != 'human':
            return outfile

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]