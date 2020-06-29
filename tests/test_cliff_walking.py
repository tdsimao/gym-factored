import unittest
import gym


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class TestCliffWalking(unittest.TestCase):
    def test_make(self):
        cost_env = gym.make("gym_factored:cliff_walking_cost-v0")
        cost_env.seed(42)
        original_env = gym.make("CliffWalking-v0")
        original_env.seed(42)

        for i in range(3):
            with self.subTest():
                state_a = cost_env.reset()
                state_b = original_env.reset()
                self.assertEqual(state_a, state_b)
                done_a = False
                while not done_a:
                    action = cost_env.action_space.sample()
                    state_a, reward_a, done_a, info_a = cost_env.step(action)
                    state_b, reward_b, done_b, info_b = original_env.step(action)
                    self.assertEqual(state_a, state_b)
                    self.assertEqual(reward_a, reward_b)
                    if not info_a.get('TimeLimit.truncated', False):
                        self.assertEqual(done_a, done_b)

    def test_cost(self):
        env = gym.make("gym_factored:cliff_walking_cost-v0")
        env.reset()
        _, _, _, info = env.step(RIGHT)
        self.assertEqual(info['cost'], 0)
        _, _, _, info = env.step(UP)
        self.assertEqual(info['cost'], 0)
        _, _, _, info = env.step(RIGHT)
        self.assertEqual(info['cost'], 2)
        _, _, _, info = env.step(UP)
        self.assertEqual(info['cost'], 1)
        _, _, _, info = env.step(UP)
        self.assertEqual(info['cost'], 0)

    def test_cost_right(self):
        env = gym.make("gym_factored:cliff_walking_cost-v0")
        env.reset()
        env.unwrapped.s = env.encode(0, 10)
        _, _, _, info = env.step(RIGHT)
        self.assertEqual(info['cost'], 0)
        _, _, _, info = env.step(DOWN)
        self.assertEqual(info['cost'], 0)
        _, _, _, info = env.step(DOWN)
        self.assertEqual(info['cost'], 0)
        _, _, _, info = env.step(DOWN)
        self.assertEqual(info['cost'], 0)

    def test_4_by_3_initial_state(self):
        env = gym.make("gym_factored:cliff_walking_cost-v0", num_cols=3)
        state = env.reset()
        self.assertEqual(state, 9)
        state, reward, done, info = env.step(RIGHT)
        self.assertEqual(state, 9)
        self.assertEqual(reward, -100)
        self.assertEqual(info.get('cost'), 0)
        self.assertFalse(done)

    def test_4_by_3_goal_state(self):
        env = gym.make("gym_factored:cliff_walking_cost-v0", num_cols=3)
        env.reset()
        env.unwrapped.s = env.encode(3, 2)
        state, reward, done, info = env.step(DOWN)
        self.assertEqual(state, 11)
        self.assertEqual(reward, -1)
        self.assertTrue(done)
        self.assertEqual(info.get('cost'), 0)

    def test_5_by_3_goal_state(self):
        env = gym.make("gym_factored:cliff_walking_cost-v0", num_cols=3, num_rows=5)
        env.reset()
        env.unwrapped.s = env.encode(0, 1)
        state, reward, done, info = env.step(DOWN)
        self.assertEqual(info.get('cost'), 1)
        state, reward, done, info = env.step(DOWN)
        self.assertEqual(info.get('cost'), 2)
        state, reward, done, info = env.step(DOWN)
        self.assertEqual(info.get('cost'), 3)
        state, reward, done, info = env.step(DOWN)
        self.assertEqual(state, 12)
        self.assertEqual(info.get('cost'), 0)
