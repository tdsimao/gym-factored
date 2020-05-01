import unittest
import gym
import gym_factored


class TestStockTradingEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:stock-trading-v0")

    def test_reset_env(self):
        state = self.env.reset()
        self.assertEqual(state, 0)

    def test_episode_length(self):
        self.env.reset()
        episode_length = 0
        done = False
        while not done:
            _, _, done, _ = self.env.step(self.env.action_space.sample())
            episode_length += 1
        self.assertEqual(40, episode_length)

    def test_decode(self):
        initial_state = self.env.reset()
        decoded_initial_state = list(self.env.decode(initial_state))
        self.assertListEqual(decoded_initial_state, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_rendering(self):
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            _, _, done, _ = self.env.step(action)
            out = self.env.render(mode='ansi')
            lines = out.getvalue().split('\n')
            if action in range(3):
                self.assertEqual('^', lines[1][action*3])
                self.assertEqual('O', lines[0][action*3])
            elif action in range(3, 6):
                self.assertIn('v', lines[1][(action - 3)*3])
                self.assertIn('X', lines[0][(action - 3)*3])
            else:
                self.assertEqual(' ' * 9, lines[1])

