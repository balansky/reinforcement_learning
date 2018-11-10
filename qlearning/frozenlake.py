import gym
import numpy as np
import random

class FrozenLake(object):

    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))


    def train(self, episodes=35000, learning_rate=0.8, max_episode_step=99, gamma=0.95):
        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.005
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = 0
            for step in range(max_episode_step):
                exp_exp_tradeoff = random.uniform(0, 1)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state, :])
                else:
                    action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                self.qtable[state, action] = self.qtable[state, action] + \
                                             learning_rate*(reward + gamma*np.max(self.qtable[next_state, :]) -
                                                            self.qtable[state, action])
                state = next_state
                episode_rewards += reward
                if done:
                    break
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
            rewards.append(episode_rewards)
        print("Score over time: " + str(sum(rewards) / episodes))


    def play(self):


        state = self.env.reset()
        print("****************************************************")

        for step in range(99):

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(self.qtable[state, :])

            new_state, reward, done, info = self.env.step(action)

            self.env.render()

            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)

                # We print the number of step it took.
                print("Number of steps", step)
                break
            state = new_state

    def __del__(self):
        self.env.close()
