import gym
import torchvision
import torch
from collections import deque, namedtuple
from utils.policy_rewards import discount_episode_rewards, normailize_rewards
import copy
import time

import numpy as np
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQNetwork(torch.nn.Module):

    def __init__(self, action_size):
        super(DQNetwork, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 128, 4, 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU()
        )
        self.v = torch.nn.Sequential(
            torch.nn.Linear(1152, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 1)
        )
        self.advantage_v = torch.nn.Sequential(
            torch.nn.Linear(1152, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, action_size)

        )

    def forward(self, input):
        h = self.block(input)
        h = h.view(-1, 1152)
        v = self.v(h)
        adv_v = self.advantage_v(h)
        q_v = v + (adv_v - torch.mean(adv_v, dim=1, keepdim=True))
        return q_v


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class JourneyEscape(object):

    def __init__(self, stack_size=4, device=torch.device("cpu")):
        self.env = gym.make("JourneyEscape-v0")
        self.action_space = self.env.action_space.n
        self.device = device
        self.stack_size = stack_size
        self.resize_fn = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop((160, 160)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((84, 84)),
                torchvision.transforms.ToTensor()
            ])
        self.net = DQNetwork(self.action_space).to(device)
        self.net.apply(self.init_weights)
        self.memory = None
        self.actions = [a for a in range(self.action_space)]
        self.model_path = "models/escape.pt"

    def init_weights(self, m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_normal(m.weight)

    def __del__(self):
        self.env.close()

    def stack_states(self, raw_state, stacked_states=None):
        state = self.resize_fn(raw_state)
        if stacked_states:
            stacked_states.append(state)
        else:
            stacked_states = deque([state for _ in range(self.stack_size*3)], maxlen=self.stack_size*3)
        torch_state = torch.cat([stacked_states[s] for s in range(2, self.stack_size*3+2, 3)], 0).unsqueeze(0).to(self.device)
        return torch_state, stacked_states

    def do_explore(self, step_done, decay_rate, min_ep):
        exp_exp_tradeoff = np.random.rand()
        explore_probability = min_ep + (1.0 - min_ep) * np.exp(-decay_rate * step_done)
        if (explore_probability > exp_exp_tradeoff):
            return True
        else:
            return False

    def select_action(self, state, step_done):
        if self.do_explore(step_done, 0.00004, 0.05):
            action = torch.tensor([[random.choice(self.actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.net(state).max(1)[1].view(1, 1)
        return action

    def optimize_model(self, optimizer, batch_size, target_net, gamma=0.95):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], 0)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        # loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss

    def play(self):
        self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path))
        self.env.reset()
        done = False
        state, stacked_states = self.stack_states(self.env.render(mode='rgb_array'))
        while not done:
            with torch.no_grad():
                action = self.net(state).max(1)[1].view(1, 1)
            next_raw_state, reward, done, info = self.env.step(action.item())
            self.env.render()
            time.sleep(0.1)
            if reward < 0:
                done = True

            if not done:
                state, stacked_states = self.stack_states(next_raw_state, stacked_states)


    def train(self, learning_rate=0.0002, max_iterations=50000, max_episode_steps=1000, batch_size=128,
              memory_capacity=30000, target_updates=1000):
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        target_net = copy.deepcopy(self.net).to(self.device)
        target_net.load_state_dict(self.net.state_dict())
        target_net.eval()
        self.memory = ReplayMemory(memory_capacity)
        steps = 0
        rewards = []
        agg_losses = []
        while True:
            self.env.reset()
            state, stacked_states = self.stack_states(self.env.render(mode='rgb_array'))
            episode_rewards = []
            episode_states = []
            episode_actions = []
            for _ in range(max_episode_steps):
                if len(self.memory) == 0:
                    print("Filling Samples into Replay Memory...")
                action = self.select_action(state, steps)
                next_raw_state, reward, done, info = self.env.step(action.item())
                if reward < 0:
                    reward = -100.
                    done = True
                else:
                    reward = 1.
                rewards.append(reward)
                episode_rewards.append(reward)
                episode_actions.append(action)
                # reward = torch.tensor([reward], device=self.device)
                if not done:
                    next_state, stacked_states = self.stack_states(next_raw_state, stacked_states)
                    episode_states.append((state, next_state))
                else:
                    next_state = None
                    episode_states.append((state, next_state))
                    discounted_rewards = discount_episode_rewards(0.95, episode_rewards)
                    discounted_rewards = normailize_rewards(discounted_rewards)
                    for i in range(len(discounted_rewards)):

                        self.memory.push(episode_states[i][0], episode_actions[i], episode_states[i][1],
                                         torch.tensor([discounted_rewards[i]], device=self.device))
                state = next_state
                if len(self.memory) > batch_size:
                    if steps == 0:
                        print("Start Training...")
                    loss = self.optimize_model(optimizer, batch_size, target_net)
                    agg_losses.append(loss.item())
                    if steps % 1000 == 0 or steps == max_iterations:
                        print("[%d] Loss: %.4f, Mean Rewards: %.4f" % (steps, sum(agg_losses)/len(agg_losses), sum(rewards)/len(rewards)))
                        agg_losses = []
                        rewards = []

                    if steps % target_updates == 0:
                        target_net.load_state_dict(self.net.state_dict())
                    steps += 1
                if done:
                    break
            if steps == max_iterations:
                break
        print("Finish Training...")


    # def train(self, learning_rate=0.0002, max_episodes=50000, max_steps=1000, batch_size=128, memory_capacity=30000,
    #           target_updates=100):
    #     optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
    #     target_net = copy.deepcopy(self.net).to(self.device)
    #     target_net.load_state_dict(self.net.state_dict())
    #     target_net.eval()
    #     self.memory = ReplayMemory(memory_capacity)
    #     episodes = 0
    #     steps = 0
    #     agg_losses = []
    #     episode_rewards = []
    #     while episodes < max_episodes:
    #         self.env.reset()
    #         state, stacked_states = self.stack_states(self.env.render(mode='rgb_array'))
    #         episode_reward = 0
    #
    #         for _ in range(max_steps):
    #             if len(self.memory) == 0:
    #                 print("Filling Samples into Replay Memory...")
    #             action = self.select_action(state, steps)
    #             next_raw_state, reward, done, info = self.env.step(action.item())
    #             if reward < 0:
    #                 reward = -1.
    #                 done = True
    #             else:
    #                 reward = 1.
    #             episode_reward += reward
    #             reward = torch.tensor([reward], device=self.device)
    #             if not done:
    #                 next_state, stacked_states = self.stack_states(next_raw_state, stacked_states)
    #                 # if random.choice([0, 1, 2]) == 0:
    #                 if self.do_explore(steps, 0.0001, 0.05):
    #                     self.memory.push(state, action, next_state, reward)
    #             else:
    #                 next_state = None
    #                 self.memory.push(state, action, next_state, reward)
    #             state = next_state
    #             if len(self.memory) > batch_size:
    #                 if steps == 0:
    #                     print("Start Training...")
    #                 loss = self.optimize_model(optimizer, batch_size, target_net)
    #                 agg_losses.append(loss.item())
    #                 if steps % 100 == 0:
    #                     print("Loss at Step %d: %.4f" % (steps, sum(agg_losses)/len(agg_losses)))
    #                     agg_losses = []
    #                 steps += 1
    #             if done:
    #                 break
    #         episode_rewards.append(episode_reward)
    #
    #         if len(self.memory) >= batch_size:
    #             if episodes % target_updates == 0:
    #                 target_net.load_state_dict(self.net.state_dict())
    #             torch.save(self.net.state_dict(), self.model_path)
    #             if episodes % 100 == 0 or episodes == max_episodes:
    #                 print("[%d]Eposide Rewards: %s" % (episodes, sum(episode_rewards)/len(episode_rewards)))
    #                 episode_rewards = []
    #             episodes += 1
    #     print("Finish Training...")
