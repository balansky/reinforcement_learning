import gym
import torchvision
import torch
from collections import deque, namedtuple
import copy

import numpy as np
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQNetwork(torch.nn.Module):

    def __init__(self):
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
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1152, 512),
            torch.nn.ELU()
        )
        self.fc2 = torch.nn.Linear(512, 5)

    def forward(self, input):
        h = self.block(input)
        h = h.view(-1, 1152)
        h = self.fc1(h)
        h = self.fc2(h)
        return h


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
        self.net = DQNetwork().to(device)
        self.memory = None
        self.actions = [0, 1, 2, 3, 4]

    def __del__(self):
        self.env.close()

    def stack_states(self, raw_state, stacked_states=None):
        state = self.resize_fn(raw_state)
        if stacked_states:
            stacked_states.append(state)
        else:
            stacked_states = deque([state for _ in range(self.stack_size)], maxlen=self.stack_size)
        torch_state = torch.cat([s for s in stacked_states], 0).unsqueeze(0)
        return torch_state, stacked_states

    def select_action(self, state, step_done):
        exp_exp_tradeoff = np.random.rand()
        explore_probability = 0.01 + (1.0 - 0.01) * np.exp(-0.0001 * step_done)
        if (explore_probability > exp_exp_tradeoff):
            action = torch.tensor([random.choice(self.actions)], device=self.device, dtype=torch.long)
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
        loss = torch.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def train(self, learning_rate=0.0002, max_episodes=500, max_steps=100, batch_size=64, memory_capacity=10000,
              target_updates=10):
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        target_net = copy.deepcopy(self.net).to(self.device)
        target_net.load_state_dict(self.net.state_dict())
        target_net.eval()
        self.memory = ReplayMemory(memory_capacity)
        episodes = 0
        steps = 0
        while episodes < max_episodes:
            self.env.reset()
            state, stacked_states = self.stack_states(self.env.render(mode='rgb_array'))
            for _ in range(max_steps):

                action = self.select_action(state, steps)
                next_raw_state, reward, done, info = self.env.step(action.item())
                if not done:
                    next_state, stacked_states = self.stack_states(next_raw_state, stacked_states)
                    self.memory.push(state, action, next_state, reward)
                else:
                    next_state = None
                    self.memory.push(state, action, next_state, reward)
                state = next_state
                if len(self.memory) > batch_size:
                    self.optimize_model(optimizer, batch_size, target_net)
                    steps += 1

            if len(self.memory) >= batch_size:
                if episodes % target_updates == 0:
                    target_net.load_state_dict(self.net.state_dict())
                episodes += 1
