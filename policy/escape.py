import torch
import torchvision
import gym
from collections import deque
import numpy as np
from utils import policy_rewards


class Network(torch.nn.Module):

    def __init__(self, action_size):
        super(Network, self).__init__()
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
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1152, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, action_size)

        )

    def forward(self, input):
        h = self.block(input)
        h = h.view(-1, 1152)
        h = self.fc(h)
        return h


class JourneyEscape(object):

    def __init__(self, model_path='models/policy_escape.pt', device=torch.device("cpu")):
        self.model_path = model_path
        self.device = device
        self.env = gym.make("JourneyEscape-v0")
        self.action_space = self.env.action_space.n
        self.stack_size = 4
        self.resize_fn = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop((160, 160)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((84, 84)),
                torchvision.transforms.ToTensor()
            ])
        self.net = Network(self.action_space).to(self.device)
        self.net.apply(self.init_weights)

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
            stacked_states = deque([state for _ in range(self.stack_size)], maxlen=self.stack_size)
        torch_state = torch.cat([stacked_states[s] for s in range(self.stack_size)], 0).unsqueeze(0).to(self.device)
        return torch_state, stacked_states

    def run_episode(self, gamma, max_steps):
        done = False
        self.env.reset()
        state, stacked_states = self.stack_states(self.env.render(mode="rgb_array"))
        episode_logits, episode_actions, episode_rewards = [], [], []
        steps = 0
        while not done:
            steps += 1
            logit_output = self.net(state)
            action_distribution = torch.nn.functional.softmax(logit_output).detach().to("cpu").numpy()[0]
            action = np.random.choice(range(action_distribution.shape[0]),
                                      p=action_distribution)
            next_raw_state, reward, done, info = self.env.step(action)
            if steps >= max_steps:
                done = True
            # if steps >= max_steps and reward == 0:
            #     reward = 1.
            #     done = True
            elif reward == 0:
                reward = 1.
            # elif reward < 0:
            #     reward = 0.
            #     done = True
            state, stacked_states = self.stack_states(next_raw_state, stacked_states)
            episode_logits.append(logit_output)
            episode_rewards.append(reward)
            episode_actions.append(action)
        episode_reward_sum = np.sum(episode_rewards)
        discounted_episode_reward = policy_rewards.discount_episode_rewards(gamma, episode_rewards)
        # discounted_episode_reward = discounted_episode_reward / np.mean(discounted_episode_reward)
        discounted_episode_reward = policy_rewards.normailize_rewards(discounted_episode_reward)
        return episode_actions, episode_logits, discounted_episode_reward, episode_reward_sum

    def make_batch(self, batch_size, gamma, max_steps):
        batch_rewards = []
        batch_discounted_rewards = []
        batch_logits = []
        batch_actions = []
        while True:
            episode_actions, episode_logits, discounted_episode_rewards, episode_rewards_sum = self.run_episode(gamma, max_steps)
            if len(episode_actions) < 2:
                continue
            batch_rewards.append(episode_rewards_sum)
            batch_discounted_rewards.extend(discounted_episode_rewards)
            batch_logits.extend(episode_logits)
            batch_actions.extend(episode_actions)
            if len(batch_logits) >= batch_size:
                break
        batch_logits = torch.cat(batch_logits, 0).to(self.device)
        batch_actions = torch.tensor(batch_actions, device=self.device, dtype=torch.long)
        batch_discounted_rewards = torch.tensor(batch_discounted_rewards, device=self.device,
                                                dtype=torch.float32)
        batch_rewards_mean = np.mean(batch_rewards)
        return batch_actions, batch_logits, batch_discounted_rewards, batch_rewards_mean


    def train(self, batch_size=128, learning_rate=0.0002, gamma=0.95, max_iterations=500000, max_steps=1000):
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        total_rewards = []
        total_losses = []
        for iter in range(max_iterations):
            batch_actions, batch_logits, batch_discounted_rewards, batch_rewards_mean = self.make_batch(batch_size, gamma, max_steps)
            neg_log_prob = criterion(batch_logits, batch_actions)
            loss = (neg_log_prob * batch_discounted_rewards).mean()
            total_rewards.append(batch_rewards_mean)
            total_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % 100 == 0:
                print("==========================================")
                print("Iteration: ", iter + 1)
                print("Loss: {}".format(np.mean(total_losses)))
                print("Reward Mean: ", np.mean(total_rewards))
                torch.save(self.net.state_dict(), self.model_path)
                total_rewards = []
                total_losses = []
        print("Training Done ! ")
