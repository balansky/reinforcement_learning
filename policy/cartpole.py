from . import *
from utils import policy_rewards


class Net(torch.nn.Module):

    def __init__(self, state_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 10)

        self.fc2 = torch.nn.Linear(10, 2)
        # self.head = nn.Linear(448, 2)
        self.head = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x.view(x.size(0), -1))


class CartPoleV0(object):

    def __init__(self, device=None):
        super(CartPoleV0, self).__init__()
        env = gym.make("CartPole-v0")
        self._env = env.unwrapped
        self._action_size = self._env.action_space.n
        self._state_size = 4
        self._device =torch.device(device) if device \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = Net(self._state_size).to(self._device)

    def run_episode(self, gamma):
        done = False
        state = torch.tensor(self._env.reset(), device=self._device, dtype=torch.float32)
        episode_logits, episode_actions, episode_rewards = [], [], []
        while not done:
            # env.render()
            logit_output = self._net(state.unsqueeze(0))[0]
            action_distribution = F.softmax(logit_output).detach().to("cpu").numpy()
            action = np.random.choice(range(action_distribution.shape[0]),
                                      p=action_distribution)
            new_state, reward, done, info = self._env.step(action)
            new_state = torch.tensor(new_state, device=self._device, dtype=torch.float32)
            state = new_state
            episode_logits.append(logit_output)
            episode_rewards.append(reward)
            episode_actions.append(action)
        episode_reward_sum = np.sum(episode_rewards)
        discounted_episode_reward = policy_rewards.discount_episode_rewards(gamma, episode_rewards)
        discounted_episode_reward = policy_rewards.normailize_rewards(discounted_episode_reward)
        return episode_actions, episode_logits, discounted_episode_reward, episode_reward_sum

    def train(self, max_eposides, lr=0.01, gamma=0.95):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for i in range(max_eposides):
            optimizer.zero_grad()
            episode_actions, episode_logits, discounted_episode_rewards, episode_rewards_sum = self.run_episode(gamma)
            episode_logits = torch.stack(episode_logits).to(self._device)
            episode_actions = torch.tensor(episode_actions, device=self._device, dtype=torch.long)
            discounted_episode_rewards = torch.tensor(discounted_episode_rewards, device=self._device,
                                                      dtype=torch.float32)
            neg_log_prob = criterion(episode_logits, episode_actions)
            loss = (neg_log_prob * discounted_episode_rewards).mean()
            loss.backward()
            optimizer.step()
            print("==========================================")
            print("Episode: ", i + 1)
            print("Reward: ", episode_rewards_sum)
            print("Loss: {}".format(loss.detach().to('cpu')))
        print("Training Done ! ")

