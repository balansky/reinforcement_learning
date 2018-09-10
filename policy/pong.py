from . import *
import gym
import cv2
from collections import deque

class Net(torch.nn.Module):

    def __init__(self, channel):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(channel, 32, 8, 4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 4, 2)

        self.fc1 = torch.nn.Linear(128*3*3, 512)
        self.head = torch.nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 128*3*3)))
        x = self.head(x)
        return x

    # def __init__(self, channel):
    #     super(Net, self).__init__()
    #     self.fc = torch.nn.Linear(80*80*channel, 200)
    #     self.head = torch.nn.Linear(200, 2)
    #     self._channel = channel
    #
    # def forward(self, x):
    #     x = x.view(-1, 80*80*self._channel)
    #     x = F.relu(self.fc(x))
    #     x = self.head(x)
    #     return x



class PongV0(object):

    def __init__(self, stack_size=4, device=None):
        super(PongV0, self).__init__()
        self._env = gym.make("Pong-v0")
        self._action_size = self._env.action_space.n
        self._stack_size = stack_size
        self._state_size = (80, 80)
        self._device =torch.device(device) if device \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = Net(self._stack_size).to(self._device)

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[35:195]
        frame = cv2.resize(frame, self._state_size)
        frame = frame / 255.
        return frame

    def stacked_state(self, frame, frame_stack=None):
        frame = self.preprocess_frame(frame)
        if frame_stack:
            frame_stack.append(frame)
        else:
            frame_stack = deque([frame for _ in range(self._stack_size)], maxlen=self._stack_size)
        state = np.stack(frame_stack, axis=0)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        return state, frame_stack


    def run_eposide(self, gamma):
        state, stack = self.stacked_state(self._env.reset())
        eposide_rewards = []
        eposide_actions = []
        eposide_logits = []
        done = False
        while not done:
            logits = self._net(state)
            action_probability_distribution = F.softmax(logits).cpu().detach().numpy()
            action = 0 if np.random.uniform() < action_probability_distribution.ravel()[0] else 1
            # action = np.random.choice(range(action_probability_distribution.shape[1]),
            #                           p=action_probability_distribution.ravel())
            next_frame, reward, done, info = self._env.step(action + 2)
            state, stack = self.stacked_state(next_frame, stack)
            eposide_logits.append(logits.squeeze(0))
            eposide_actions.append(action)
            eposide_rewards.append(reward)
            if reward == -1 or reward == 1:
                done = True
        discount_eposide_rewards = discount_and_normalize_rewards(gamma, eposide_rewards)
        return eposide_logits, eposide_actions, discount_eposide_rewards, np.sum(eposide_rewards)

    def train(self, num_batchs, batch_eposides, lr=0.0001, gamma=0.95):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.RMSprop(self._net.parameters(), lr)
        for i in range(num_batchs):
            logits = []
            actions = []
            discount_rewards = []
            rewards = []
            optimizer.zero_grad()
            for _ in range(batch_eposides):
                eposide_logits, eposide_actions, discount_eposide_rewards, eposide_reward = self.run_eposide(gamma)
                logits.extend(eposide_logits)
                actions.extend(eposide_actions)
                discount_rewards.extend(discount_eposide_rewards)
                rewards.append(eposide_reward)
            batch_actions = torch.tensor(actions, dtype=torch.long).to(self._device)
            batch_logits = torch.stack(logits)
            batch_discount_rewards = torch.tensor(discount_rewards, dtype=torch.float32).to(self._device)
            batch_reward = np.sum(rewards)
            loss = criterion(batch_logits, batch_actions)
            loss = torch.mean(loss * batch_discount_rewards)
            loss.backward()
            # for group in optimizer.param_groups:
            #     for p in group['params']:
            #         p.grad = -1 * p.grad
            for param in self._net.parameters():
                param.grad.data.clamp_(-1, 1)

            optimizer.step()
            print("==========================================")
            print("Batch: ", i+1, "/", num_batchs)
            print("-----------")
            print("Number of training episodes: {}".format((i+1)*batch_eposides))
            print("Batch reward: {}".format(batch_reward))
            print("Mean Reward of that batch {}".format(batch_reward/batch_eposides))
            print("Batch Loss: {}".format(loss))
        print("Training Done ! ")
