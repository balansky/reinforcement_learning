from . import *
from vizdoom import *
from collections import deque
import os
from utils import policy_rewards
from utils import doom
import cv2

class Net(torch.nn.Module):

    def __init__(self, action_size, stack_size):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(stack_size, 32, 8, 4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 4, 2)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.fc1 = torch.nn.Linear(128*3*3, 512)
        self.head = torch.nn.Linear(512, action_size)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.fc1(x.view(-1, 128*3*3)))
        x = self.head(x)
        return x


class DoomBasic(object):

    def __init__(self, model_path, doom_config_path, doom_scenario_path, device=None):
        self._model_path = model_path
        self._env = doom.create_environment(doom_config_path, doom_scenario_path)
        self._state_size = 84
        self._stack_size = 4
        self._device =torch.device(device) if device \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._action_size = self._env.get_available_buttons_size()
        self._net = Net(self._action_size, self._stack_size).to(self._device)
        self.possible_actions = np.identity(3, dtype=int).tolist()

    def __del__(self):
        self._env.close()


    def preprocess_frame(self, frame):
        frame = frame[30:-10, 30:-30]
        frame = frame / 255.
        frame = cv2.resize(frame, (self._state_size, self._state_size))
        return frame


    def stacked_state(self, frame, frame_stack=None):
        frame = self.preprocess_frame(frame)
        if frame_stack:
            frame_stack.append(frame)
        else:
            frame_stack = deque([np.zeros((self._state_size, self._state_size), dtype=np.int) for _ in range(self._stack_size)], maxlen=self._stack_size)
            for _ in range(self._stack_size):
                frame_stack.append(frame)
        state = np.stack(frame_stack, axis=0)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        return state, frame_stack

    def run_eposide(self, gamma):
        self._env.new_episode()
        state, stacked_frames = self.stacked_state(self._env.get_state().screen_buffer)
        eposide_rewards = []
        eposide_actions = []
        eposide_logits = []
        done = False
        while not done:
            logits = self._net(state)
            action_probability_distribution = F.softmax(logits).cpu().detach().numpy()
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())
            reward = self._env.make_action(self.possible_actions[action])
            done = self._env.is_episode_finished()
            if done:
                next_frame = np.zeros((self._state_size, self._state_size), dtype=np.int)
            else:
                next_frame = self._env.get_state().screen_buffer

            state, stacked_frames = self.stacked_state(next_frame, stacked_frames)
            eposide_logits.append(logits.squeeze(0))
            eposide_actions.append(action)
            eposide_rewards.append(reward)
        discount_eposide_rewards = policy_rewards.discount_episode_rewards(gamma, eposide_rewards)
        return eposide_logits, eposide_actions, discount_eposide_rewards, np.sum(eposide_rewards)


    def train(self, num_batchs, batch_eposides, lr=1e-4, gamma=0.95, resume=False):
        if resume and os.path.exists(self._model_path):
            self._net.load_state_dict(torch.load(self._model_path))
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
            discount_rewards = policy_rewards.normailize_rewards(discount_rewards)
            batch_actions = torch.tensor(actions, dtype=torch.long).to(self._device)
            batch_logits = torch.stack(logits)
            batch_discount_rewards = torch.tensor(discount_rewards, dtype=torch.float32).to(self._device)
            batch_reward = np.sum(rewards)
            loss = criterion(batch_logits, batch_actions)
            loss = (loss * batch_discount_rewards).mean()
            loss.backward()
            # for group in optimizer.param_groups:
            #     for p in group['params']:
            #         p.grad = -1 * p.grad
            # for param in self._net.parameters():
            #     param.grad.data.clamp_(-1, 1)

            optimizer.step()
            print("==========================================")
            print("Batch: ", i+1, "/", num_batchs)
            print("-----------")
            print("Number of training episodes: {}".format((i+1)*batch_eposides))
            print("Batch reward: {}".format(batch_reward))
            print("Mean Reward of that batch {}".format(batch_reward/batch_eposides))
            print("Batch Loss: {}".format(loss))
            torch.save(self._net.state_dict(), self._model_path)
        print("Training Done ! ")


class DoomHealth(DoomBasic):

    def preprocess_frame(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[80:, :]
        frame = frame / 255.
        frame = cv2.resize(frame, (self._state_size, self._state_size))
        return frame
