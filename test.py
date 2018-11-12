# import torch
#
#
# c = torch.nn.CrossEntropyLoss(reduction='none')
# a = torch.randn(1, 3, dtype=torch.float)
# w = torch.randn(3, 2, dtype=torch.float, requires_grad=True)
#
# y_ = torch.mm(a, w)
# y = torch.empty(1, dtype=torch.long)
#
# y[0 ] = 1
# # y[1 ] = 0
#
# softmax = torch.nn.Softmax(dim=1)
#
# s = softmax(y_)
# loss = c(y_, y)
#
# loss_sum = torch.sum(loss)
#
# loss_sum.backward()
#
# grad = w.grad
#
# print("input: {}".format(a))
# print("weight: {}".format(w))
# print("target: {}".format(y))
# print("output: {}".format(y_))
# print("softmax output: {}".format(s))
# print("loss: {}".format(loss))
# print("loss sum: {}".format(loss_sum))
# print("w grads: {}".format(grad))
# print(loss.grads)
# from vizdoom import *
import random
import time
import cv2
# from qlearning import frozenlake, escape
from policy import escape
import gym
import torch
import matplotlib.pyplot as plt


def gym_run():
    env = gym.make("JourneyEscape-v0")
    action_space = env.action_space.n
    actions = [i for i in range(action_space)]
    print(action_space)
    env.reset()
    for i in range(1000):
        action = random.choice(actions)
        state, reward, done, info = env.step(action)
        if i % 3 == 0 or i == 0:
            env.render()
        print(reward)
        time.sleep(0.2)
        if done:
            env.reset()
            print("reset")

# gym_run()
device = torch.device("cuda:0")
je = escape.JourneyEscape(device=device)
je.train()
# je.play()
# mm = je.env.reset()
# ss, stacked_ss = je.stack_states(mm)
# mn, _, _, _ = je.env.step(1)
# ll, stacked_ss = je.stack_states(mn, stacked_ss)
# ll = torch.unsqueeze(ll, 0)
#
# je.net(ll)
#
# plt.imshow(mm)
# plt.show()
# for _ in range(3):
#     env.step(1)
#     env.render()
#     time.sleep(1)
# for _ in range(10):
#     env.step(4)
#     env.render()
#     time.sleep(1)
print('ff')



# fl = frozenlake.FrozenLake()
# fl.train()
# fl.play()


# game = DoomGame()
# game.load_config("scenarios/basic.cfg")
# game.set_doom_scenario_path("scenarios/basic.wad")
# game.init()
#
# shoot = [0, 0, 1]
# left = [1, 0, 0]
# right = [0, 1, 0]
# actions = [shoot, left, right]
#
# episodes = 10
# for i in range(episodes):
#     game.new_episode()
#     while not game.is_episode_finished():
#         state = game.get_state()
#         img = state.screen_buffer
#         img = img[30:-10, 30:-30]
#         img = img / 255.
#         img = cv2.resize(img, (84, 84))
#         misc = state.game_variables
#         reward = game.make_action(random.choice(actions))
#         print("\treward:", reward)
#         cv2.imshow('process', img)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
#         time.sleep(0.02)
#     print ("Result:", game.get_total_reward())
#     time.sleep(2)

