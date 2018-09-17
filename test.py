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
from vizdoom import *
import random
import time
import cv2

game = DoomGame()
game.load_config("scenarios/basic.cfg")
game.set_doom_scenario_path("scenarios/basic.wad")
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        img = img[30:-10, 30:-30]
        img = img / 255.
        img = cv2.resize(img, (84, 84))
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        print("\treward:", reward)
        cv2.imshow('process', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        time.sleep(0.02)
    print ("Result:", game.get_total_reward())
    time.sleep(2)