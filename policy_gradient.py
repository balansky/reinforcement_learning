from policy.pong import PongV0
from policy.cartpole import CartPoleV0
from policy.doom import *
import cv2



def train():
    # net = CartPoleV0(device='cpu')
    net = PongV0("models/pongv0.pt")
    # net.play()
    net.train(10000, 100, resume=True)
    # net = DoomBasic("models/doom.pt", "scenarios/basic.cfg", "scenarios/basic.wad")
    # net.train(10000, 20, 0.002)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__=="__main__":
    train()
