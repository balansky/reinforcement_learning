from policy.pong import PongV0
from policy.cartpole import CartPoleV0
import cv2



def train():
    # net = CartPoleV0(device='cpu')
    net = PongV0(stack_size=3, device='cpu')
    net.train(10000, 20)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__=="__main__":
    train()
