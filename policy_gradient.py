from policy.pong import PongV0
import cv2



def train():
    net = PongV0()
    net.train(10000, 100)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__=="__main__":
    train()
