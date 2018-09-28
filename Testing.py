import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]


def pre_process(obs):
    # img = obs[1:176:2, ::2]  # crop and downsize
    img = obs[::2, ::2]
    img = crop_center(img, 80, 80)
    img = img.sum(axis=2)  # to greyscale
    # img[img == 210 + 164 + 74] = 0  # Improve contrast
    # img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    img =  img.reshape(80, 80, 1)
    plt.imshow(img.reshape(80, 80), interpolation="nearest", cmap="gray")
    plt.show()


def create():
    env = gym.make("MsPacman-v0")
    obs = env.reset()
    print(obs.shape)
    print(env.action_space)
    # plt.imshow(obs)
    # plt.show()
    # img1 = Image.fromarray(obs, 'RGB')
    # img1.save('Init.png')
    pre_process(obs)

create()