import random
import time

import gym
import numpy as np
from collections import deque

from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, Adam
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K

file_name = 'prioritized_pacman'


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess_old(img):
    return to_grayscale(downsample(img))


def preprocess(obs):
    image = Image.fromarray(obs, 'RGB').convert('L').resize((84, 110))
    # Convert image to array and return it
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])


def plot(data):
    x = []
    y = []
    for i, j in data:
        x.append(i)
        y.append(j)
    plt.plot(x, y)
    plt.savefig(file_name)


def get_next_state(curr_state, ob):
    return np.append(curr_state[1:], [ob], axis = 0)