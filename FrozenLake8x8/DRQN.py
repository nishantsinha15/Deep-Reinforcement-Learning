import random
import time
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, Adam
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K, optimizers
import pickle

EPISODE = 100
env = gym.make('FrozenLake8x8-v0')
average = 0
for e in range(EPISODE):
    ob = env.reset()
    start_time = time.time()
    for iter in range(1000000):
        temp = env.render()
        print("Observation = ", ob)
        # Select the action
        # action = agent.act(np.asarray([curr_state]), testing=True)
        action = env.action_space.sample()
        # Take next action and Observe
        ob, reward, done, _ = env.step(action)
        print("Info = ", _)
        if done:
            print("Test: {}/{}, score: {}, took = {}"
                  .format(e, len, reward, time.time() - start_time))
            average += reward
            break
average /= EPISODE
print("Average test Score = ", average)



