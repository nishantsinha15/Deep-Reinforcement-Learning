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
import Util
import pickle


def test(env, len = 10):
    average = 0
    cumulative_score = deque(maxlen=10)
    plot_score = []
    for e in range(len):
        ob = (env.reset())
        total_reward = 0
        start_time = time.time()
        for iter in range(1000000):
            # env.render()

            # Select the action
            action = env.action_space.sample()

            # Take next action and Observe
            ob, reward, done, _ = env.step(action)
            print(_)
            total_reward += reward
            if done:
                print("Test: {}/{}, score: {}, took = {}"
                      .format(e, len, total_reward, time.time() - start_time))
                average += total_reward
                cumulative_score.append(total_reward)
                plot_score.append( sum(cumulative_score)/len(cumulative_score) )
                break
    average /= len
    print("Average test Score = ", average)
    plt.plot(plot_score)
    plt.savefig("Random_pacman.png")
    return average


env = gym.make('MsPacmanDeterministic-v4')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# agent = DeepQAgent(state_size, action_size)
test(env, len = 100)