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


class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = MyQueue(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = -(9 / 10000000)
        self.learning_rate = 0.00025
        input_shape = (4, 110, 84)
        self.model = self._build_model(input_shape)
        self.target_model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4,4), padding='valid', activation='relu', input_shape=input_shape, data_format='channels_first'))
        model.add(Conv2D(64, 4, strides=(2,2), padding='valid', activation='relu', data_format='channels_first'))
        model.add(Conv2D(64, 3, strides=(1,1), padding='valid', activation='relu', data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        rmsprop = optimizers.RMSprop(lr=self.learning_rate)

        model.compile(loss='mean_squared_error', optimizer = rmsprop, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state, testing = False):
        if not testing:
            epsilon = self.epsilon
        else:
            epsilon = self.epsilon_min
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        if random.random() < 0.5:
            act_values = self.model.predict(state)
        else:
            act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        coin_toss = random.random() < 0.5
        if coin_toss:
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    best_action = 0
                    val = -100000000
                    temp_val = self.model.predict(next_state)[0]
                    for a in range(self.action_size):
                        if temp_val[a] > val:
                            val = temp_val[a]
                            best_action = a
                    target = (reward + self.gamma * self.target_model.predict(next_state)[0][
                        best_action])  # Double Q learning
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        else:
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    best_action = 0
                    val = -100000000
                    temp_val = self.target_model.predict(next_state)[0]
                    for a in range(self.action_size):
                        if temp_val[a] > val:
                            val = temp_val[a]
                            best_action = a
                    target = (reward + self.gamma * self.model.predict(next_state)[0][
                        best_action])  # Double Q learning
                target_f = self.target_model.predict(state)
                target_f[0][action] = target
                self.target_model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name + "behaviour")
        self.target_model.load_weights(name + "target")

    def save(self, name):
        self.model.save_weights(name + "behaviour")
        self.target_model.save_weights(name + "target")


class MyQueue:
    def __init__(self, maxlen):
        self.q = []
        self.max_len = maxlen

    def add(self, element):
        if len(self.q) == 200000:
            self.q = self.q[1:] + [element]
        else:
            self.q.append(element)

    def sample(self, size):
        return random.sample(self.q, size)

    def __len__(self):
        return len(self.q)


def test(env, agent, len = 10):
    average = 0
    for e in range(len):
        ob = Util.preprocess(env.reset())
        curr_state = np.array([ob, ob, ob, ob])
        total_reward = 0
        start_time = time.time()
        for iter in range(1000000):
            env.render()

            # Select the action
            action = agent.act(np.asarray([curr_state]), testing=True)

            # Take next action and Observe
            ob, reward, done, _ = env.step(action)
            ob = Util.preprocess(ob)
            next_state = Util.get_next_state(curr_state, ob)
            total_reward += reward
            curr_state = next_state
            if done:
                print("Test: {}/{}, score: {}, took = {}"
                      .format(e, len, total_reward, time.time() - start_time))
                average += total_reward
                break
    average /= len
    print("Average test Score = ", average)
    return average

file_name = 'pacman'

env = gym.make('MsPacmanDeterministic-v4')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DeepQAgent(state_size, action_size)
agent.load(file_name)
test(env, agent, len = 100)
