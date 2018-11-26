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
from keras import backend as K

# todo create a deque type class

class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = -(9 / 10000000)
        self.learning_rate = 0.00025
        self.model = self._build_model()
        input_shape = (4, 110, 84)
        self.target_model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4,4), padding='valid', activation='relu', input_shape=input_shape, data_format='channels_first'))
        model.add(Conv2D(64, 4, strides=(2,2), padding='valid', activation='relu', data_format='channels_first'))
        model.add(Conv2D(64, 3, strides=(1,1), padding='valid', activation='relu', data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mean_squared_error', optimizer='rmsprop', lr=self.learning_rate, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if random.random() < 0.5:
            act_values = self.model.predict(state)
        else:
            act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
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