import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, Adam
import matplotlib.pyplot as plt

from keras import backend as K, optimizers

EPISODES = 1000


def plot(data):
    x = []
    y = []
    for i, j in data:
        x.append(i)
        y.append(j)
    plt.plot(x, y)
    plt.savefig('cartpole_240.png')


class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, agent2):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(agent2.model.predict(next_state)[0]))
            target_f = self.model.predict(state)  # What does this return? Ans type = [[0.08708638 0.4333976 ]]
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    eVSs = deque(maxlen=1000)
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DeepQAgent(state_size, action_size)
    agent2 = DeepQAgent(state_size, action_size)
    c = 0
    # agent.load("cartpole-dqn.h5")
    done = False
    batch_size = 32
    recent_average = deque(maxlen=10)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            c += 1
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                recent_average.append(total_reward)
                av = sum(recent_average) / len(recent_average)
                print(" Recent Average = ", av)
                eVSs.append((e + 1, av))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, agent2)

            if c > 500:
                c = 0
                agent2.model.set_weights(agent.model.get_weights())
                print("Updated the target model")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 10 == 0:
            plot(eVSs)
