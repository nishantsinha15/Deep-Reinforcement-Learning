import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import sgd, Adam
import matplotlib.pyplot as plt

from keras import backend as K, optimizers

EPISODES = 1000
time_lstm = 2


def plot(data):
    x = []
    y = []
    for i, j in data:
        x.append(i)
        y.append(j)
    plt.plot(x, y)
    plt.savefig('cartpole_240.png')

class MyQueue:
    def __init__(self, maxlen):
        self.q = []
        self.max_len = maxlen

    def add(self, element):
        if len(self.q) == 1000:
            self.q = self.q[1:] + [element]
        else:
            self.q.append(element)

    def sample(self, size):
        return random.sample(self.q, size)

    def __len__(self):
        return len(self.q)



class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = MyQueue(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    # sample, time step, features
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(24, input_shape=(time_lstm, self.state_size)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, episode_frame):
        self.memory.add(episode_frame)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, agent2):
        sample_episodes = np.random.randint(0, high=len(self.memory), size = batch_size)
        minibatch = []
        for i in sample_episodes:
            start = np.random.randint(0, high=len(self.memory.q[i])-2*time_lstm) # might be buggy
            state_a = []
            # print("Episode selected = ", i)
            # print("Frame selected = ", start, " / ", len(self.memory.q[i]))
            for j in range(start, start+time_lstm):
                state_a.append(self.memory.q[i][j][0])
            act_taken = self.memory.q[i][start + time_lstm - 1 ][1]
            reward_got = self.memory.q[i][start + time_lstm - 1][2]
            next_state_reached = []
            for j in range(start + time_lstm, start + time_lstm + time_lstm):
                next_state_reached.append(self.memory.q[i][j][0])
            done = self.memory.q[i][start + 2*time_lstm - 1][4]
            minibatch.append((reshape_frames(state_a), act_taken, reward_got, reshape_frames(next_state_reached), done))

        # minibatch = random.sample(self.memory, batch_size)
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


'''
remember[ ]
is a list of episodes:

remember[i] 
is a list of frames of i'th episode

remember[i][j]
is A 4 tuple of previous four screens


a = [[0 for i in range(ob_space)] for j in range(time_sequence) ]
for i_ob in the time_sequence:
    for j in ob_space of i_ob:
        a[i_ob_index][j].append(i_ob[j])

a = a.reshape(1, 10, 4)


'''

def reshape_frames(l):
    state_size = 4
    l = list(l)
    return np.array(l).reshape(1,time_lstm, state_size)


# todo what's all that about resetting the weights of the lstm?

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
    act_frame = deque(maxlen=time_lstm)
    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        episode_frames = []
        for time in range(500):
            act_frame.append(state)
            c += 1

            # env.render()
            if len(act_frame) == time_lstm:
                action = agent.act(reshape_frames(act_frame))
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward = reward if not done else -10
            episode_frames.append((state, action, reward, next_state, done))
            # agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                recent_average.append(total_reward)
                av = sum(recent_average) / len(recent_average)
                print(" Recent Average = ", av)
                eVSs.append((e + 1, av))

                # remember the episode
                agent.remember(episode_frames)

                break
            if len(agent.memory) > 5:
                agent.replay(batch_size, agent2)

            if c > 500:
                c = 0
                agent2.model.set_weights(agent.model.get_weights())
                print("Updated the target model")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 10 == 0:
            plot(eVSs)
