import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd,Adam
import matplotlib.pyplot as plt


EPISODES = 1000
file_name = "cartPole_ddqn"
# actions_space = get_actions()


def plot(data):
    x=[]
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    plt.plot(x,y)
    plt.savefig(file_name + '.png')


class State:
    def __init__(self, states):
        self.states = states # 4 sattes
        # self.actions = actions # 3 actions

    def get_input_layer(self):
        ret = []
        for i in range(4):
            ret = ret + list(self.states[i])
            # ret = ret + self.actions[i].tolist()
        ret = np.array(ret)
        ret = np.reshape(ret, [1, 4*4])
        # print(ret.shape)
        # print(ret)
        return ret



class DeepQAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_size = len(action_space)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.min_learning_rate = 0.05
        self.model = self._build_model()
        self.action_space = action_space

    def get_action_from_prediction(self, predict):
        return self.action_space[np.argmax(predict[0])]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))  # changed layer count from 24
        model.add(Dense(80, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))  # changed this from linear
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        act_values = self.model.predict(state.get_input_layer())  # what does this return
        return self.get_action_from_prediction(act_values)

    def replay(self, batch_size, agent2):
        minibatch = random.sample(self.memory, batch_size -1)
        minibatch += [self.memory[-1]]
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_action = -1
                val = -100000000
                temp_val = self.model.predict(next_state.get_input_layer())[0]
                for a in self.action_space:
                    if temp_val[a] > val:
                        val = temp_val[a]
                        best_action = a
                '''check understading '''
                target = (reward + self.gamma * agent2.model.predict(next_state.get_input_layer())[0][best_action])  # Double Q learning
            target_f = self.model.predict(state.get_input_layer())
            target_f[0][action] = target
            '''check understading '''
            self.model.fit(state.get_input_layer(), target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    eVSs = deque(maxlen=1000)
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    print(state_size)
    print(env.action_space)
    action_space = [i for i in range(2)] # hardcoded this
    agent1 = DeepQAgent(state_size*4, action_space)
    agent2 = DeepQAgent(state_size*4, action_space)
    # agent1.load("agent1-ddqn-nishant-v2.h5")
    # agent2.load("agent2-ddqn-nishant-v2.h5")
    batch_size = 32
    c = 0
    recent_average = deque(maxlen=10)
    for e in range(EPISODES):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        total_reward = 0
        prev_state = State([state for i in range(4)])
        curr_state = State([state for i in range(4)])
        my_state = deque(maxlen=4)
    #     my_actions = deque(maxlen=3)
        my_state.append(state)
        flag = True
        for time in range(500):
            # env.render()
            coin_toss = random.random() <= 0.5
            if coin_toss:
                action = agent1.act(curr_state)
            else:
                action = agent2.act(curr_state)

    #         my_actions.append(action)
    #         action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            # if reward != 0: print(reward)
            total_reward += reward
            # next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            my_state.append(state)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, alpha1: {}, alpha2: {}, replay_memory = {},{}"
                      .format(e, EPISODES, total_reward, agent1.epsilon, agent2.learning_rate, agent1.learning_rate, len(agent1.memory), len(agent2.memory)))
                recent_average.append(total_reward)
                av = sum(recent_average) / len(recent_average)
                print( " Recent Average = ", av)
                eVSs.append((e+1,av))
                break

            if len(my_state) == 4:
                curr_state = State(my_state)
                if flag:
                    prev_state = curr_state
                    flag = False
                if coin_toss:
                    agent1.remember(prev_state, action, reward, curr_state, done)
                else:
                    agent2.remember(prev_state, action, reward, curr_state, done)

                prev_state = curr_state

            if coin_toss and len(agent1.memory) > batch_size:
                agent1.replay(batch_size, agent2)
            elif not coin_toss and len(agent2.memory) > batch_size:
                agent2.replay(batch_size, agent1)

        if agent1.epsilon > agent1.epsilon_min:
            agent1.epsilon *= agent1.epsilon_decay
        if agent1.learning_rate > agent1.min_learning_rate:
            agent1.learning_rate *= agent1.epsilon_decay

        if agent2.epsilon > agent2.epsilon_min:
            agent2.epsilon *= agent2.epsilon_decay
        if agent2.learning_rate > agent2.min_learning_rate:
            agent2.learning_rate *= agent2.epsilon_decay
        #
        # if e % 100 == 0:
        #     agent1.save(file_name + "agent1.h5")
        #     agent2.save(file_name + "agent2.h5")

        if e % 10 == 0:
            plot(eVSs)
