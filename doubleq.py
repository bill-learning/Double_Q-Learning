#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:45:55 2017

@author: bill
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
import numpy as np
import matplotlib.pyplot as plt

def build_model(n_inputs,n_outputs):
    """ Builds a keras neural network """
    Q = Sequential()
    Q.add(Dense(16, activation='relu', kernel_initializer='lecun_uniform',input_dim=n_inputs))
    Q.add(Dense(16, activation='relu', kernel_initializer='lecun_uniform'))
    Q.add(Dense(n_outputs,activation='linear',kernel_initializer='lecun_uniform'))
    Q.compile(Adam(),'mse')
    return Q

def main():
    # environment info
    env = gym.make('CartPole-v1')
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    n_episodes = 501
    max_steps = 1000
    env._max_episode_steps = max_steps

    # Double Q Networks
    A = build_model(n_inputs,n_outputs)
    B = build_model(n_inputs,n_outputs)
    Q = [A,B]
    
    # memory for experience replay
    memory = []
    
    # hyperparams
    gamma = .99
    epsilon = 1
    decay = .99
    n_replay = 1000
    
    # Learning
    scores = [] # scores
    mean = [] # rolling mean (100 epsiode)
    std = [] # rolling std
    for episode in xrange(n_episodes):
        done = False
        total = 0
        state = env.reset().reshape(1,-1)
        
        # Perform an episode
        steps = 0
        while not done:
            steps += 1
            # Epsilon-Greedy action selection (Network consensus)
            action = np.argmax((Q[0].predict(state)+Q[1].predict(state))[0])
            if np.random.rand() < epsilon:
                action = np.random.randint(0,n_outputs)
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1,-1)
            total += reward
            # Store experience
            if steps > max_steps:
                memory.append((state,action,next_state,reward,False))
            else:
                memory.append((state,action,next_state,reward,done))
            state = next_state
        
        # Experience Replay
        indicies = np.random.choice(len(memory),size=n_replay)
        # iterate over random memories
        for mem in xrange(n_replay):
            m_state, m_action, m_next_state,m_reward,m_done = memory[indicies[mem]]
            # randomly choose the training/testing network
            train = np.random.randint(0,2)
            test = 1-train
            
            # Get training net's Q-Values for state
            prediction = Q[train].predict(m_state)[0]
            
            # Update the value for the action chosen 
            if m_done:    # terminal states don't have future value
                prediction[m_action] = m_reward
            else:       # testing network gets next state values
                value = Q[test].predict(m_next_state)[0]
                prediction[m_action] = m_reward + gamma*np.max(value)
            # Training network learns on memory
            Q[train].train_on_batch(m_state.reshape(1,-1),prediction.reshape(1,-1))
         
        # Decay epsilon->0
        epsilon*=decay
        
        # Gather data
        scores.append(total)
        mean.append(np.mean(scores[-100:]))
        std.append(np.std(scores[-100:]))
        if episode%50 == 0:
            print "Episode:",episode,"Reward:",total
        if mean[-1] > 500: # stop when solved
            break
    env.close()

    # Plot results
    scores = np.array(scores)   
    mean = np.array(mean)
    std = np.array(std)   
    plt.plot(range(mean.shape[0]),mean)
    plt.fill_between(range(mean.shape[0]),mean-std,mean+std,alpha=.5)
    plt.grid()
    plt.title("100-Episode Rolling Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.show()
    
if __name__ == '__main__':
    main()    