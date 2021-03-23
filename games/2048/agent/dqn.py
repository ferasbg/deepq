import numpy as np
import gym
import pygame
from pygame.locals import *
import random

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.regularizers import Regularizer, l2

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.metrics import MeanSquaredError

# define M number of episodes
EPISODES = 999
BATCH_SIZE = 5

class Agent():
    def __init__(self, state_size, action_size, optimizer):
      # agent
      self.state_size = state_size
      self.action_size = action_size
      self.dqn_memory = deque(maxlen=20000)
      self.reward = 50
      self.time_step = 5
      self.gamma = 0.96
      self.optimizer = optimizer
      # epsilon decay
      self.epsilon_decay = 0.98
      self.learning_rate = 0.001
      self.target_model = self.target_model()
      self.update_target_model()
      self.optimizer = optimizer


    def target_model(self):
      agent = Sequential()
      agent.add(Input(shape=(4, 4, 1), name='board'))
      agent.add(Conv2D(64, (2, 2), activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='conv1'))
      agent.add(Conv2D(64, (2, 2), activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='conv2'))
      agent.add(Flatten())
      agent.add(Dense(256, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='1'))
      agent.add(Dense(128, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='2'))
      agent.add(Dense(64, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='3'))
      agent.add(Dense(1, activation='linear', name='output'))
      agent.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[MeanSquaredError()])
      agent.save('agent.h5', overwrite=True)
      return agent

  # learning rate
  # huber loss, MSE
  # terminated (check for termination state)
  # successor_state
  # def experience replay
  # def huber_loss
	# def load_weights
	# def save_weights
	# def action(self, state)
	# def dqn_memory
  # def expectimax(node, depth)
  # def bellman_curve(self, state_size, action_size)
  # fc
  # gradient descent
  # markov chain
  # def action_space

game = GameOf2048()

# initialize variables
# score, reward, state, action
