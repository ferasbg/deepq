from 2048.agent import target_model
import copy
from util import *
import os

import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

agentMoves = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}

