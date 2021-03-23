from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import numpy as np
import matplotlib as plt

# deque = define array size
from collections import deque
import time

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE= = 1000
MODEL_NAME = "256x2"
DISCOUNT = 0.99
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5



class DQNAgent():
    '''
    description: General Deep DQN Network.
    
    '''

    def __init__(self):
        # main model  # gets train every step
        self.model = self.create_model()
        # target model  # call .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        # need to copy weights to target weights for c set of iterations

        # adjust weights with respect to each sample in the batch_size
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # display log metrics
        self.tensorboard = ModifiedTensorboard(Log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # counter to update target network with prediction network weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        # pass through dense layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # add transitions to update replay memory
        self.replay_memory.append(transition)

    # call prediction network for q-values given current observation space (environment state)
    def get_q_values(self, state):
        return self.model.predict(np.array(state),reshape(-1, *state.shape)/255.0)[0]

    def train(self, state, terminal_state, step):
        # check for saved samples to meet requirements to train
        if len(self.replay_memory() < MIN_REPLAY_MEMORY_SIZE):
            return

    # separate states and matching q-values from minibatch of memory (successor_state, q-values, and future Q-Values)

    # get set of random samples from replay memory table
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

    # get current states from minibatch, then query prediction network for q-Values
    current_states = np.array([transition[0] for transition in minibatch]/255)
    currentqValues = self.model_predict(current_states)

    # get future states from minibatch, query prediction network, unless u need to make model weight updates to target network
    new_current_states = np.array([transition[3] for transition in minibatch])/255
    future_qs_list = self.target_model.predict(new_current_states)



    # init
    x = []
    y = []

    # enumerate batches
    for index, (current_state, action, reward, succesor_state, done) in enumerate(minibatch):
        if not done:
            maxfutureQ = np.max(future_qs_list[index])
            # compute new q-value
            newQ = reward + DISCOUNT * maxfutureQ
        else:
            newQ = reward

        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        x.append(current_state)
        y.append(current_qs)

    self.model_fit(np.array(x)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal state else None)

    # update target_network_counter for every M episode
    if terminal_state:
        self.target_update_counter +=1

    if self.target_update_counter > UPDATE_TARGET_EVERY:
        self.target_model.set_weights(self.model.get_weights()
        self.target_update_counter = 0


class ModifiedTensorboard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # overriding method to stop creating default logwriter
    def set_model(self, model):
        pass

    def on_epoch_end(self, epochs, logs=None):
        self.update_stats(**logs)

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# run
agent = DQNAgent()

