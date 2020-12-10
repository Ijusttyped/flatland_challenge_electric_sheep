import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random
import logging
import datetime


logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(module)s:%(funcName)s:line %(lineno)d:%(message)s")
logger = logging.getLogger(__name__)


class DQNAgent:

    def __init__(self, state_size, action_size):
        """ Initializes the Agent """
        self.state_size = state_size
        self.action_size = action_size
        # Only store last 2000 (s, a) pairs
        self.memory = deque(maxlen=2000)
        # Discount factor for future rewards
        self.gamma = 0.95
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.q_act = self._build_model()
        self.q_learn = self._build_model()

    def _build_model(self):
        """ Builds the model """
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, activation="relu", input_shape=(self.state_size,)))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, memories):
        """ Stores example to replay buffer """
        self.memory.append(memories)

    def act(self, state, eps):
        """ Executes an action either random or based on observation """
        if np.random.rand() <= eps:
            return random.choice(np.arange(self.action_size))
        state = np.reshape(state, (1, self.state_size))
        action_values = self.q_act.predict(state)
        action = np.argmax(np.asarray(action_values))
        return action

    def step(self, batch_size):
        """ Samples from replay buffer and trains the model """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, (1, self.state_size))
                next_q_vals = self.q_learn.predict(next_state)
                target = (reward + self.gamma * np.amax(next_q_vals[0]))
            state = np.reshape(state, (1, self.state_size))
            target_f = self.q_learn.predict(state)
            target_f[0][action] = target
            self.q_learn.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.tensorboard_callback])

    def save(self, filename):
        """ Stores model weights """
        self.q_learn.save_weights(f"saved_weights/{filename}")

    def load(self, filename):
        """ Loads model weights """
        self.q_learn.load_weights(f"saved_weights/{filename}")
