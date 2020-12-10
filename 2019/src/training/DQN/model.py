"""
This file has the class for the DQN Agent with the NN and the functions required to train it
"""
import tensorflow as tf
from tensorflow import keras


# Initialize the weights
# winit = tf.initializers.GlorotUniform()


class QNetwork(keras.Model):
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(QNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(32, activation="relu")
        self.dense3 = keras.layers.Dense(action_size, activation="softmax")

    # To define the forward pass
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
