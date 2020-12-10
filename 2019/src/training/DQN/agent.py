import copy
import os
import random
from collections import namedtuple, deque, Iterable
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import dill
import h5py
from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, double_dqn=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)

        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.optimizer = keras.optimizers.Adam()
        self.loss_object = keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = keras.metrics.Mean(name="train_loss")
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def save(self, filename):
        keras.models.saved_model.save(self.qnetwork_local, "models/q-local/")
        tf.compat.v2.saved_model.save(self.qnetwork_target, "models/q-target/")
        # self.qnetwork_local.save("models/" + filename + "-local.h5")
        # self.qnetwork_target.save("models/" + filename + "-target.h5")
        # torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        # torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, model_dir):
        self.qnetwork_local = keras.models.saved_model.load(export_dir=f"{model_dir}/q-local")
        self.qnetwork_target = tf.compat.v2.saved_model.load(export_dir=f"{model_dir}/q-target")
        logging.info("Models sucessfully loaded.")

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if train:
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 1)
        action_values = self.qnetwork_local(state)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #     action_values = self.qnetwork_local(state)
        # self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(np.asarray(action_values))
        else:
            return random.choice(np.arange(self.action_size))

    @tf.function
    def train_step(self, model, x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = self.loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(y, predictions)

    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Q_expected = self.qnetwork_local(actions)

        # if self.double_dqn:
        #     # Double DQN
        #     q_best_action = self.qnetwork_local(next_states).max(1)[1]
        #     Q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        # else:
        #     # DQN
        #     # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)
        #     Q_targets_next = tf.math.reduce_max(self.qnetwork_target(next_states))

            # Compute Q targets for current states
        # Q_targets = rewards + (tf.cast(gamma, tf.uint8) * Q_targets_next * (1 - dones))

        self.train_step(self.qnetwork_target, states, actions)
        # self.qnetwork_target.fit(states, actions)
        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        # loss = keras.optimizers.losses.MSE(Q_expected, Q_targets)
        # Minimize the loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        t = experiences[0]

        states = tf.convert_to_tensor(self.__v_stack_impr([e.state for e in experiences if e is not None]))
        actions = tf.convert_to_tensor(self.__v_stack_impr([e.action for e in experiences if e is not None]))
        rewards = tf.convert_to_tensor(self.__v_stack_impr([e.reward for e in experiences if e is not None]))
        next_states = tf.convert_to_tensor(self.__v_stack_impr([e.next_state for e in experiences if e is not None]))
        dones = tf.convert_to_tensor(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
