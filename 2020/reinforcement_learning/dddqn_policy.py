import copy
import os
import pickle
import random
from collections import namedtuple, deque, Iterable

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reinforcement_learning.model import DuelingQNetwork
from reinforcement_learning.policy import Policy
from replay_buffers.prioritized_exp_replay import PrioritizedReplayBuffer


# prioritized experience replay
UPDATE_MEM_EVERY = 10          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000     # how often to update the hyperparameters


class DDDQNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = 1
        self.compute_weights = True

        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size
            self.experiences_per_sampling = math.ceil(self.batch_size * UPDATE_MEM_EVERY / self.update_every)

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            # print("🐢 Using CPU")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = PrioritizedReplayBuffer(
                action_size=action_size,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                experiences_per_sampling=self.experiences_per_sampling,
                seed=42,
                device=self.device,
                compute_weights=self.compute_weights,
            )
            # Initialize time step
            self.t_step = 0
            # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
            self.t_step_mem_par = 0
            # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
            self.t_step_mem = 0
            self.loss = 0.0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Update the parameters
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and self.memory.experience_count > self.experiences_per_sampling:
                try:
                    experiences = self.memory.sample()
                    self._learn(experiences)
                except Exception as e:
                    print("------------ERROR---------- Could not learn because of " + str(e))

        # Update the memory sampling
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def _learn(self, experiences):
        # experiences = self.memory.sample()
        states, actions, rewards, next_states, dones, weights, indices = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, self.loss.data.cpu().numpy()))
            self.loss *= weight

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # ------------------- update priorities ------------------- #
        delta = abs(q_expected - q_targets).cpu().detach().numpy()
        # print("delta", delta)
        self.memory.update_priorities(delta, indices)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        try:
            experiences = self.memory.sample()
            self._learn(experiences)
        except Exception as e:
            print("------------ERROR---------- Could not learn because of " + str(e))


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
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
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
