from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
import numpy as np
from agent import DQNAgent
from utils import normalize_observation
import time


# ------------------------------------------------------
# 1. Setup of the environment
# ------------------------------------------------------
start = time.time()
# Seed for reproducibility
np.random.seed(420)

# Parameters for the environment

x_dim = 5
y_dim = 5
n_agents = 1

# Custom observation builder
tree_depth = 2
tree_obs = TreeObsForRailEnv(max_depth=tree_depth)

# Environment setup
env = RailEnv(
    width=x_dim,
    height=y_dim,
    number_of_agents=n_agents,
    rail_generator=random_rail_generator(),
    obs_builder_object=tree_obs
)

# Render and show the env
env_renderer = RenderTool(env=env)

# ------------------------------------------------------
# 2. Define state & action size
# ------------------------------------------------------

# Calculate the state size (based on number of actions and observations)
features_per_node = env.obs_builder.observation_dim
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = features_per_node * nr_nodes
print(state_size)
# Set the action size (we have 5 discrete actions)
action_size = 5

# ------------------------------------------------------
# 3. Define training parameters & variables to track progress
# ------------------------------------------------------

# Set the training hyperparameters
n_trials = 100
max_steps = 100
batch_size = 32

# Set the parameters for epsilon decay
# Start exploring 100% at the beginning
eps = 1.
# Shift slowly from exploration to exploitation
eps_decay = 0.0005
# Never move to full exploitation, leave some time for exploration
eps_end = 0.998

# Define some variables to keep track of training progress
# Empty dict for all agents
action_dict = dict()
# Score for all rewards
score = 0

# ------------------------------------------------------
# 4. Load the agent
# ------------------------------------------------------
# Load the agent
agent = DQNAgent(state_size=state_size, action_size=action_size)
# Load the weights (if pretrained agent)
# agent.load("run-003.ckpt")
# agent.q_act.set_weights(agent.q_learn.get_weights())

# ------------------------------------------------------
# 5. Main training loop
# ------------------------------------------------------

for trial in range(1, n_trials + 1):

    # Reset the environment
    obs = env.reset()
    obs = obs[0]
    env_renderer.reset()

    # Run a episode (until successful or max number of steps reached)
    for step in range(max_steps):
        # Normalize the observations
        # norm_obs = normalize_observation(obs[0], tree_depth=tree_depth)

        # Agent performs an action
        for _idx in range(n_agents):
            if obs[_idx] is not None:
                norm_obs = normalize_observation(obs[_idx], tree_depth=tree_depth)
                action = agent.act(state=norm_obs, eps=eps)
                action_dict.update({_idx: action})

        # Environment executes action and returns
        #     1. next observations for all agents
        #     2. corresponding rewards for all agents
        #     3. status if the agents are done
        #     4. information about actions, malfunction, speed and status
        next_obs, all_rewards, done, info = env.step(action_dict)
        for _idx in range(n_agents):
            if not done[_idx]:
                next_norm_obs = normalize_observation(next_obs[_idx], tree_depth=tree_depth)
                agent.remember((norm_obs, action_dict[_idx], all_rewards[_idx], next_norm_obs, done[_idx]))

        # Render the environment -> show me what you got!
        env_renderer.render_env(show=True, show_observations=True)

        #  Prepare for new step and stop if agent is done
        obs = next_obs.copy()
        if done["__all__"]:
            break

    # Train the agent
    if len(agent.memory) > batch_size:
        agent.step(batch_size)
    score += all_rewards[0]

    # Epsilon decay
    eps = max(eps_end, eps_decay * eps)

    # Copy weights from Q' to Q
    if trial % 100 == 0:
        agent.q_act.set_weights(agent.q_learn.get_weights())

    # Save weights
    agent.save("run-004.ckpt")

    # Print progress
    print("\rTraining {} Agents on ({},{}).\t Episode {}\t Eps: {}\t Score: {:.3f}\tDones: {:.2f}%".format(
            env.get_num_agents(), x_dim, y_dim,
            trial,
            eps,
            score,
            done[0]), end="")

end = time.time()
print()
print(f"Total runtime: {end - start} seconds.")
