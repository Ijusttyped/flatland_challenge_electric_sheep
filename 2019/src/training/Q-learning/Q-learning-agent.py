import numpy as np
import time
import sys
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.generators import complex_rail_generator
from flatland.utils.rendertools import RenderTool
import logging
import pickle
import matplotlib.pyplot as plt

start = time.time()

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(name)s:%(module)s:%(funcName)s:line %(lineno)d:%(message)s',
                    filename="runlog.log")
logger = logging.getLogger(__name__)


np.random.seed(42)

n_trials = 50
# Training Parameters
x_dim = 10
y_dim = 10
action_size = 5
n_agents = 1
tree_depth = 1
Q_filename = "Q-values.pkl"

# Set the hyperparameters for training
epsilon = 0.1
alpha = 0.1
gamma = 0.95

# Initialize the q-values
# Q = np.zeros((x_dim, y_dim, action_size), dtype=np.float)
with open(Q_filename, "rb") as f:
    Q = pickle.load(f)

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment
TreeObservation = TreeObsForRailEnv(max_depth=tree_depth)
env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=2, min_dist=5, max_dist=99999),
              obs_builder_object=TreeObservation,
              number_of_agents=n_agents)

env_renderer = RenderTool(env, gl="PILSVG", )

# Given the depth of the tree observation and the number of features per node we get the following state_size
features_per_node = 9
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = features_per_node * nr_nodes

# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent here


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, agt, Q):
        """
        Explore or exploit based on epsilon and epsilon-greedy strategy
        :param state: input is the observation of the agent
        :return: returns an action
        """
        # Always go straight at start
        if agt.old_position is None:
            return 2
        # Get a random number 0<r<1
        x = agt.position[0]
        y = agt.position[1]
        r = np.random.uniform()
        if r < epsilon:
            # explore
            a = self.explore()
        else:
            # exploit
            a = self.exploit(x, y, Q)
        return a

    def step(self, agt, action, reward, Qs1a1, Q, done):
        """
        Step function to improve agent by adjusting policy given the observations
        :param memories: SARS Tuple to be
        :return:
        """
        if done is True:
            return
        x = agt.old_position[0]
        y = agt.old_position[1]
        Q = self.bellman(x, y, action, reward, Qs1a1, Q)
        return

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(Q, f)
        return

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return

    def explore(self):
        return np.random.choice(np.arange(self.action_size))

    def exploit(self, x, y, Q):
        """
        Takes location and gives maximum Q value for it
        """
        a = np.argmax(Q[y, x, :])
        return a

    def bellman(self, x, y, a, reward, Qs1a1, Q):
        """
        Bellman equation to update Q
        """
        # Perfom the bellman update
        Q[y, x, a] = Q[y, x, a] + alpha * (reward + gamma * Qs1a1 - Q[y, x, a])
        return Q

    def max_Q(self, agt, Q):
        x = agt.position[0]
        y = agt.position[1]
        a = np.argmax(Q[y, x, :])
        return Q[y, x, a]


# Initialize the agent with the parameters corresponding to the environment and observation_builder
agent = RandomAgent(state_size, action_size)

# Empty dictionary for all agent action
action_dict = dict()
logger.info("Starting Training...")

for trials in range(1, n_trials + 1):

    # Reset environment and get initial observations for all agents
    obs = env.reset()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    # Run episode
    for step in range(500):
        logger.info(f"Agent data: {env.agents[0]}")
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(agt=env.agents[a], Q=Q)
            action_dict.update({a: action})
            logger.info(f"Action dict: {action_dict}")
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
        logger.info(f"Next observation {next_obs}")
        logger.info(f"Rewards: {all_rewards}")

        # Update Q-table and train agent
        for a in range(env.get_num_agents()):
            Qs1a1 = agent.max_Q(env.agents[a], Q)
            agent.step(agt=env.agents[a],
                       action=action_dict[a],
                       reward=all_rewards[a],
                       Qs1a1=Qs1a1,
                       Q=Q,
                       done=done[a])
            score += all_rewards[a]
        obs = next_obs.copy()
        # time.sleep(0.5)
        if done['__all__']:
            break
    print('Episode Nr. {}\t Score = {}'.format(trials, score))
agent.save(Q_filename)

end = time.time()

print(f"Execution time for {n_trials} trials: {end-start}")

# Print everything
for i in range(action_size):
    plt.subplot(action_size, 1, i + 1)
    plt.imshow(Q[:, :, i])
    plt.axis('off')
    plt.colorbar()
    if i == 0:
        plt.title('Q-nothing')
    elif i == 1:
        plt.title('Q-left')
    elif i == 2:
        plt.title('Q-forward')
    elif i == 3:
        plt.title('Q-right')
    elif i == 4:
        plt.title('Q-stop')
plt.savefig('Q_Q-learning.png')
plt.clf()
plt.close()
