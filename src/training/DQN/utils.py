"""
This file file contains utility functions, e.g. epsilon greedy policy or populate replay buffer
"""

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
import sys
import tensorflow as tf


# Copy parameters from the Q-network to the target network
def copy_model_parameters(sess, qnet1, qnet2):
    q1_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet1.scope)]
    q1_params = sorted(q1_params, key=lambda v: v.name)
    q2_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet2.scope)]
    q2_params = sorted(q2_params, key=lambda v: v.name)
    update_ops = []
    for q1_v, q2_v in zip(q1_params, q2_params):
        op = q2_v.assign(q1_v)
        update_ops.append(op)
    sess.run(update_ops)


# Epsilon greedy
def epsilon_greedy_policy(qnet, num_actions):
    def policy_fn(sess, observation, epsilon):
        if np.random.rand() < epsilon:
            # Explore -> Every action has the same probability
            A = np.ones(num_actions, dtype=float) / float(num_actions)
        else:
            # Exploit -> Action with highest probability is chosen, based on Q-value
            q_values = qnet.predict(sess, np.expand_dims(observation, 0))[0]
            max_Q_action = np.argmax(q_values)
            A = np.zeros(num_actions, dtype=float)
            A[max_Q_action] = 1.0
        return A
    return policy_fn


# Populate replay memory
def populate_replay_mem(sess, env, state_processor, replay_memory_init_size, policy, epsilon_start, epsilon_end,
                        epsilon_decay_steps, VALID_ACTIONS, Transition):
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    delta_epsilon = (epsilon_start - epsilon_end) / float(epsilon_decay_steps)

    replay_memory = []

    for i in range(replay_memory_init_size):
        epsilon = max(epsilon_start - float(i) * delta_epsilon, epsilon_end)
        action_probs = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        env.render()
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
    return replay_memory


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node: TreeObsForRailEnv.Node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node: TreeObsForRailEnv.Node, current_tree_depth: int, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):

    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4**(remaining_depth+1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes*6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes*4

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree: TreeObsForRailEnv.Node, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation: TreeObsForRailEnv.Node, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs