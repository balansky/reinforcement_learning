import numpy as np


def discount_episode_rewards(gamma, episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    return discounted_episode_rewards


def normailize_rewards(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    normalized_rewards = (rewards - mean) / (std)
    return normalized_rewards