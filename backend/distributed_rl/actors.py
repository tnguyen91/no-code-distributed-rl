import time
import torch.multiprocessing as mp

import gymnasium as gym
import torch

from .model import ActorCritic

def actor_loop(
    actor_id: int,
    experience_queue: mp.Queue,
    env_id: str,
    shared_model: ActorCritic,
    episode_stats_queue: mp.Queue = None,
):
    env = gym.make(env_id)
    obs, _ = env.reset()

    episode_reward = 0.0
    episode_length = 0

    while True:
        done = False
        while not done:
            with torch.no_grad():
                action, log_prob, value = shared_model.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            experience_queue.put({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "log_prob": float(log_prob.numpy()),
                "value": float(value.numpy()),
                "actor_id": actor_id,
            })
            obs = next_obs

        if episode_stats_queue is not None:
            episode_stats_queue.put({
                "actor_id": actor_id,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
            })

        episode_reward = 0.0
        episode_length = 0
        obs, _ = env.reset()
        time.sleep(0.001)
