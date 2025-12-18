import time
from multiprocessing import Process, Queue
from typing import Callable

import gymnasium as gym
import torch

from .model import PolicyNetwork

def actor_loop(actor_id: int, experience_queue: Queue, env_id: str = "CartPole-v1"):
    env = gym.make(env_id)
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim)

    while True:
        done = False
        episode_reward = 0.0

        while not done:
            action, log_prob = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            experience_queue.put({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "log_prob": float(log_prob.detach().numpy()),
                "actor_id": actor_id,
            })

            obs = next_obs

        obs, _ = env.reset()
        time.sleep(0.01)
