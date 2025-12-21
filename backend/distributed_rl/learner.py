import torch.multiprocessing as mp
from collections import deque
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from metrics_store import init_experiment_metrics, add_metric_to_list
from .model import ActorCritic
from .actors import actor_loop

BATCH_SIZE = 2048
GAMMA = 0.99
LR = 3e-4

def compute_returns(rewards, dones, gamma=GAMMA):
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d)
        returns.append(R)
    returns.reverse()
    return np.array(returns, dtype=np.float32)


def learner_loop(
    experience_queue: mp.Queue,
    episode_stats_queue: mp.Queue,
    metrics_list,
    shared_model: ActorCritic,
):
    optimizer = optim.Adam(shared_model.parameters(), lr=LR)
    update_count = 0
    recent_episodes = deque(maxlen=100)

    while True:
        batch = []
        while len(batch) < BATCH_SIZE:
            batch.append(experience_queue.get())

        while not episode_stats_queue.empty():
            try:
                stats = episode_stats_queue.get_nowait()
                recent_episodes.append(stats["episode_reward"])
            except:
                break

        obs = torch.as_tensor(np.stack([e["obs"] for e in batch]), dtype=torch.float32)
        actions = torch.as_tensor([e["action"] for e in batch], dtype=torch.int64)
        rewards = np.array([e["reward"] for e in batch], dtype=np.float32)
        dones = np.array([float(e["done"]) for e in batch], dtype=np.float32)
        returns = torch.as_tensor(compute_returns(rewards, dones), dtype=torch.float32)

        logits, values = shared_model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        advantages = returns - values.squeeze(-1)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_count += 1
        avg_episode_reward = np.mean(recent_episodes) if recent_episodes else 0.0
        add_metric_to_list(metrics_list, update_count, avg_episode_reward)
        print(f"[Learner] Update={update_count} Loss={loss.item():.3f} AvgEpReward={avg_episode_reward:.1f} Episodes={len(recent_episodes)}", flush=True)

def start_distributed(
    exp_id: str, num_actors: int = 2, env_id: str = "CartPole-v1"
) -> Tuple[mp.Process, List[mp.Process]]:
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    shared_model = ActorCritic(obs_dim, act_dim)
    shared_model.share_memory()

    experience_queue: mp.Queue = mp.Queue(maxsize=10_000)
    episode_stats_queue: mp.Queue = mp.Queue(maxsize=1_000)
    metrics_list = init_experiment_metrics(exp_id)

    learner = mp.Process(
        target=learner_loop,
        args=(experience_queue, episode_stats_queue, metrics_list, shared_model),
    )
    learner.start()

    actors: List[mp.Process] = []
    for i in range(num_actors):
        p = mp.Process(
            target=actor_loop,
            args=(i, experience_queue, env_id, shared_model, episode_stats_queue),
        )
        p.start()
        actors.append(p)

    return learner, actors
