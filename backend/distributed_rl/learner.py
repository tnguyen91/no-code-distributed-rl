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

BATCH_SIZE = 64
GAMMA = 0.99
LR = 3e-4
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


def a2c_learner_loop(
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
        rewards = torch.as_tensor([e["reward"] for e in batch], dtype=torch.float32)
        dones = torch.as_tensor([float(e["done"]) for e in batch], dtype=torch.float32)
        next_obs = torch.as_tensor(np.stack([e["next_obs"] for e in batch]), dtype=torch.float32)
        masks = 1.0 - dones

        logits, values = shared_model(obs)
        with torch.no_grad():
            _, next_values = shared_model(next_obs)
        values = values.squeeze(-1)
        next_values = next_values.squeeze(-1)

        targets = rewards + GAMMA * next_values * masks
        advantages = targets - values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = (targets.detach() - values).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shared_model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        update_count += 1
        avg_episode_reward = np.mean(recent_episodes) if recent_episodes else 0.0
        add_metric_to_list(metrics_list, update_count, avg_episode_reward)
        print(f"[A2C] Update={update_count} Loss={loss.item():.3f} AvgEpReward={avg_episode_reward:.1f}", flush=True)


def ppo_learner_loop(
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
        rewards = torch.as_tensor([e["reward"] for e in batch], dtype=torch.float32)
        dones = torch.as_tensor([float(e["done"]) for e in batch], dtype=torch.float32)
        next_obs = torch.as_tensor(np.stack([e["next_obs"] for e in batch]), dtype=torch.float32)
        old_log_probs = torch.as_tensor([e["log_prob"] for e in batch], dtype=torch.float32)
        masks = 1.0 - dones

        with torch.no_grad():
            _, next_values = shared_model(next_obs)
            _, old_values = shared_model(obs)
        next_values = next_values.squeeze(-1)
        old_values = old_values.squeeze(-1)

        targets = rewards + GAMMA * next_values * masks
        advantages = targets - old_values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            logits, values = shared_model(obs)
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (targets - values).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        update_count += 1
        avg_episode_reward = np.mean(recent_episodes) if recent_episodes else 0.0
        add_metric_to_list(metrics_list, update_count, avg_episode_reward)
        print(f"[PPO] Update={update_count} Loss={loss.item():.3f} AvgEpReward={avg_episode_reward:.1f}", flush=True)

def start_distributed(
    exp_id: str,
    num_actors: int = 2,
    env_id: str = "CartPole-v1",
    algorithm: str = "ppo",
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

    learner_fn = ppo_learner_loop if algorithm == "ppo" else a2c_learner_loop
    learner = mp.Process(
        target=learner_fn,
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
