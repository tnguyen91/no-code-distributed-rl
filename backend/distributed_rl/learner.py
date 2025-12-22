import queue
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

def get_device():
    return torch.device("cpu")

BATCH_SIZE_PER_ACTOR = 32
GAMMA = 0.99
LR = 3e-4
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
RHO_BAR = 1.0
C_BAR = 1.0

def a2c_learner_loop(
    experience_queue: mp.Queue,
    episode_stats_queue: mp.Queue,
    metrics_list,
    shared_model: ActorCritic,
    num_actors: int = 2,
):
    device = get_device()
    print(f"[A2C] Using device: {device}", flush=True)

    batch_size = BATCH_SIZE_PER_ACTOR * num_actors
    optimizer = optim.Adam(shared_model.parameters(), lr=LR)
    update_count = 0
    recent_episodes = deque(maxlen=100)

    while True:
        batch = []
        while len(batch) < batch_size:
            batch.append(experience_queue.get())

        while not episode_stats_queue.empty():
            try:
                stats = episode_stats_queue.get_nowait()
                recent_episodes.append(stats["episode_reward"])
            except queue.Empty:
                break

        obs = torch.as_tensor(np.stack([e["obs"] for e in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([e["action"] for e in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([e["reward"] for e in batch], dtype=torch.float32, device=device)
        dones = torch.as_tensor([float(e["done"]) for e in batch], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack([e["next_obs"] for e in batch]), dtype=torch.float32, device=device)
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
    num_actors: int = 2,
):
    device = get_device()
    print(f"[PPO] Using device: {device}", flush=True)

    batch_size = BATCH_SIZE_PER_ACTOR * num_actors
    optimizer = optim.Adam(shared_model.parameters(), lr=LR)
    update_count = 0
    recent_episodes = deque(maxlen=100)

    while True:
        batch = []
        while len(batch) < batch_size:
            batch.append(experience_queue.get())

        while not episode_stats_queue.empty():
            try:
                stats = episode_stats_queue.get_nowait()
                recent_episodes.append(stats["episode_reward"])
            except queue.Empty:
                break

        obs = torch.as_tensor(np.stack([e["obs"] for e in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([e["action"] for e in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([e["reward"] for e in batch], dtype=torch.float32, device=device)
        dones = torch.as_tensor([float(e["done"]) for e in batch], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack([e["next_obs"] for e in batch]), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor([e["log_prob"] for e in batch], dtype=torch.float32, device=device)
        behavior_values = torch.as_tensor([e["value"] for e in batch], dtype=torch.float32, device=device)
        masks = 1.0 - dones

        with torch.no_grad():
            _, next_values = shared_model(next_obs)
        next_values = next_values.squeeze(-1)

        targets = rewards + GAMMA * next_values * masks
        advantages = targets - behavior_values

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

def vtrace_learner_loop(
    experience_queue: mp.Queue,
    episode_stats_queue: mp.Queue,
    metrics_list,
    shared_model: ActorCritic,
    num_actors: int = 2,
):
    device = get_device()
    print(f"[V-trace] Using device: {device}", flush=True)

    batch_size = BATCH_SIZE_PER_ACTOR * num_actors
    optimizer = optim.Adam(shared_model.parameters(), lr=LR)
    update_count = 0
    recent_episodes = deque(maxlen=100)

    while True:
        batch = []
        while len(batch) < batch_size:
            batch.append(experience_queue.get())

        while not episode_stats_queue.empty():
            try:
                stats = episode_stats_queue.get_nowait()
                recent_episodes.append(stats["episode_reward"])
            except queue.Empty:
                break

        obs = torch.as_tensor(np.stack([e["obs"] for e in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([e["action"] for e in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([e["reward"] for e in batch], dtype=torch.float32, device=device)
        dones = torch.as_tensor([float(e["done"]) for e in batch], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack([e["next_obs"] for e in batch]), dtype=torch.float32, device=device)
        behavior_log_probs = torch.as_tensor([e["log_prob"] for e in batch], dtype=torch.float32, device=device)
        behavior_values = torch.as_tensor([e["value"] for e in batch], dtype=torch.float32, device=device)
        masks = 1.0 - dones

        # Get current policy's log probs and values
        logits, values = shared_model(obs)
        values = values.squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        target_log_probs = dist.log_prob(actions)

        with torch.no_grad():
            _, next_values = shared_model(next_obs)
        next_values = next_values.squeeze(-1)

        # Compute importance sampling ratios
        log_rhos = target_log_probs.detach() - behavior_log_probs
        rhos = torch.exp(log_rhos)

        # Clip importance sampling ratios (V-trace truncation)
        clipped_rhos = torch.clamp(rhos, max=RHO_BAR)

        # V-trace targets: v_s = V_behavior(s) + clipped_ρ * δV
        # where δV = r + γV(s') - V_behavior(s)
        td_errors = rewards + GAMMA * next_values * masks - behavior_values
        vtrace_targets = behavior_values + clipped_rhos * td_errors

        advantages = vtrace_targets - behavior_values

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient with importance sampling
        policy_loss = -(target_log_probs * clipped_rhos.detach() * advantages).mean()
        value_loss = (vtrace_targets.detach() - values).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shared_model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        update_count += 1
        avg_rho = rhos.mean().item()
        avg_episode_reward = np.mean(recent_episodes) if recent_episodes else 0.0
        add_metric_to_list(metrics_list, update_count, avg_episode_reward)
        print(f"[V-trace] Update={update_count} Loss={loss.item():.3f} AvgRho={avg_rho:.2f} AvgEpReward={avg_episode_reward:.1f}", flush=True)

def start_distributed(
    exp_id: str,
    num_actors: int = 2,
    env_id: str = "CartPole-v1",
    algorithm: str = "ppo",
) -> Tuple[mp.Process, List[mp.Process], ActorCritic]:
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    shared_model = ActorCritic(obs_dim, act_dim)
    shared_model.share_memory()

    experience_queue: mp.Queue = mp.Queue(maxsize=10_000)
    episode_stats_queue: mp.Queue = mp.Queue(maxsize=1_000)
    metrics_list = init_experiment_metrics(exp_id)

    if algorithm == "ppo":
        learner_fn = ppo_learner_loop
    elif algorithm == "vtrace":
        learner_fn = vtrace_learner_loop
    else:
        learner_fn = a2c_learner_loop
    learner = mp.Process(
        target=learner_fn,
        args=(experience_queue, episode_stats_queue, metrics_list, shared_model, num_actors),
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

    return learner, actors, shared_model
