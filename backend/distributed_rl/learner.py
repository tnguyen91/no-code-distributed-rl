from multiprocessing import Process, Queue
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import ActorCritic
from .actors import actor_loop

BATCH_SIZE = 2048
GAMMA = 0.99
LR = 3e-4

def compute_returns(rewards, dones, last_value, gamma=GAMMA):
    returns = []
    R = last_value
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d)
        returns.append(R)
    returns.reverse()
    return np.array(returns, dtype=np.float32)

def learner_loop(experience_queue: Queue):
    model = None
    optimizer = None

    episodes_seen = 0
    while True:
        batch = []
        while len(batch) < BATCH_SIZE:
            exp = experience_queue.get()
            batch.append(exp)

        # set up model if needed
        if model is None:
            obs_dim = len(batch[0]["obs"])
            act_dim = max(e["action"] for e in batch) + 1
            model = ActorCritic(obs_dim, act_dim)
            optimizer = optim.Adam(model.parameters(), lr=LR)

        # build tensors
        obs = torch.as_tensor(
            np.stack([e["obs"] for e in batch]), dtype=torch.float32
        )
        actions = torch.as_tensor(
            [e["action"] for e in batch], dtype=torch.int64
        )
        rewards = np.array([e["reward"] for e in batch], dtype=np.float32)
        dones = np.array([float(e["done"]) for e in batch], dtype=np.float32)

        # simple no-bootstrap: last_value = 0
        last_value = 0.0
        returns = torch.as_tensor(
            compute_returns(rewards, dones, last_value), dtype=torch.float32
        )

        logits, values = model(obs)
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

        avg_return = returns.mean().item()
        episodes_seen += 1
        print(f"[Learner] Loss={loss.item():.3f} AvgReturn={avg_return:.2f} EpisodesSeen~{episodes_seen}")

def start_distributed(num_actors: int = 2) -> Tuple[Process, List[Process]]:
    experience_queue: Queue = Queue(maxsize=10_000)

    learner = Process(target=learner_loop, args=(experience_queue,))
    learner.start()

    actors: List[Process] = []
    from .actors import actor_loop
    for i in range(num_actors):
        p = Process(target=actor_loop, args=(i, experience_queue))
        p.start()
        actors.append(p)

    # Don't join here; caller will manage lifecycle
    return learner, actors