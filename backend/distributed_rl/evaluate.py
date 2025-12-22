import os
from typing import Dict, Optional

import gymnasium as gym
import torch

from .model import ActorCritic


def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    num_episodes_to_record: int = 1,
) -> Dict:
    checkpoint = torch.load(model_path, weights_only=False)
    env_id = checkpoint.get("env_id", "CartPole-v1")

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episode_rewards = []
    episode_lengths = []
    video_paths = []

    episodes_to_record = min(num_episodes_to_record, num_episodes) if record_video else 0

    for ep in range(num_episodes):
        should_record = record_video and ep < episodes_to_record and video_dir

        if should_record:
            os.makedirs(video_dir, exist_ok=True)
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                name_prefix=f"eval-ep{ep}",
                episode_trigger=lambda x: True,
            )
        else:
            env = gym.make(env_id)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        env.close()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if should_record:
            potential_video = os.path.join(video_dir, f"eval-ep{ep}-episode-0.mp4")
            if os.path.exists(potential_video):
                video_paths.append(potential_video)

    return {
        "avg_reward": sum(episode_rewards) / len(episode_rewards),
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "video_paths": video_paths,
        "env_id": env_id,
        "algorithm": checkpoint.get("algorithm", "unknown"),
    }
