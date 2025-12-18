import gymnasium as gym
from stable_baselines3 import PPO

def train_cartpole():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("models/cartpole_ppo")
    env.close()

if __name__ == "__main__":
    train_cartpole()