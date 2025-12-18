import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.logits = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def act(self, obs):
        # obs: numpy array
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_t)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        return int(action.item()), probs.log_prob(action)
