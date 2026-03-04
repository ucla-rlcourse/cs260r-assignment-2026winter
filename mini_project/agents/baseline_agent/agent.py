"""Standalone racing agent converted from SB3 PPO checkpoint."""

import os
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """MLP policy network matching SB3 PPO architecture."""

    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        return self.action_mean(x)


class Policy:
    CREATOR_NAME = "Baseline Agent"
    CREATOR_UID = "123456789"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.action_dim = checkpoint["action_dim"]
        hidden_sizes = checkpoint["hidden_sizes"]

        self.model = PolicyNetwork(self.obs_dim, self.action_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)
