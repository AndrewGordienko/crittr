import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0001


class actor_network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # ints, not tuples/objects
        self.obs_dim    = self.env.observation_space.flatten()   # e.g., N*4*3
        self.action_dim = self.env.action_space.flatten()        # e.g., N*4*3

        self.fc1 = nn.Linear(self.obs_dim, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, obs_dim) or (obs_dim,)
        returns: (batch, action_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, obs_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # in [-1, 1]
        return x

    @torch.no_grad()
    def act(self, obs_np):
        """
        obs_np: numpy array with shape (N, 4, 3) or anything that flattens to obs_dim.
        Returns: numpy action shaped like env.action_space.shape, scaled to env bounds.
        """
        # Flatten observation and make batch of 1
        obs_np = np.asarray(obs_np, dtype=np.float32).reshape(1, -1)
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)

        # Network output in [-1, 1]
        out = self.forward(x)          # (1, action_dim)
        out = out.squeeze(0).cpu().numpy()  # (action_dim,)

        # Scale from [-1, 1] to [low, high]
        low = self.env.action_space.low
        high = self.env.action_space.high

        # low/high can be scalars or arrays; this broadcasts correctly:
        # [-1,1] -> [0,1] -> [low,high]
        scaled = low + (out + 1.0) * 0.5 * (high - low)

        # Reshape to action_space.shape, e.g. (N, 4, 3)
        return scaled.reshape(self.env.action_space.shape)
