import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0001

class actor_network(nn.Module):
    def __init__(self, env):
        super().__init__()
        # ints, not tuples/objects
        self.env = env
        self.obs_dim    = self.env.observation_space.flatten()  # e.g., 10*4*3 = 120
        self.action_dim = self.env.action_space.flatten()

        self.fc1 = nn.Linear(self.obs_dim, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x):
        # x must be (batch, obs_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # use tanh if actions are in [-1, 1]
        return x
    
    @torch.no_grad()
    def act(self, x):
        """
        obs_np: numpy with shape (10, 4, 3) or already flattened.
        Returns: numpy action shaped like env.action_space.shape
        """
        x = x.flatten()
        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

        obs_np = self.forward(x)
        if obs_np.ndim > 2:
            obs_np = obs_np.reshape(1, -1)           # (1, 120)
        else:
            obs_np = obs_np.reshape(1, self.obs_dim)  # ensure (1, obs_dim)
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
        out = self.forward(x)                         # (1, action_dim)
        out = out.view(-1).cpu().numpy()
        return out.reshape(self.env.action_space.shape)    # e.g., (10, …)