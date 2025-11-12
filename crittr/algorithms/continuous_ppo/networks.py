import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam

FC1_DIMS = 1024
FC2_DIMS = 512
LEARNING_RATE = 1e-4

class actor_network:
    def __init__(self, env):
        self.env = env
        self.obs_dim    = self.env.observation_space.flatten()   # e.g., N*4*3
        self.action_dim = self.env.action_space.flatten()        # e.g., N*4*3

        # Xavier-like uniform init
        k1 = math.sqrt(1.0 / self.obs_dim)
        k2 = math.sqrt(1.0 / FC1_DIMS)
        k3 = math.sqrt(1.0 / FC2_DIMS)

        self.w1 = Tensor.uniform(self.obs_dim, FC1_DIMS,  low=-k1, high=k1, requires_grad=True)
        self.b1 = Tensor.zeros(FC1_DIMS,                 requires_grad=True)

        self.w2 = Tensor.uniform(FC1_DIMS, FC2_DIMS,     low=-k2, high=k2, requires_grad=True)
        self.b2 = Tensor.zeros(FC2_DIMS,                 requires_grad=True)

        self.w3 = Tensor.uniform(FC2_DIMS, self.action_dim, low=-k3, high=k3, requires_grad=True)
        self.b3 = Tensor.zeros(self.action_dim,              requires_grad=True)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.optimizer = Adam(self.params, lr=LEARNING_RATE)

    def forward(self, x):
        """
        x: np.ndarray (batch, obs_dim) or (obs_dim,)
        returns: Tensor (batch, action_dim) in [-1, 1]
        """
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        h1 = (x.matmul(self.w1) + self.b1).relu()
        h2 = (h1.matmul(self.w2) + self.b2).relu()
        out = (h2.matmul(self.w3) + self.b3).tanh()
        return out