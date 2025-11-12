import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam

# ---- shared MLP sizes ----
FC1_DIMS = 1024
FC2_DIMS = 512
LEARNING_RATE = 1e-4


# ---------- utilities ----------
def _to_tensor_2d(x: "np.ndarray | Tensor") -> Tensor:
    """
    Ensure x is a tinygrad Tensor of shape (B, D).
    Accepts (D,), (B, D), or higher: (B, ...)->(B, prod(...))
    """
    if not isinstance(x, Tensor):
        x = Tensor(np.asarray(x, dtype=np.float32))
    if len(x.shape) == 1:
        # (D,) -> (1, D)
        x = x.reshape(1, -1)
    elif len(x.shape) > 2:
        # (B, ...)->(B, prod(...))
        x = x.reshape(x.shape[0], -1)
    return x


# ---------- Tinygrad actor network ----------
class actor_network:
    def __init__(self, env):
        self.env = env
        # e.g., obs_dim = N*4*3
        self.obs_dim    = int(self.env.observation_space.flatten())
        # e.g., action_dim = N*4*3
        self.action_dim = int(self.env.action_space.flatten())

        # Xavier-like uniform init
        k1 = math.sqrt(1.0 / self.obs_dim)
        k2 = math.sqrt(1.0 / FC1_DIMS)
        k3 = math.sqrt(1.0 / FC2_DIMS)

        self.w1 = Tensor.uniform(self.obs_dim, FC1_DIMS,            low=-k1, high=k1, requires_grad=True)
        self.b1 = Tensor.zeros(FC1_DIMS,                            requires_grad=True)

        self.w2 = Tensor.uniform(FC1_DIMS, FC2_DIMS,                low=-k2, high=k2, requires_grad=True)
        self.b2 = Tensor.zeros(FC2_DIMS,                            requires_grad=True)

        self.w3 = Tensor.uniform(FC2_DIMS, self.action_dim,         low=-k3, high=k3, requires_grad=True)
        self.b3 = Tensor.zeros(self.action_dim,                     requires_grad=True)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.optimizer = Adam(self.params, lr=LEARNING_RATE)

    def forward(self, x):
        """
        x: np.ndarray or Tensor, shape (D,), (B, D), or (B, ...).
        returns: Tensor (B, action_dim) in [-1, 1]
        """
        x = _to_tensor_2d(x)
        h1 = (x.matmul(self.w1) + self.b1).relu()
        h2 = (h1.matmul(self.w2) + self.b2).relu()
        out = (h2.matmul(self.w3) + self.b3).tanh()
        return out


# ---------- Minimal Adam optimizer (for critic; avoids extra imports) ----------
class AdamOpt:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        # use explicit zeros with param shapes (zeros_like may not be present in all tinygrad builds)
        self.m = [Tensor.zeros(*p.shape) for p in self.params]
        self.v = [Tensor.zeros(*p.shape) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None  # reset grad accumulation

    def step(self):
        self.t += 1
        b1t = self.b1 ** self.t
        b2t = self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = (self.b1 * self.m[i] + (1 - self.b1) * g)
            self.v[i] = (self.b2 * self.v[i] + (1 - self.b2) * (g * g))
            # bias correction
            mhat = self.m[i] * (1.0 / (1.0 - b1t))
            vhat = self.v[i] * (1.0 / (1.0 - b2t))
            upd = mhat * (1.0 / (vhat.sqrt() + self.eps)) * self.lr
            p.assign(p - upd)


# ---------- Tinygrad critic network ----------
class critic_network:
    def __init__(self, obs_dim: int, lr: float = 3e-4):
        # Xavier-like uniform
        k1 = math.sqrt(1.0 / obs_dim)
        k2 = math.sqrt(1.0 / FC1_DIMS)
        k3 = math.sqrt(1.0 / FC2_DIMS)

        self.w1 = Tensor.uniform(obs_dim, FC1_DIMS,  low=-k1, high=k1, requires_grad=True)
        self.b1 = Tensor.zeros(FC1_DIMS,             requires_grad=True)

        self.w2 = Tensor.uniform(FC1_DIMS, FC2_DIMS, low=-k2, high=k2, requires_grad=True)
        self.b2 = Tensor.zeros(FC2_DIMS,             requires_grad=True)

        self.w3 = Tensor.uniform(FC2_DIMS, 1,        low=-k3, high=k3, requires_grad=True)
        self.b3 = Tensor.zeros(1,                     requires_grad=True)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.opt = AdamOpt(self.params, lr=lr)

    def forward(self, x):
        """
        x: np.ndarray or Tensor, shape (D,), (B, D), or (B, ...).
        returns: Tensor shape (B, 1)
        """
        x = _to_tensor_2d(x)
        h1 = (x.matmul(self.w1) + self.b1).relu()
        h2 = (h1.matmul(self.w2) + self.b2).relu()
        out = (h2.matmul(self.w3) + self.b3)  # no activation
        return out
