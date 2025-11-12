import math
import numpy as np

from .networks import actor_network
from .memory import PPOMemory

BATCH_SIZE = 64

class Agent:
    def __init__(self, env, alpha: float = 3e-4, action_std: float = 0.5):
        # PPO hyperparams (kept for compatibility with your pipeline)
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95

        self.env = env
        self.actor = actor_network(env)          # tinygrad network you provided
        self.memory = PPOMemory(BATCH_SIZE)      # your existing memory buffer
        self.action_std = float(action_std)      # std in pre-tanh (z) space

    def choose_action(self, observation, deterministic: bool = False):
        """
        observation: np.ndarray shaped like (N, 4, 3) or anything flattenable to obs_dim.

        Returns:
          action   -> np.ndarray shaped like env.action_space.shape, scaled to [low, high]
          log_prob -> float, log π(a|s), with tanh squash correction
          value    -> float (critic value). We return 0.0 since no critic here.
        """
        # ---- prep input
        obs_np = np.asarray(observation, dtype=np.float32).reshape(1, -1)

        # ---- actor forward: returns Tensor in [-1,1]
        mu_u = self.actor.forward(obs_np).numpy().reshape(-1)   # mean in action (squashed) space

        # ---- map to pre-tanh (z) space so we can sample a proper Gaussian and apply correction
        eps = 1e-6
        mu_u = np.clip(mu_u, -1.0 + eps, 1.0 - eps)            # avoid atanh blow-up
        mu_z = np.arctanh(mu_u)                                 # tanh(mu_z) ≈ mu_u

        # ---- sample in z-space, then squash
        if deterministic:
            z = mu_z
        else:
            z = mu_z + np.random.normal(loc=0.0, scale=self.action_std, size=mu_z.shape)

        u = np.tanh(z)                                          # squashed action in [-1,1]

        # ---- log prob with tanh correction
        # base log prob in z-space (Gaussian)
        # log N(z | mu_z, std^2) = -0.5 * [ ((z-mu_z)/std)^2 + 2 log std + log(2π) ]
        base = -0.5 * np.sum(((z - mu_z) / self.action_std) ** 2
                              + 2 * math.log(self.action_std)
                              + math.log(2 * math.pi))
        # tanh squash correction: sum log(1 - tanh(z)^2)
        squash_correction = np.sum(np.log(1.0 - u * u + 1e-8))
        log_prob = float(base - squash_correction)

        # ---- rescale from [-1,1] to [low, high], reshape to env action shape
        low  = np.asarray(self.env.action_space.low,  dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        scaled = low + (u + 1.0) * 0.5 * (high - low)
        action = scaled.reshape(self.env.action_space.shape)

        # ---- no critic in this tinygrad rewrite (return 0.0 to keep interface)
        value = 0.0

        return action, log_prob, value
