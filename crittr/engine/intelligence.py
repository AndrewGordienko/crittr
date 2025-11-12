import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .networks import actor_network
from .memory import PPOMemory

DEVICE = torch.device("cpu")

BATCH_SIZE = 64

class Agent:
    def __init__(self, env, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95

        self.env = env


        self.actor = actor_network(env)
        self.memory = PPOMemory(BATCH_SIZE)
    
    def choose_action(self, observation, deterministic: bool = False):
        """
        observation: np.ndarray shaped like (N, 4, 3) or anything flattenable to obs_dim.
        Returns:
        action     -> numpy array shaped like env.action_space.shape, scaled to [low, high]
        log_prob   -> float (log π(a|s)) in pre-scale (squashed) space with tanh correction
        value      -> float (critic value) if self.critic exists, else 0.0
        """
        # --- Prep state ---
        obs_np = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        state = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)  # (1, obs_dim)

        # --- Actor output: allow either a Distribution or raw tensor ---
        actor_out = self.actor(state)
        if hasattr(actor_out, "rsample") and hasattr(actor_out, "log_prob"):
            # Actor already returns a torch Distribution over R^d (pre-tanh)
            dist = actor_out
            z = dist.mean if deterministic else dist.rsample()        # (1, action_dim)
            base_log_prob = dist.log_prob(z).sum(dim=-1)              # (1,)
        else:
            # Actor returns raw tensor -> treat as mean and use fixed std
            mu = actor_out                                           # (1, action_dim)
            action_std = getattr(self, "action_std", 0.5)            # tune if needed
            dist = Normal(mu, torch.full_like(mu, action_std))
            z = mu if deterministic else dist.rsample()
            base_log_prob = dist.log_prob(z).sum(dim=-1)

        # --- Squash to [-1,1] with tanh (keeps motors bounded), apply correction ---
        u = torch.tanh(z)  # squashed action in [-1,1]
        # Tanh correction term: sum log(1 - u^2) across action dims
        squash_correction = torch.log(1 - u.pow(2) + 1e-8).sum(dim=-1)
        log_prob = (base_log_prob - squash_correction).squeeze(0).item()

        # --- Rescale from [-1,1] to [low, high] (broadcast-safe), then reshape ---
        low  = torch.as_tensor(self.env.action_space.low,  device=DEVICE, dtype=torch.float32)
        high = torch.as_tensor(self.env.action_space.high, device=DEVICE, dtype=torch.float32)
        scaled = low + (u + 1.0) * 0.5 * (high - low)                 # (1, action_dim)
        action = scaled.squeeze(0).detach().cpu().numpy().reshape(self.env.action_space.shape)

        # --- Critic value (optional) ---
        if hasattr(self, "critic") and self.critic is not None:
            with torch.no_grad():
                value = self.critic(state).squeeze().item()
        else:
            value = 0.0

        return action, log_prob, value
