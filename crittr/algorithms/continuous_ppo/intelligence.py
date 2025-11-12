import math
import numpy as np
from tinygrad.tensor import Tensor

from .networks import actor_network, critic_network , AdamOpt
from .memory import PPOMemory

BATCH_SIZE = 64

class Agent:
    def __init__(self, env, alpha: float = 3e-4, action_std: float = 0.5):
        # PPO hyperparams
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95

        self.env = env
        self.actor = actor_network(env)                   # returns u in [-1,1] (tanh)
        self.action_std = float(action_std)               # scalar std in z-space
        self.memory = PPOMemory(BATCH_SIZE)

        # critic
        self.obs_dim = self.env.observation_space.flatten()
        self.critic = critic_network(self.obs_dim, lr=alpha)

        # If your actor has its own optimizer internally, great; else add:
        if not hasattr(self.actor, "optimizer"):
            self.actor.optimizer = AdamOpt(self.actor.params, lr=alpha)

    # ---------- action selection (same as before, tinygrad-friendly) ----------
    def choose_action(self, observation, deterministic: bool = False):
        obs_np = np.asarray(observation, dtype=np.float32).reshape(1, -1)

        # actor forward gives μ_u in [-1,1]
        mu_u = self.actor.forward(obs_np).numpy().reshape(-1)

        # map to z-space
        eps = 1e-6
        mu_u = np.clip(mu_u, -1.0 + eps, 1.0 - eps)
        mu_z = np.arctanh(mu_u)

        # sample in z-space, squash
        if deterministic:
            z = mu_z
        else:
            z = mu_z + np.random.normal(0.0, self.action_std, size=mu_z.shape)
        u = np.tanh(z)

        # log prob with tanh correction
        base = -0.5 * np.sum(((z - mu_z) / self.action_std) ** 2
                             + 2 * math.log(self.action_std)
                             + math.log(2 * math.pi))
        squash_correction = np.sum(np.log(1.0 - u * u + 1e-8))
        log_prob = float(base - squash_correction)

        # rescale to env bounds
        low  = np.asarray(self.env.action_space.low,  dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        scaled = low + (u + 1.0) * 0.5 * (high - low)
        action = scaled.reshape(self.env.action_space.shape)

        return action, log_prob, 0.0  # value computed during learn

    # ---------- helpers used in learn ----------
    def _atanh_t(self, x):
        # atanh(x) = 0.5 * ln((1+x)/(1-x))
        eps = 1e-8
        return 0.5 * ((1 + x + eps) / (1 - x + eps)).log()

    def _log_prob_current_policy(self, states_t: Tensor, actions_t: Tensor, low_t: Tensor, high_t: Tensor):
        """
        states_t: (B, obs_dim)
        actions_t: (B, A) scaled to [low, high]
        returns: (B,) log_probs under current actor with tanh correction
        """
        # u in [-1,1] from scaled action
        u = 2.0 * (actions_t - low_t) / (high_t - low_t) - 1.0
        u = u.clip(-0.999999, 0.999999)

        # actor outputs μ_u in [-1,1]
        mu_u = self.actor.forward(states_t)             # (B, A)
        mu_u = mu_u.clip(-0.999999, 0.999999)

        # map to z-space
        z = self._atanh_t(u)                            # (B, A)
        mu_z = self._atanh_t(mu_u)                      # (B, A)

        # Gaussian log prob in z-space (diag covariance = action_std^2 I)
        std = Tensor([self.action_std], requires_grad=False)
        two_pi = Tensor([2.0 * math.pi], requires_grad=False)

        # ((z - mu_z)/std)^2 term
        quad = ((z - mu_z) / std) ** 2

        # proper Gaussian log-prob: -0.5 * [∑((z−μ)/σ)^2 + D·log(2πσ²)]
        D = z.shape[-1]
        const = math.log(2.0 * math.pi) + 2.0 * math.log(self.action_std)
        base = -0.5 * (quad.sum(axis=1) + D * const)

        # tanh correction
        squash_corr = (1.0 - (u * u) + 1e-8).log().sum(axis=1)

        logp = base - squash_corr

        return logp

    def _critic_values(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float32)
        x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x.reshape(1, -1)
        v = self.critic.forward(x).reshape(-1)
        if hasattr(v, "detach"): v = v.detach()
        return v.numpy()
                                    # materialize to cpu numpy

    # ---------- PPO learn (tinygrad) ----------
    def learn(self):
        Tensor.training = True
        # ---- fetch rollouts ----
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()

        # ---- critic bootstrap (recompute values) ----
        values = self._critic_values(state_arr).astype(np.float32)   # shape (T,)

        # ---- GAE advantage ----
        T = len(reward_arr)
        advantage = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T - 1)):
            mask  = 1.0 - float(dones_arr[t])
            delta = reward_arr[t] + self.gamma * values[t + 1] * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantage[t] = gae
        returns = advantage + values                                  # shape (T,)

        # ---- action bounds as tensors (shared) ----
        low_t  = Tensor(np.asarray(self.env.action_space.low,  dtype=np.float32))
        high_t = Tensor(np.asarray(self.env.action_space.high, dtype=np.float32))

        for _ in range(self.n_epochs):
            for batch in batches:
                # ---- batch tensors (ensure 2D for states/actions) ----
                states_np  = state_arr[batch].astype(np.float32)
                actions_np = action_arr[batch].astype(np.float32)

                states_t   = Tensor(states_np.reshape(states_np.shape[0], -1))   # (B, obs_dim)
                actions_t  = Tensor(actions_np.reshape(actions_np.shape[0], -1)) # (B, A)
                old_logp_t = Tensor(old_prob_arr[batch].astype(np.float32))      # (B,)
                adv_t      = Tensor(advantage[batch].astype(np.float32))         # (B,)
                ret_t      = Tensor(returns[batch].astype(np.float32))           # (B,)

                # normalize advantages
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                # ---- actor: PPO clipped surrogate ----
                # new_logp depends on current actor params; include bounds for scaling
                new_logp_t = self._log_prob_current_policy(states_t, actions_t, low_t, high_t)  # (B,)
                ratio      = (new_logp_t - old_logp_t).exp()                                    # (B,)

                surr1 = ratio * adv_t
                clipped_ratio = ratio.clip(1.0 - self.policy_clip, 1.0 + self.policy_clip)
                surr2 = clipped_ratio * adv_t

                # elementwise min without Tensor.minimum: min(a,b) = 0.5*(a+b - |a-b|)
                min_surr = 0.5 * (surr1 + surr2 - (surr1 - surr2).abs())
                actor_loss = -min_surr.mean()

                # ---- critic: MSE(returns - V) ----
                v_pred = self.critic.forward(states_t).reshape(-1)                  # (B,)
                critic_loss = ((ret_t - v_pred) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # ---- optimize both nets ----
                self.actor.optimizer.zero_grad()
                self.critic.opt.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.opt.step()

        self.memory.clear_memory()
