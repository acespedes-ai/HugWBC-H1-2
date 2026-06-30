"""Augmented Random Search (ARS) — Mania et al. 2018.

Connection to numerical methods:
  - Delta sampling    → random search (stochastic exploration)
  - (r+ - r-) / 2*nu → centered finite differences on the policy parameter space
  - M ← M + alpha*g  → steepest descent update
  - Top-b elite       → partial CEM filtering
"""

import torch
import numpy as np


class RunningMeanStd:
    """Online running mean / std tracker (Welford algorithm, GPU tensors)."""

    def __init__(self, dim: int, device, clip: float = 5.0):
        self.mean  = torch.zeros(dim, device=device)
        self.var   = torch.ones(dim, device=device)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: torch.Tensor):
        """x: (batch, dim)"""
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)
        batch_n    = x.shape[0]

        total_n   = self.count + batch_n
        delta     = batch_mean - self.mean
        new_mean  = self.mean + delta * (batch_n / total_n)
        m_a = self.var   * self.count
        m_b = batch_var  * batch_n
        new_var = (m_a + m_b + delta ** 2 * self.count * batch_n / total_n) / total_n

        self.mean  = new_mean
        self.var   = new_var
        self.count = total_n

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


class LinearPolicy:
    """Linear policy: a = M * normalize(s).

    M has shape (n_act, n_obs) — directly interpretable as a heatmap
    showing which observations drive which joint targets.
    """

    def __init__(self, n_obs: int, n_act: int, device, clip_act: float = 1.0):
        self.M        = torch.zeros(n_act, n_obs, device=device)
        self.obs_rms  = RunningMeanStd(n_obs, device)
        self.clip_act = clip_act
        self.device   = device

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, n_obs) → actions: (batch, n_act)"""
        obs_norm = self.obs_rms.normalize(obs)
        return torch.clamp(obs_norm @ self.M.T, -self.clip_act, self.clip_act)

    def act_with_delta(self, obs: torch.Tensor, delta: torch.Tensor, sign: float) -> torch.Tensor:
        obs_norm = self.obs_rms.normalize(obs)
        M_pert   = self.M + sign * delta
        return torch.clamp(obs_norm @ M_pert.T, -self.clip_act, self.clip_act)

    def update(self,
               rewards_pos:  torch.Tensor,
               rewards_neg:  torch.Tensor,
               deltas:       list,
               nu:           float,
               alpha:        float,
               elite_pairs:  int,
               sigma_r_min:  float = 1.0,
               max_norm:     float = 10.0):
        """ARS policy gradient update.

        Args:
            rewards_pos: (N,) total reward for M + nu*delta_i
            rewards_neg: (N,) total reward for M - nu*delta_i
            deltas:      list of N tensors shaped (n_act, n_obs)
            nu:          perturbation std
            alpha:       step size
            elite_pairs: number of top pairs to use
            sigma_r_min: floor on reward std to prevent update blowup when
                         r+ ≈ r- (e.g. all episodes fall at same speed)
            max_norm:    hard cap on ||M||_F; prevents runaway policy growth
        Returns:
            sigma_r: actual reward std (before floor) for logging
        """
        N = len(deltas)

        # 1. Select top-b pairs by max(r+, r-)
        scores = torch.max(rewards_pos, rewards_neg)
        elite_idx = torch.argsort(scores)[-elite_pairs:]

        # 2. Finite-difference gradient estimate
        grad = torch.zeros_like(self.M)
        for i in elite_idx:
            grad += (rewards_pos[i] - rewards_neg[i]) * deltas[i]
        grad /= elite_pairs

        # 3. Normalize by std of elite rewards (ARS-V2 step).
        # Floor sigma_r at sigma_r_min: without it, when all episodes give
        # similar rewards (sigma_r≈0), the effective lr = alpha/(sigma_r*nu)
        # blows up and drives M to large norms without improving performance.
        elite_rews = torch.cat([rewards_pos[elite_idx], rewards_neg[elite_idx]])
        sigma_r_raw = elite_rews.std().item()
        sigma_r = max(sigma_r_raw, sigma_r_min)

        # 4. Steepest descent update
        self.M = self.M + (alpha / (sigma_r * nu)) * grad

        # 5. Hard cap on policy norm to prevent runaway growth
        M_norm = self.M.norm().item()
        if M_norm > max_norm:
            self.M *= max_norm / M_norm

        return sigma_r_raw

    def state_dict(self) -> dict:
        return {
            'M':         self.M.cpu().numpy(),
            'obs_mean':  self.obs_rms.mean.cpu().numpy(),
            'obs_var':   self.obs_rms.var.cpu().numpy(),
            'obs_count': self.obs_rms.count,
        }

    def load_state_dict(self, d: dict):
        self.M = torch.tensor(d['M'], device=self.device)
        self.obs_rms.mean  = torch.tensor(d['obs_mean'],  device=self.device)
        self.obs_rms.var   = torch.tensor(d['obs_var'],   device=self.device)
        self.obs_rms.count = d['obs_count']
