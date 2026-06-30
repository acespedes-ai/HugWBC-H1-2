"""ARS runner for Isaac Gym environments.

Requires num_envs == 2 * num_pairs:
  envs  0 .. N-1   → policies M + nu*delta_i
  envs  N .. 2N-1  → policies M - nu*delta_i
"""

import os
import time
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.ars import LinearPolicy


class ARSRunner:

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.env      = env
        self.cfg      = train_cfg.runner
        self.device   = device
        self.log_dir  = log_dir
        self.writer   = None

        self.num_pairs    = self.cfg.num_pairs     # N
        self.elite_pairs  = self.cfg.elite_pairs   # b
        self.alpha        = self.cfg.step_size     # α
        self.nu           = self.cfg.noise_std     # ν
        self.max_steps    = self.cfg.rollout_steps

        assert env.num_envs == 2 * self.num_pairs, (
            f"num_envs ({env.num_envs}) must equal 2 * num_pairs ({2*self.num_pairs})")

        self.n_obs = env.num_obs       # BaseTask usa num_obs, no num_observations
        self.n_act = env.num_actions

        self.policy = LinearPolicy(self.n_obs, self.n_act, device,
                                   clip_act=self.cfg.clip_actions)

        self.current_iter = 0
        self.save_interval = self.cfg.save_interval

    # ------------------------------------------------------------------

    def learn(self, num_iterations: int):
        if self.log_dir and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.env.reset()

        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)

        tot = self.current_iter + num_iterations
        for it in range(self.current_iter, tot):
            t0 = time.time()

            # 1. Sample N perturbations
            deltas = [torch.randn(self.n_act, self.n_obs, device=self.device)
                      for _ in range(self.num_pairs)]

            # 2. Rollout 2N policies in parallel
            rewards_pos, rewards_neg, ep_lens = self._rollout(deltas)

            rollout_time = time.time() - t0
            t1 = time.time()

            # 3. Update obs normalizer with all collected obs
            # (already updated inside _rollout)

            # 4. Policy gradient update
            sigma_r_raw = self.policy.update(
                rewards_pos, rewards_neg, deltas,
                self.nu, self.alpha, self.elite_pairs,
                sigma_r_min=getattr(self.cfg, 'sigma_r_min', 1.0),
                max_norm=getattr(self.cfg, 'max_policy_norm', 10.0),
            )

            update_time = time.time() - t1

            # 5. Logging
            all_rews = torch.cat([rewards_pos, rewards_neg])
            rew_buffer.extend(all_rews.cpu().numpy().tolist())
            len_buffer.extend(ep_lens.cpu().numpy().tolist())
            mean_rew = np.mean(rew_buffer)
            mean_len = np.mean(len_buffer)

            if self.log_dir:
                sigma_r_min = getattr(self.cfg, 'sigma_r_min', 1.0)
                eff_lr = self.alpha / (max(sigma_r_raw, sigma_r_min) * self.nu)
                self.writer.add_scalar('ARS/mean_episode_reward', mean_rew, it)
                self.writer.add_scalar('ARS/mean_episode_length', mean_len, it)
                self.writer.add_scalar('ARS/rewards_pos_mean', rewards_pos.mean().item(), it)
                self.writer.add_scalar('ARS/rewards_neg_mean', rewards_neg.mean().item(), it)
                self.writer.add_scalar('ARS/policy_norm', self.policy.M.norm().item(), it)
                self.writer.add_scalar('ARS/sigma_r_raw', sigma_r_raw, it)
                self.writer.add_scalar('ARS/effective_lr', eff_lr, it)
                self.writer.add_scalar('Perf/rollout_time', rollout_time, it)
                self.writer.add_scalar('Perf/update_time',  update_time,  it)

            print(f"[ARS] iter {it:5d}  rew={mean_rew:7.2f}  ep_len={mean_len:6.1f}  "
                  f"r+={rewards_pos.mean():6.2f}  r-={rewards_neg.mean():6.2f}  "
                  f"||M||={self.policy.M.norm():.3f}  "
                  f"[rollout {rollout_time:.1f}s | update {update_time:.2f}s]")

            if it % self.save_interval == 0 and self.log_dir:
                self.save(os.path.join(self.log_dir, f'policy_{it}.npz'))

        self.current_iter = tot
        if self.log_dir:
            self.save(os.path.join(self.log_dir, 'policy_final.npz'))

    # ------------------------------------------------------------------

    def _rollout(self, deltas):
        """Run one episode per delta pair.

        envs 0..N-1   use M + nu*delta_i
        envs N..2N-1  use M - nu*delta_i
        """
        N = self.num_pairs
        rewards_pos = torch.zeros(N, device=self.device)
        rewards_neg = torch.zeros(N, device=self.device)
        ep_lens     = torch.zeros(2 * N, device=self.device)
        done_pos    = torch.zeros(N, dtype=torch.bool, device=self.device)
        done_neg    = torch.zeros(N, dtype=torch.bool, device=self.device)

        self.env.reset()
        obs = self.env.get_observations().to(self.device)  # (2N, obs_dim)

        for step in range(self.max_steps):
            # Update obs normalizer
            self.policy.obs_rms.update(obs)
            obs_norm = self.policy.obs_rms.normalize(obs)  # (2N, obs_dim)

            # Compute actions: env i → +delta_i, env N+i → -delta_i
            actions = torch.zeros(2 * N, self.n_act, device=self.device)
            for i, delta in enumerate(deltas):
                M_p = self.policy.M + self.nu * delta
                M_n = self.policy.M - self.nu * delta
                actions[i]   = torch.clamp(obs_norm[i]   @ M_p.T, -self.cfg.clip_actions, self.cfg.clip_actions)
                actions[N+i] = torch.clamp(obs_norm[N+i] @ M_n.T, -self.cfg.clip_actions, self.cfg.clip_actions)

            obs, _, rews, dones, _ = self.env.step(actions)
            obs = obs.to(self.device)
            rews = rews.to(self.device)
            dones = dones.to(self.device).bool()

            # Accumulate rewards only while episode is still running
            rewards_pos += rews[:N]  * (~done_pos).float()
            rewards_neg += rews[N:]  * (~done_neg).float()
            ep_lens[:N] += (~done_pos).float()
            ep_lens[N:] += (~done_neg).float()

            done_pos |= dones[:N]
            done_neg |= dones[N:]

            if done_pos.all() and done_neg.all():
                break

        return rewards_pos, rewards_neg, ep_lens

    # ------------------------------------------------------------------

    def save(self, path: str):
        np.savez(path, **self.policy.state_dict())
        print(f"[ARS] saved policy → {path}")

    def load(self, path: str):
        d = np.load(path)
        self.policy.load_state_dict({k: d[k] for k in d.files})
        print(f"[ARS] loaded policy ← {path}")
