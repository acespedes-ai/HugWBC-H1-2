# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 use_wbc_sym_loss=False,
                 symmetry_loss_coef=0.5,
                 sync_update=False,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 robot_type = 'h1'
                 ):

        self.device = device
        self.robot_type = robot_type

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.use_wbc_sym_loss = use_wbc_sym_loss
        self.symmetry_loss_coef = symmetry_loss_coef
        self.sync_update = sync_update

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self._build_symmetry_matrices()

    def _build_symmetry_matrices(self):
        if self.robot_type == 'h1_2':
            # H1-2 DOF order (27): l_hip_yaw/pitch/roll, l_knee, l_ankle_pitch/roll,
            #   r_hip_yaw/pitch/roll, r_knee, r_ankle_pitch/roll, torso,
            #   l_sh_pitch/roll/yaw, l_elbow, l_wr_roll/pitch/yaw,
            #   r_sh_pitch/roll/yaw, r_elbow, r_wr_roll/pitch/yaw
            # Option B: 21 actions (12 legs + 1 torso + 4 left arm + 4 right arm, no wrists)
            # Action order: l_hip_yaw/pitch/roll, l_knee, l_ankle_pitch/roll (0-5)
            #               r_hip_yaw/pitch/roll, r_knee, r_ankle_pitch/roll (6-11)
            #               torso (12)
            #               l_sh_pitch/roll/yaw, l_elbow (13-16)
            #               r_sh_pitch/roll/yaw, r_elbow (17-20)
            actions_permutation = torch.tensor([
                -6,  7, -8,  9, 10, -11,          # left leg (0-5)   → right leg
                -0.001, 1, -2,  3,  4,  -5,        # right leg (6-11) → left leg
                -12,                                # torso (12)       → torso (negated)
                17, -18, -19, 20,                   # left arm (13-16) → right arm
                13, -14, -15, 16,                   # right arm (17-20)→ left arm
            ])
            # Option B partial_obs = 81 + 11 + 2 = 94 dims
            # Layout: ang_vel(3) gravity(3) dof_pos(27) dof_vel(27) actions(21) cmds(11) clock(2)
            observations_permutation = torch.tensor([
                # [0:6] ang_vel + gravity
                -0.0001, 1, -2, 3, -4, 5,
                # [6:33] dof_pos  (obs_idx = 6 + dof_idx) — all 27 DOFs still observed
                -12, 13, -14, 15, 16, -17,
                -6, 7, -8, 9, 10, -11,
                -18,
                26, -27, -28, 29, -30, 31, -32,
                19, -20, -21, 22, -23, 24, -25,
                # [33:60] dof_vel (obs_idx = 33 + dof_idx) — all 27 DOFs still observed
                -39, 40, -41, 42, 43, -44,
                -33, 34, -35, 36, 37, -38,
                -45,
                53, -54, -55, 56, -57, 58, -59,
                46, -47, -48, 49, -50, 51, -52,
                # [60:81] actions (obs_idx = 60 + action_idx, 21 non-wrist DOFs)
                -66, 67, -68, 69, 70, -71,          # left leg (act 0-5)
                -60, 61, -62, 63, 64, -65,           # right leg (act 6-11)
                -72,                                  # torso (act 12)
                77, -78, -79, 80,                     # left arm (act 13-16) → right arm
                73, -74, -75, 76,                     # right arm (act 17-20) → left arm
                # [81:92] commands (11): vx, vy, yaw, freq, phase, dur, swing_h, body_h, pitch, waist, interrupt
                81, -82, -83, 84, 85, 86, 87, 88, 89, -90, 91,
                # [92:94] clock: left_foot, right_foot → swap
                93, 92,
            ])
        else:
            # H1 DOF order (19): l_hip_yaw/roll/pitch, l_knee, l_ankle,
            #   r_hip_yaw/roll/pitch, r_knee, r_ankle, torso,
            #   l_sh_pitch/roll/yaw, l_elbow, r_sh_pitch/roll/yaw, r_elbow
            actions_permutation = torch.tensor([
                -5, -6, 7, 8, 9, -0.001, -1, 2, 3, 4, -10,
                15, -16, -17, 18, 11, -12, -13, 14,
            ])
            # H1int partial_obs = 63 + 11 + 2 = 76 dims
            observations_permutation = torch.tensor([
                -0.0001, 1, -2, 3, -4, 5,
                -11, -12, 13, 14, 15, -6, -7, 8, 9, 10, -16, 21, -22, -23, 24, 17, -18, -19, 20,
                -30, -31, 32, 33, 34, -25, 26, 27, 28, 29, -35, 40, -41, -42, 43, 36, -37, -38, 39,
                -49, -50, 51, 52, 53, -44, -45, 46, 47, 48, -54, 59, -60, -61, 62, 55, -56, -57, 58,
                63, -64, -65, 66, 67, 68, 69, 70, 71, -72, 73, 75, 74,
            ])

        n_act = len(actions_permutation)
        n_obs = len(observations_permutation)
        self.act_perm_mat = torch.zeros(n_act, n_act, requires_grad=False, device=self.device)
        self.obs_perm_mat = torch.zeros(n_obs, n_obs, requires_grad=False, device=self.device)
        for i, perm in enumerate(actions_permutation):
            self.act_perm_mat[i][int(torch.abs(perm))] = torch.sign(perm)
        for i, perm in enumerate(observations_permutation):
            self.obs_perm_mat[i][int(torch.abs(perm))] = torch.sign(perm)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs=critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        # self.adaptation_module.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        metrics = defaultdict(float)
        adaptation_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, 
                                      masks=masks_batch,
                                      privileged_obs=critic_obs_batch,
                                      sync_update=self.sync_update)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                if self.sync_update:
                    adaptation_loss = self.actor_critic.actor.compute_adaptation_pred_loss(metrics)

                # symmetry loss — matrices built once in __init__ via _build_symmetry_matrices()
                act_perm_mat = self.act_perm_mat
                obs_perm_mat = self.obs_perm_mat

                origin_act, _ = self.actor_critic.act_inference(obs_batch, masks=masks_batch, privileged_obs=critic_obs_batch)
                n_obs_perm = obs_perm_mat.shape[0]
                mirror_partial_obs_batch = torch.matmul(obs_batch[..., :n_obs_perm], obs_perm_mat)
                mirror_obs_batch = torch.cat((mirror_partial_obs_batch, obs_batch[..., n_obs_perm:]), dim=-1)
                mirror_act, _ = self.actor_critic.act_inference(mirror_obs_batch, masks=masks_batch, privileged_obs=critic_obs_batch)
                recovery_act = torch.matmul(mirror_act, act_perm_mat)

                sym_loss = self.symmetry_loss_coef * (origin_act.detach() - recovery_act).pow(2).mean()

                if self.sync_update:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + sym_loss + adaptation_loss
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + sym_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['value_function'] += value_loss.item()
                metrics['surrogate'] += surrogate_loss.item()
                metrics['actor_sample_ratio'] += ratio.mean().item()
                metrics['sym_loss'] += sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        
        for k in metrics.keys():
            metrics[k] /= num_updates

        self.storage.clear()

        return metrics
