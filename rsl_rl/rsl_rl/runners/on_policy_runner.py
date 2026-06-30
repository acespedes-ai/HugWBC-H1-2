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

import time
import os
from collections import deque
import statistics
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        num_actor_obs = self.env.num_partial_obs
        num_critic_obs = self.env.num_obs

        actor_critic = ActorCritic(num_actor_obs,
                                   num_critic_obs,
                                   self.env.num_actions,
                                   **self.policy_cfg).to(self.device)
        
        self.alg = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.env.include_history_steps is not None:
            actor_obs_shape = [self.env.include_history_steps, self.env.num_partial_obs]
        else:
            actor_obs_shape = [self.env.num_partial_obs]

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, actor_obs_shape, [self.env.num_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        metrics = defaultdict(float)
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self._write_custom_scalars_layout()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        critic_obs = self.env.get_privileged_observations()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            metrics = self.alg.update()
            self.env.training_curriculum() 
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            # Collect all keys present in ANY ep_info (not just ep_infos[0]),
            # so sparse metrics like ep_len_disturb/clean are always logged.
            all_keys = set()
            for ep_info in locs['ep_infos']:
                all_keys.update(ep_info.keys())
            for key in sorted(all_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    v = ep_info[key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.Tensor([v])
                    if len(v.shape) == 0:
                        v = v.unsqueeze(0)
                    infotensor = torch.cat((infotensor, v.to(self.device)))
                if len(infotensor) == 0:
                    continue
                # nanmean: skip any NaN entries that slipped through
                value = infotensor[~infotensor.isnan()].mean() if infotensor.isnan().any() else infotensor.mean()
                if value.isnan():
                    continue
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if locs['metrics']:
            for k,v in locs['metrics'].items():
                self.writer.add_scalar('Loss/' + k, v, locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True, load_adaptation=False):
        print("load_path:", path)
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def _write_custom_scalars_layout(self):
        """Write a Custom Scalars layout for TensorBoard.

        Creates a 'CUSTOM SCALARS' tab with the key metrics grouped and
        ordered by relevance — independent of alphabetical SCALARS ordering.
        """
        layout = {
            "1 Locomotion": {
                "Episode Length":        ["Multiline", ["Train/mean_episode_length"]],
                "Tracking Lin Vel":      ["Multiline", ["Episode/rew_tracking_lin_vel"]],
                "Tracking Ang Vel":      ["Multiline", ["Episode/rew_tracking_ang_vel"]],
                "No Fly":               ["Multiline", ["Episode/rew_no_fly"]],
                "Alive":                ["Multiline", ["Episode/rew_alive"]],
                "Termination":          ["Multiline", ["Episode/rew_termination"]],
            },
            "2 Curricula": {
                "Penalty Scale":         ["Multiline", ["Episode/curriculum_scales"]],
                "Max Command X":         ["Multiline", ["Episode/max_command_x"]],
                "Max Command Yaw":       ["Multiline", ["Episode/max_command_yaw"]],
                "Disturb Curriculum":    ["Multiline", ["Episode/disturb_curriculum"]],
            },
            "3 Penalties": {
                "DOF Vel Limits":        ["Multiline", ["Episode/rew_dof_vel_limits"]],
                "DOF Pos Limits":        ["Multiline", ["Episode/rew_dof_pos_limits"]],
                "Base Height":           ["Multiline", ["Episode/rew_base_height"]],
                "Orientation Control":   ["Multiline", ["Episode/rew_orientation_control"]],
                "Stand Still":          ["Multiline", ["Episode/rew_stand_still"]],
                "Action Rate Lower":     ["Multiline", ["Episode/rew_action_rate_lower"]],
                "Action Rate Upper":     ["Multiline", ["Episode/rew_action_rate_upper"]],
                "Torques":              ["Multiline", ["Episode/rew_torques"]],
            },
            "4 Stability": {
                "Feet Contact Forces":   ["Multiline", ["Episode/rew_feet_contact_forces"]],
                "Feet Slip":            ["Multiline", ["Episode/rew_feet_slip"]],
                "Feet Stumble":         ["Multiline", ["Episode/rew_feet_stumble"]],
                "Standing Air":         ["Multiline", ["Episode/rew_standing_air"]],
                "Hip Deviation":        ["Multiline", ["Episode/rew_hip_deviation"]],
                "Shoulder Deviation":   ["Multiline", ["Episode/rew_shoulder_deviation"]],
            },
            "5 Training": {
                "Noise Std":            ["Multiline", ["Policy/mean_noise_std"]],
                "Value Loss":           ["Multiline", ["Loss/value_function"]],
                "Surrogate Loss":       ["Multiline", ["Loss/surrogate"]],
                "Sym Loss":             ["Multiline", ["Loss/sym_loss"]],
                "Priv Recon Loss":      ["Multiline", ["Loss/privileged_recon_loss"]],
                "Learning Rate":        ["Multiline", ["Loss/learning_rate"]],
            },
            "6 Disturb Detail": {
                "Tracking Disturb":     ["Multiline", ["Episode/tracking_disturb"]],
                "Tracking Clean":       ["Multiline", ["Episode/tracking_clean"]],
                "EP Len Disturb":       ["Multiline", ["Episode/ep_len_disturb"]],
                "EP Len Clean":         ["Multiline", ["Episode/ep_len_clean"]],
                "Disturb Active Ratio": ["Multiline", ["Episode/disturb_active_ratio"]],
            },
        }
        self.writer.add_custom_scalars(layout)
    
    
