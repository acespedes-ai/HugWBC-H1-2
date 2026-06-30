import torch
from isaacgym.torch_utils import torch_rand_float
from legged_gym.envs.h1_2.h1_2 import H1_2Robot
from legged_gym.envs.h1_2.h1_2_walk_config import H1_2WalkCfg, WALK_NUM_OBS, WALK_PROP_DIM


class H1_2WalkRobot(H1_2Robot):
    """H1-2 with wrists fixed to default_dof_pos.

    Policy controls 21 DOFs (legs×12 + torso×1 + shoulders+elbows×8).
    Wrists (6 DOFs) are commanded to default_dof_pos every step.
    Observation is 74-dim flat: 69 proprioception + 3 cmd (vx,vy,yaw) + 2 clock.
    """

    def __init__(self, cfg: H1_2WalkCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Buffer init: p_gains/d_gains must span all 27 DOFs even though
    # the policy only outputs 13 actions.
    # ------------------------------------------------------------------

    def _init_buffers(self):
        policy_act = self.num_actions   # 21 (from config)
        self.num_actions = self.num_dof  # temporarily 27 so parent sizes PD buffers correctly
        super()._init_buffers()
        self.num_actions = policy_act   # restore to 21

        # Controlled DOF = all joints EXCEPT wrists (wrists fixed to default_dof_pos).
        # Computed dynamically so it works regardless of DOF ordering in URDF.
        names_lower = [n.lower() for n in self.dof_names]
        wrist_set = {i for i, n in enumerate(names_lower) if 'wrist' in n}
        ctrl_list = [i for i in range(self.num_dof) if i not in wrist_set]
        self._ctrl_inds = torch.tensor(ctrl_list, device=self.device, dtype=torch.long)
        # Expected: 21 indices (legs×12 + torso×1 + shoulders×6 + elbows×2)

        # Re-create policy-side action buffers at 21
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)

        # Rebuild noise_scale_vec for the 74-dim obs
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    # ------------------------------------------------------------------
    # Torque computation: pad 13-dim policy output → 27-dim PD targets
    # Arms padded with 0 → PD drives them to default_dof_pos
    # ------------------------------------------------------------------

    def _compute_torques(self, actions):
        # Direct torque control: actions in [-1, 1] scaled by joint torque limits.
        # Wrists get 0 torque. Bypasses PD entirely — robot must actively balance.
        full = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        full[:, self._ctrl_inds] = actions * self.custom_torque_limits[self._ctrl_inds]
        return torch.clip(full, -self.custom_torque_limits, self.custom_torque_limits)

    # ------------------------------------------------------------------
    # Observations: 50-dim flat
    # ------------------------------------------------------------------

    def _preprocess_obs(self):
        # Controlled joints only (21 DOF: legs+torso+shoulders+elbows)
        dof_pos_ctrl = self.dof_pos[:, self._ctrl_inds]
        dof_vel_ctrl = self.dof_vel[:, self._ctrl_inds]
        dof_default  = self.default_dof_pos[:, self._ctrl_inds]
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,              # 3
            self.projected_gravity,                                     # 3
            (dof_pos_ctrl - dof_default) * self.obs_scales.dof_pos,   # 21
            dof_vel_ctrl * self.obs_scales.dof_vel,                    # 21
            self.actions,                                               # 21
            self.commands[:, :3] * self.commands_scale[:3],            # 3 (vx,vy,yaw)
            self.clock_inputs,                                          # 2
        ), dim=-1)   # total: 74

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise   # side-effect que el padre setea aquí
        ns = cfg.noise.noise_scales
        nl = cfg.noise.noise_level
        n  = self.num_actions  # 21
        vec = torch.zeros(WALK_NUM_OBS, device=self.device)
        vec[0:3]       = ns.ang_vel * nl * self.obs_scales.ang_vel
        vec[3:6]       = ns.gravity * nl
        vec[6:6+n]     = ns.dof_pos * nl * self.obs_scales.dof_pos
        vec[6+n:6+2*n] = ns.dof_vel * nl * self.obs_scales.dof_vel
        # actions, commands, clock: no noise
        return vec

    # ------------------------------------------------------------------
    # Observations: override compute_observations to use our 74-dim noise_scale_vec.
    # The parent H1Robot._init_buffers creates noise_scale_vec via
    # torch.zeros_like(obs_buf[0]) while num_actions=27 (temporary), which can
    # produce the wrong size.  We regenerate it here when needed and skip
    # latency / history stacking (both disabled for the walk task).
    # ------------------------------------------------------------------

    def compute_observations(self, reset_env_ids):
        self._preprocess_obs()
        if self.add_noise:
            if self.noise_scale_vec.shape[0] != self.obs_buf.shape[-1]:
                self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.partial_obs_buf = self.obs_buf

    # ------------------------------------------------------------------
    # Privileged obs = same as actor obs (no separate critic stream)
    # ------------------------------------------------------------------

    def get_privileged_observations(self):
        return self.obs_buf

    # ------------------------------------------------------------------
    # Simplified command resampling: vx/vy/yaw only; gait fixed to
    # walking (phase=0.5, freq=2.0) so the clock is well-defined.
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

        # Fixed gait: bipedal walking at 2 Hz
        self.commands[env_ids, 3] = 2.0   # gait_frequency
        self.commands[env_ids, 4] = 0.5   # gait_phase (0.5 = antiphase = walking)
        self.commands[env_ids, 5] = 0.5   # gait_duration
        self.commands[env_ids, 6] = 0.15  # foot_swing_height

        # ~10% standing envs
        standing = torch.rand(len(env_ids), device=self.device) < 0.1
        self.standing_envs_mask[env_ids] = standing
        self.commands[env_ids[standing], :3] = 0.

        # Zero out very small commands
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
        self.commands[env_ids, 2]  *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)

        self.velocity_level[env_ids] = torch.clip(
            torch.norm(self.commands[env_ids, :2], dim=-1) + 0.5 * torch.abs(self.commands[env_ids, 2]),
            min=1)

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    # ------------------------------------------------------------------
    # Termination: pelvis contact + timeout only.
    # MuJoCo Humanoid terminates only on height (≈ body contact with ground).
    # Removing pitch/roll termination lets the robot stumble and recover,
    # matching the paper's conditions and giving ARS longer, more informative episodes.
    # ------------------------------------------------------------------

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1)
        # gravity_termination_buf kept as attribute so parent rewards don't break
        self.gravity_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.large_ori_buf           = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf

    # ------------------------------------------------------------------
    # No-op curriculum (flat terrain, no penalize curriculum)
    # ------------------------------------------------------------------

    def training_curriculum(self):
        pass

    # ------------------------------------------------------------------
    # Paper-identical rewards (Mania et al. 2018)
    # ------------------------------------------------------------------

    def _reward_forward_vel(self):
        """Raw signed forward velocity — paper uses (x_after - x_before)/dt directly."""
        return self.base_lin_vel[:, 0]

    def _reward_control_cost(self):
        """Quadratic control cost matching MuJoCo Humanoid: 0.001 * ||u||²."""
        return torch.sum(self.actions ** 2, dim=1)
