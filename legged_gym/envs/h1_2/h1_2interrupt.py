import torch
from legged_gym.envs.h1.h1interrupt import H1InterruptRobot
from legged_gym.envs.h1_2.h1_2interrupt_config import H1_2InterruptCfg


class H1_2InterruptRobot(H1InterruptRobot):
    """H1-2 with physical arm perturbations — the HugWBC paper contribution.

    Inherits all interrupt/curriculum logic from H1InterruptRobot.
    Overrides every method that hardcodes H1-specific DOF indices:

      H1  : legs(10) + torso(1) = arm_start at DOF 11 | disturb_dim = 8
      H1-2: legs(12) + torso(1) = arm_start at DOF 13 | disturb_dim = 14

    arm_start_idx is computed dynamically in _create_envs, so if the URDF
    tree order ever changes, no constant needs patching.
    """

    def __init__(self, cfg: H1_2InterruptCfg, sim_params, physics_engine,
                 sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Environment creation — adds H1-2-specific index groups
    # ------------------------------------------------------------------

    def _create_envs(self):
        super()._create_envs()
        dof_names_lower = [n.lower() for n in self.dof_names]

        self.wrist_inds = [i for i, n in enumerate(dof_names_lower) if 'wrist'  in n]
        self.knee_inds  = [i for i, n in enumerate(dof_names_lower) if 'knee'   in n]

        # Precompute as tensor for fast indexing in calculate_action
        self.wrist_t = torch.tensor(self.wrist_inds, dtype=torch.long, device=self.device)

        # Option B: DOF indices that map into the 21-dim action space (all except wrists)
        self.non_wrist_idx = torch.tensor(
            [i for i in range(self.num_dof) if i not in self.wrist_inds],
            dtype=torch.long, device=self.device)

        # First DOF that belongs to the arm (shoulder / elbow / wrist)
        arm_dof_indices = [i for i, n in enumerate(dof_names_lower)
                           if 'shoulder' in n or 'elbow' in n or 'wrist' in n]
        self.arm_start_idx = min(arm_dof_indices) if arm_dof_indices else 13

        # Per-joint clip scale: weaker joints get proportionally smaller disturbance radius
        self.disturb_rad_scale = torch.tensor(
            self.cfg.disturb.disturb_rad_scale, dtype=torch.float, device=self.device
        )  # shape (disturb_dim,)

        # Precompute relative arm-group indices (relative to arm_start_idx)
        s = self.arm_start_idx
        self._sh_r = torch.tensor([i - s for i in self.shoulder_inds], device=self.device)
        self._el_r = torch.tensor([i - s for i in self.elbow_inds],    device=self.device)
        self._wr_r = torch.tensor([i - s for i in self.wrist_inds],    device=self.device)

        # Per-episode accumulators — all reset to 0 at episode end
        self._diag = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            # tracking split
            'track_disturb', 'track_clean', 'steps_disturb', 'steps_clean',
            # arm velocity by group (episode mean, rad/s)
            'vel_sh', 'vel_el', 'vel_wr',
            # arm deviation from default in clean envs (episode mean, rad²)
            'dev_sh_clean', 'dev_el_clean', 'dev_wr_clean', 'steps_clean_dev',
            # disturb magnitude vs policy action in disturb envs (episode mean, rad)
            'dmag_sh', 'dmag_el', 'dmag_wr', 'steps_disturb_mag',
        ]}

        # Global step-level accumulators for tracking split — drained each reset_idx
        # call so tracking_disturb/clean are always valid (never NaN).
        self._gtrack = {k: torch.zeros(1, device=self.device) for k in
                        ['d_sum', 'd_cnt', 'c_sum', 'c_cnt']}

    # ------------------------------------------------------------------
    # Option B: num_actions=21 (no wrists in NN), num_dof=27
    # ------------------------------------------------------------------

    def _init_buffers(self):
        # p_gains / d_gains / torques must cover all 27 DOFs, even though the NN
        # only outputs 21 actions.  We temporarily pretend num_actions==num_dof so
        # the base _init_buffers creates correctly-sized tensors, then restore and
        # resize the action buffers to the real 21.
        saved = self.num_actions
        self.num_actions = self.num_dof   # temporary: makes p_gains/d_gains/torques 27-dim
        super()._init_buffers()
        self.num_actions = saved          # restore to 21

        # Resize action-history buffers (super() created them at 27)
        self.actions         = torch.zeros(self.num_envs, self.num_actions,
                                           dtype=torch.float, device=self.device)
        self.last_actions    = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)

        # 27-dim pre-disturb expansion stored each step for diagnostics
        self._pre_disturb_actions = torch.zeros(self.num_envs, self.num_dof,
                                                dtype=torch.float, device=self.device)

        # Rebuild noise_scale_vec with correct dof-vs-action split
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _get_noise_scale_vec(self, cfg):
        """Use num_dof (27) for dof_pos/vel sections; num_actions (21) for last-actions."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        ns = cfg.noise.noise_scales
        nl = cfg.noise.noise_level
        noise_vec[:3]                              = ns.ang_vel * nl * self.obs_scales.ang_vel
        noise_vec[3:6]                             = ns.gravity * nl
        noise_vec[6:6+self.num_dof]                = ns.dof_pos * nl * self.obs_scales.dof_pos
        noise_vec[6+self.num_dof:6+2*self.num_dof] = ns.dof_vel * nl * self.obs_scales.dof_vel
        # [6+2*num_dof : 6+2*num_dof+num_actions] = 0 (no noise on last actions)
        return noise_vec

    def calculate_action(self, actions_21):
        """Option B: NN outputs 21-dim actions (no wrists).

        Expands to 27-dim before PD/disturb computation.  Wrists stay at
        delta_q=0 → PD holds default_dof_pos in clean envs.  In disturb envs
        the parent's mechanism overwrites the last 14 arm DOFs (13-26), so
        wrists DO receive external targets.

        self.actions is kept as 21-dim for obs, action-rate rewards, etc.
        The 27-dim result is returned for _compute_torques.
        """
        clip_a = self.cfg.normalization.clip_actions
        # Store 21-dim for obs / rewards (before super() overwrites self.actions)
        self.actions = torch.clip(actions_21, -clip_a, clip_a)

        # Expand to 27-dim: non-wrist joints from policy, wrists stay 0 (→ default_pos)
        full_27 = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        full_27[:, self.non_wrist_idx] = self.actions

        # Save pre-disturb expansion for diagnostics (before disturb overwrites arm joints)
        self._pre_disturb_actions.copy_(full_27)

        # H1InterruptRobot.calculate_action applies disturb on last 14 DOFs and
        # calls H1Robot.calculate_action which overwrites self.actions → restore after
        result_27 = super().calculate_action(full_27)
        self.actions = torch.clip(actions_21, -clip_a, clip_a)  # restore 21-dim

        return result_27  # 27-dim used by _compute_torques via step()

    def initial_disturb(self, cfg):
        super().initial_disturb(cfg)
        # super() created executed_actions with num_actions=21; torques need 27-dim
        self.executed_actions = torch.zeros(self.num_envs, self.num_dof,
                                            dtype=torch.float, device=self.device)

    def update_disturb_curriculum_grid(self, env_ids, noise_env_ids):
        if len(env_ids) == 0:
            return

        if len(noise_env_ids) > 0:
            timesteps = int(self.cfg.commands.resampling_time / self.dt)
            ep_len = min(self.max_episode_length, timesteps)

            curr_is_pass = torch.ones(len(noise_env_ids), dtype=bool, device=self.device)
            curr_is_down = torch.zeros(len(noise_env_ids), dtype=bool, device=self.device)

            for key, value in self.curriculum_thresholds['disturb'].items():
                all_rew = self.command_sums[key][noise_env_ids] / ep_len
                success_threshold = value * self.reward_scales[key]
                if key in self.curriculum_reward_list:
                    success_threshold *= self.curriculum_scale
                curr_is_pass *= (all_rew > success_threshold)
                curr_is_down += (all_rew < success_threshold / 2)

            step = 0.03
            self.disturb_rad_curriculum[noise_env_ids] = torch.where(
                curr_is_down,
                (self.disturb_rad_curriculum[noise_env_ids] - step).clip(min=0),
                torch.where(
                    curr_is_pass,
                    (self.disturb_rad_curriculum[noise_env_ids] + step).clip(
                        max=self.cfg.disturb.max_curriculum),
                    self.disturb_rad_curriculum[noise_env_ids]
                )
            )

        # ---- Resample (identical to parent) ----
        self.disturb_masks[env_ids] = (torch.rand(len(env_ids)) <= 0.5).to(self.device)
        is_noise = torch.rand(len(env_ids)) <= self.disturb_noise_ratio
        self.disturb_isnoise[env_ids] = is_noise.to(self.device)
        self.disturb_actions[env_ids] = (
            self.dof_pos[env_ids, -self.disturb_dim:]
            - self.default_dof_pos[:, -self.disturb_dim:])
        if self.disturb_replace_action:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids]
        else:
            self.interrupt_mask[env_ids] = (
                self.disturb_masks[env_ids] * (~self.disturb_isnoise[env_ids]))

    def training_curriculum(self):
        from legged_gym.envs.base.base_task import BaseTask
        BaseTask.training_curriculum(self)  # increments learning_iter only
        if self.cfg.rewards.penalize_curriculum and (self.learning_iter % 100 == 0):
            self.curriculum_scale = pow(self.curriculum_scale, self.cfg.rewards.penalize_curriculum_sigma)

    # ------------------------------------------------------------------
    # Diagnostic logging
    # ------------------------------------------------------------------

    def post_physics_step(self):
        super().post_physics_step()
        self._accumulate_diagnostics()

    def _accumulate_diagnostics(self):
        """Per-step accumulation — called every control step."""
        s  = self.arm_start_idx
        d  = self.disturb_masks.float()
        nd = 1.0 - d

        # ── tracking split ───────────────────────────────────────────────
        track = self._reward_tracking_lin_vel()
        self._diag['track_disturb'] += track * d
        self._diag['track_clean']   += track * nd
        self._diag['steps_disturb'] += d
        self._diag['steps_clean']   += nd

        # Global accumulators: span multiple reset batches so tracking_disturb/clean
        # are always valid regardless of which envs are in any given reset batch.
        self._gtrack['d_sum'] += (track * d).sum()
        self._gtrack['d_cnt'] += d.sum()
        self._gtrack['c_sum'] += (track * nd).sum()
        self._gtrack['c_cnt'] += nd.sum()

        # ── arm velocity by group (rad/s, episode mean) ──────────────────
        arm_v = torch.abs(self.dof_vel[:, s:])
        self._diag['vel_sh'] += arm_v[:, self._sh_r].mean(dim=1)
        self._diag['vel_el'] += arm_v[:, self._el_r].mean(dim=1)
        self._diag['vel_wr'] += arm_v[:, self._wr_r].mean(dim=1)

        # ── arm deviation from default in clean envs (rad², episode mean) ─
        dev_sq = torch.square(self.dof_pos[:, s:] - self.default_dof_pos[:, s:])
        self._diag['dev_sh_clean']    += dev_sq[:, self._sh_r].mean(dim=1) * nd
        self._diag['dev_el_clean']    += dev_sq[:, self._el_r].mean(dim=1) * nd
        self._diag['dev_wr_clean']    += dev_sq[:, self._wr_r].mean(dim=1) * nd
        self._diag['steps_clean_dev'] += nd

        # ── disturb target vs policy action magnitude (rad, episode mean) ─
        if self.disturb_masks.any():
            # Use pre-disturb 27-dim expansion: indices [s:] = 14 arm DOFs, matches disturb_actions shape
            dmag = (torch.abs(self.disturb_actions - self._pre_disturb_actions[:, s:])
                    * self.cfg.control.action_scale)
            self._diag['dmag_sh']           += dmag[:, self._sh_r].mean(dim=1) * d
            self._diag['dmag_el']           += dmag[:, self._el_r].mean(dim=1) * d
            self._diag['dmag_wr']           += dmag[:, self._wr_r].mean(dim=1) * d
            self._diag['steps_disturb_mag'] += d

    def reset_idx(self, env_ids):
        # Capture episode lengths BEFORE super() zeroes episode_length_buf
        ep_lens = self.episode_length_buf[env_ids].float()

        super().reset_idx(env_ids)
        ex = self.extras['episode']

        # helper: safe per-episode mean; returns None when no valid envs in batch
        def ep_mean(sum_key, step_key):
            n = self._diag[step_key][env_ids]
            has = n > 0
            if not has.any():
                return None
            return (self._diag[sum_key][env_ids][has] / n[has]).mean().item()

        n_steps_tot = (self._diag['steps_disturb'][env_ids]
                       + self._diag['steps_clean'][env_ids]).clamp(min=1)

        # ── Group 1: arm velocity by group (episode mean, rad/s) ─────────
        ex['arm_vel_shoulder'] = (self._diag['vel_sh'][env_ids] / n_steps_tot).mean()
        ex['arm_vel_elbow']    = (self._diag['vel_el'][env_ids] / n_steps_tot).mean()
        ex['arm_vel_wrist']    = (self._diag['vel_wr'][env_ids] / n_steps_tot).mean()

        # ── Group 2a: tracking split — global accumulators (never NaN) ───
        # Drain and reset after each call: each ep_info gets the tracking quality
        # for all steps since the previous reset_idx.  The runner averages these
        # across all ep_infos in the iteration → weighted step-level estimate.
        if self._gtrack['d_cnt'] > 0:
            ex['tracking_disturb'] = (self._gtrack['d_sum'] / self._gtrack['d_cnt']).item()
            self._gtrack['d_sum'].zero_()
            self._gtrack['d_cnt'].zero_()
        if self._gtrack['c_cnt'] > 0:
            ex['tracking_clean'] = (self._gtrack['c_sum'] / self._gtrack['c_cnt']).item()
            self._gtrack['c_sum'].zero_()
            self._gtrack['c_cnt'].zero_()
        ex['disturb_step_frac'] = (
            self._diag['steps_disturb'][env_ids] / n_steps_tot).mean()

        # ── Group 2b: episode-length split (disturb vs clean envs) ───────
        # was_disturb: envs that had active disturb during this episode
        was_disturb = self._diag['steps_disturb'][env_ids] > 0
        if was_disturb.any():
            ex['ep_len_disturb'] = ep_lens[was_disturb].mean().item()
        if (~was_disturb).any():
            ex['ep_len_clean'] = ep_lens[~was_disturb].mean().item()

        # ── Group 3: disturb magnitude vs policy (episode mean, rad) ──────
        ex['disturb_active_ratio'] = self.disturb_masks.float().mean()
        v = ep_mean('dmag_sh', 'steps_disturb_mag')
        if v is not None: ex['disturb_mag_shoulder'] = v
        v = ep_mean('dmag_el', 'steps_disturb_mag')
        if v is not None: ex['disturb_mag_elbow'] = v
        v = ep_mean('dmag_wr', 'steps_disturb_mag')
        if v is not None: ex['disturb_mag_wrist'] = v

        # ── Group 4: arm deviation from default in clean envs (rad²) ──────
        v = ep_mean('dev_sh_clean', 'steps_clean_dev')
        if v is not None: ex['dev_shoulder_clean'] = v
        v = ep_mean('dev_el_clean', 'steps_clean_dev')
        if v is not None: ex['dev_elbow_clean'] = v
        v = ep_mean('dev_wr_clean', 'steps_clean_dev')
        if v is not None: ex['dev_wrist_clean'] = v

        # Reset all per-episode accumulators for envs whose episode just ended
        for k in self._diag:
            self._diag[k][env_ids] = 0.0

    # ------------------------------------------------------------------
    # Disturb resampling — adapted for 14 arm joints (H1 had 8)
    # ------------------------------------------------------------------

    def curriculum_disturb_clipping_mean_rad(self, actions):
        noise_mean = (
            self.disturb_rad_curriculum.unsqueeze(-1)
            * (self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:])
            + (1 - self.disturb_rad_curriculum.unsqueeze(-1))
            * (actions[:, -self.disturb_dim:] * self.cfg.control.action_scale)
        )
        # Per-joint clip radius: global rad × curriculum × per-joint scale
        rad = (self.disturb_rad
               * self.disturb_rad_curriculum.unsqueeze(-1)
               * self.disturb_rad_scale.unsqueeze(0))
        return torch.clamp(
            self.disturb_actions,
            (-rad + noise_mean) / self.cfg.control.action_scale,
            ( rad + noise_mean) / self.cfg.control.action_scale,
        )

    def Gaussian_disturb_resample(self):
        mean = torch.zeros(self.disturb_dim, device=self.device)
        std  = torch.ones(self.disturb_dim,  device=self.device) * self.disturb_scale
        return torch.clamp(
            torch.normal(mean, std)
                + self.dof_pos[:, -self.disturb_dim:]
                - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 0].view(1, -1).repeat(self.num_envs, 1)
                - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 1].view(1, -1).repeat(self.num_envs, 1)
                - self.default_dof_pos[:, -self.disturb_dim:]
        )

    def Uniform_disturb_resample(self):
        """Sample uniform pose targets within the H1-2 arm joint ranges.

        Disturb-array layout (14 joints):
          Indices 0-6 : left  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
          Indices 7-13: right shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
        """
        scale = self.disturb_uniform_scale
        targets = (scale * self.disturb_noise_scale
                   * torch.rand((self.num_envs, self.disturb_dim), device=self.device)
                   + self.disturb_noise_lowerbound
                   + self.disturb_noise_scale * (1 - scale) / 2)

        # Left shoulder_roll at disturb index 1.  Range: -0.38~3.40.
        # When roll < 0.5 rad (arm near body), zero distal left-arm joints
        # to avoid physically impossible poses.
        left_folded = targets[:, 1] < 0.5
        targets[left_folded, 2:7] = 0   # l_sh_yaw, l_elbow, l_wr_roll/pitch/yaw

        # Right shoulder_roll at disturb index 8.  Range: -3.40~0.38.
        # Mirror: roll > -0.5 means arm near body on the right side.
        right_folded = targets[:, 8] > -0.5
        targets[right_folded, 9:14] = 0  # r_sh_yaw, r_elbow, r_wr_roll/pitch/yaw

        return torch.clamp(
            targets - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 0].view(1, -1).repeat(self.num_envs, 1)
                - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 1].view(1, -1).repeat(self.num_envs, 1)
                - self.default_dof_pos[:, -self.disturb_dim:]
        )

    # ------------------------------------------------------------------
    # Reward overrides — replace all hardcoded index 11 (H1) with
    # self.arm_start_idx (computed dynamically for H1-2)
    # ------------------------------------------------------------------

    def _reward_action_rate_upper(self):
        s = self.arm_start_idx
        diff_1 = torch.sum(torch.square(
            self.actions[:, s:] - self.last_actions[:, s:]), dim=1)
        diff_2 = torch.sum(torch.square(
            self.actions[:, s:] - 2 * self.last_actions[:, s:]
            + self.last_last_actions[:, s:]), dim=1)
        return (diff_1 + diff_2) * (~self.interrupt_mask)

    def _reward_action_rate_wrist(self):
        """Stronger action-rate penalty exclusively for wrist DOFs.
        Wrists have no locomotion role — the policy should learn to keep them
        stable by default, not oscillate randomly. At deploy without external
        reference, this prevents uncontrolled wrist motion.
        Zero'd when disturb controls wrists externally (same as action_rate_upper).
        """
        wi = torch.tensor(self.wrist_inds, dtype=torch.long, device=self.device)
        diff_1 = torch.sum(torch.square(
            self.actions[:, wi] - self.last_actions[:, wi]), dim=1)
        diff_2 = torch.sum(torch.square(
            self.actions[:, wi] - 2 * self.last_actions[:, wi]
            + self.last_last_actions[:, wi]), dim=1)
        return (diff_1 + diff_2) * (~self.interrupt_mask)

    def _reward_action_rate_lower(self):
        s = self.arm_start_idx
        diff_1 = torch.sum(torch.square(
            self.actions[:, :s] - self.last_actions[:, :s]), dim=1)
        diff_2 = torch.sum(torch.square(
            self.actions[:, :s] - 2 * self.last_actions[:, :s]
            + self.last_last_actions[:, :s]), dim=1)
        return diff_1 + diff_2

    def _reward_dof_pos_limits(self):
        out_of_limits  = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        out_of_limits[:, self.arm_start_idx:] = 0  # arm limits not penalised during disturb
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_acc(self):
        reward = torch.square((self.last_dof_vel - self.dof_vel) / self.dt)
        reward[:, self.arm_start_idx:] = 0  # arm acc not penalised during disturb
        return torch.sum(reward, dim=1)

    def _reward_dof_vel_limits(self):
        s = self.arm_start_idx
        lim_scale = 1.0   # wrists excluded → 8 effective DOFs = same as H1 → same limits (10-20 rad/s)
        dof_vel_limits = torch.clip(
            10 * self.velocity_level.unsqueeze(-1).repeat(1, self.num_dof),
            min=10 * lim_scale, max=20 * lim_scale)
        # Wrists oscillate at ~30 rad/s in Phase 1 due to random target-chasing
        # (untrained policy changes targets every 20ms, fast PD follows).
        # They dominate the sum and block terrain graduation despite shoulder/elbow
        # being well-controlled (<3 rad/s). Wrist velocity is handled separately
        # via _reward_wrist_vel_cost; here we count only shoulder+elbow DOFs.
        arm_vel = torch.abs(self.dof_vel[:, s:])
        arm_vel[:, self._wr_r] = 0.0
        error = torch.sum(
            (arm_vel - dof_vel_limits[:, s:]).clip(min=0., max=15.),
            dim=1)
        return 1 - torch.exp(-1 * error)

    def _reward_joint_power_distribution(self):
        knee_inds = torch.tensor(self.knee_inds, dtype=torch.long, device=self.device)
        penalize = (torch.abs(self.torques) * torch.abs(self.dof_vel))[:, knee_inds].var(dim=-1)
        return (torch.square(penalize) * 1.e-8).clip(max=1)

    def _reward_standing_joint_deviation(self):
        arm_inds = torch.tensor(
            self.shoulder_inds + self.elbow_inds + self.wrist_inds,
            dtype=torch.long, device=self.device)
        reward = torch.square(self.dof_pos - self.default_dof_pos)[:, arm_inds]
        reward[~self.standing_envs_mask] = 0
        # Suppress penalty when arms are externally perturbed (same as H1InterruptRobot)
        return torch.sum(reward, dim=-1) * (~self.interrupt_mask)

    def _reward_wrist_deviation(self):
        """Penalize wrist joints deviating from default during walking.
        H1 has no wrists; H1-2 has 6 wrist DOFs that oscillate freely without this.
        Zero'd when arms are under external perturbation (same pattern as shoulder_deviation).
        """
        wrist_t = torch.tensor(self.wrist_inds, dtype=torch.long, device=self.device)
        dev_sq = torch.sum(
            torch.square(self.dof_pos[:, wrist_t] - self.default_dof_pos[:, wrist_t]),
            dim=-1)
        return dev_sq * (~self.interrupt_mask)

    def _reward_wrist_vel_cost(self):
        """Penalize wrist joint VELOCITY directly (not position).
        Wrists oscillate at ~30 rad/s due to random target-chasing (untrained policy
        changes wrist targets every 20ms; fast PD follows → high velocity, low mean
        displacement). Position penalty doesn't help; velocity penalty does.
        Zero'd when arms are under external perturbation.
        """
        wrist_t = torch.tensor(self.wrist_inds, dtype=torch.long, device=self.device)
        vel_sq = torch.mean(
            torch.square(self.dof_vel[:, wrist_t]),
            dim=-1)
        return vel_sq * (~self.interrupt_mask)

