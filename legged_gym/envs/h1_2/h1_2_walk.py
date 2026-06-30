import torch
from isaacgym.torch_utils import torch_rand_float
from legged_gym.envs.h1_2.h1_2 import H1_2Robot
from legged_gym.envs.h1_2.h1_2_walk_config import H1_2WalkCfg

# Proprioception WITHOUT actions (actions not in Mania 2018).
# ang_vel(3) + gravity(3) + base_lin_vel(3) + root_height(1) + dof_pos(21) + dof_vel(21) = 52
# base_lin_vel: paper has it in qvel[0:3]; also in body_vels[pelvis,0:3] but implicit there.
# root_height:  paper has it in qpos[2];   also in cinert[pelvis,com_z] but implicit there.
WALK_PROP_DIM = 52
WALK_CLOCK    = 2


class H1_2WalkRobot(H1_2Robot):
    """H1-2 with wrists fixed to default_dof_pos.

    Policy controls 21 DOFs (legs×12 + torso×1 + shoulders+elbows×8).

    Observation matches Mania 2018 (ARS paper) as closely as Isaac Gym allows:
      prop:          ang_vel, gravity, dof_pos, dof_vel              (48)
      torques:       qfrc_actuator equivalent, controlled DOFs        (21)
      contact:       cfrc_ext equivalent, net force per body          (B×3)
      body_vels:     cvel equivalent, lin+ang vel per body            (B×6)
      cinert:        mass + COM_world + I_world(6) per body           (B×10)
      clock:         gait phase signals                               (2)

    NOT included vs paper:
      - actions: paper doesn't have them; Markov state is sufficient
      - contact torques: Isaac Gym net_contact_force_tensor is force-only (no torque)

    Total obs = 48 + 21 + B*(3+6+10) + 2 = 71 + B*19
    With B=28 → 603 dims.  Policy M shape: (21, 603) = 12,663 params.
    Paper (B=14, MuJoCo): M = (17, 376) = 6,392 params.
    """

    def __init__(self, cfg: H1_2WalkCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Buffer init
    # ------------------------------------------------------------------

    def _init_buffers(self):
        policy_act = self.num_actions   # 21 (from config)
        self.num_actions = self.num_dof  # temporarily 27 so parent sizes PD buffers correctly
        super()._init_buffers()
        self.num_actions = policy_act   # restore to 21

        # Controlled DOF indices (exclude wrists)
        names_lower = [n.lower() for n in self.dof_names]
        wrist_set = {i for i, n in enumerate(names_lower) if 'wrist' in n}
        ctrl_list = [i for i in range(self.num_dof) if i not in wrist_set]
        self._ctrl_inds = torch.tensor(ctrl_list, device=self.device, dtype=torch.long)

        # Load static body inertia tensors from URDF via Isaac Gym
        self._load_body_inertias()

        # Compute actual obs dim (num_bodies known after _create_envs)
        torque_dim   = self.num_actions          # 21
        contact_dim  = self.num_bodies * 3       # cfrc_ext equivalent
        body_vel_dim = self.num_bodies * 6       # cvel equivalent
        cinert_dim   = self.num_bodies * 10      # mass(1)+com(3)+I_world(6)
        self._walk_obs_dim = (WALK_PROP_DIM + torque_dim + contact_dim
                              + body_vel_dim + cinert_dim + WALK_CLOCK)

        # Override obs buffer with correct size
        self.num_obs = self._walk_obs_dim
        self.obs_buf = torch.zeros(self.num_envs, self._walk_obs_dim,
                                   dtype=torch.float, device=self.device)

        # Policy-side action buffers at 21
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        print(f"[H1_2WalkRobot] num_bodies={self.num_bodies}  obs_dim={self._walk_obs_dim}  "
              f"policy_params={self.num_actions * self._walk_obs_dim}")

    def _load_body_inertias(self):
        """Load static body mass and local inertia tensors (called once at init)."""
        props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        masses  = torch.zeros(self.num_bodies, dtype=torch.float)
        I_local = torch.zeros(self.num_bodies, 3, 3, dtype=torch.float)
        for i, p in enumerate(props):
            masses[i] = p.mass
            try:
                I_local[i, 0] = torch.tensor([p.inertia.x.x, p.inertia.x.y, p.inertia.x.z])
                I_local[i, 1] = torch.tensor([p.inertia.y.x, p.inertia.y.y, p.inertia.y.z])
                I_local[i, 2] = torch.tensor([p.inertia.z.x, p.inertia.z.y, p.inertia.z.z])
            except (AttributeError, TypeError):
                # Fallback: diagonal approx (sphere r≈0.1m → I≈m×0.004)
                I_local[i] = torch.eye(3) * float(p.mass) * 0.004
        self._body_masses = masses.to(self.device)
        self._I_local     = I_local.to(self.device)

        # Pre-normalization scale for cinert (10 values per body):
        #   mass  / 10    → pelvis(15kg)→1.5,  wrist(0.1kg)→0.01
        #   com   / 1     → positions already in meters (0–2m range)
        #   I6    / 0.1   → pelvis I(~0.5)→5,  wrist I(~0.001)→0.01
        # Brings all cinert components into roughly the same order of magnitude
        # BEFORE obs_rms sees them, avoiding cold-start normalization failure.
        pattern = torch.tensor([10.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self._cinert_scale = pattern.repeat(self.num_bodies).to(self.device)  # (B*10,)

    # ------------------------------------------------------------------
    # Torque computation: direct torque control (not PD)
    # ------------------------------------------------------------------

    def _compute_torques(self, actions):
        full = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        full[:, self._ctrl_inds] = actions * self.custom_torque_limits[self._ctrl_inds]
        return torch.clip(full, -self.custom_torque_limits, self.custom_torque_limits)

    # ------------------------------------------------------------------
    # cinert: world-frame inertia tensors per body (Mania 2018)
    # ------------------------------------------------------------------

    def _compute_cinert(self):
        """(mass, COM_world, I_world_upper_tri) × num_bodies → (N, B*10)."""
        # COM world positions (in Isaac Gym, rigid_body origin is the COM)
        com_world = self.rigid_body_states[:, :, :3]       # (N, B, 3)

        # Quaternion → rotation matrix (Isaac Gym uses xyzw format)
        q = self.rigid_body_states[:, :, 3:7]
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = torch.stack([
            1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
            2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
            2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
        ], dim=-1).reshape(self.num_envs, self.num_bodies, 3, 3)

        # I_world = R @ I_local @ R^T
        I_loc   = self._I_local.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        I_world = R @ I_loc @ R.transpose(-1, -2)          # (N, B, 3, 3)

        # 6 unique components (upper triangle of symmetric matrix)
        I6 = torch.stack([
            I_world[..., 0, 0], I_world[..., 1, 1], I_world[..., 2, 2],
            I_world[..., 0, 1], I_world[..., 0, 2], I_world[..., 1, 2],
        ], dim=-1)                                         # (N, B, 6)

        mass = self._body_masses.view(1, -1, 1).expand(self.num_envs, -1, 1)

        cinert = torch.cat([mass, com_world, I6], dim=-1).reshape(self.num_envs, -1)
        return cinert / self._cinert_scale.unsqueeze(0)

    # ------------------------------------------------------------------
    # Observations: paper-matched structure
    # ------------------------------------------------------------------

    def _preprocess_obs(self):
        dof_pos_ctrl = self.dof_pos[:, self._ctrl_inds]
        dof_vel_ctrl = self.dof_vel[:, self._ctrl_inds]
        dof_default  = self.default_dof_pos[:, self._ctrl_inds]

        body_vels   = self.rigid_body_states[:, :, 7:13].reshape(self.num_envs, -1)
        contact     = self.contact_forces.reshape(self.num_envs, -1)
        torques_obs = self.torques[:, self._ctrl_inds]

        root_height = self.root_states[:, 2:3]  # z position of root body (paper: qpos[2])

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,              # 3
            self.projected_gravity,                                     # 3
            self.base_lin_vel * self.obs_scales.lin_vel,              # 3  (paper: qvel[0:3])
            root_height,                                               # 1  (paper: qpos[2])
            (dof_pos_ctrl - dof_default) * self.obs_scales.dof_pos,   # 21
            dof_vel_ctrl * self.obs_scales.dof_vel,                    # 21
            torques_obs * self.obs_scales.dof_vel,                     # 21
            contact * self.obs_scales.contact_force,                   # B×3
            body_vels * self.obs_scales.lin_vel,                       # B×6
            self._compute_cinert(),                                     # B×10
            self.clock_inputs,                                          # 2
        ), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise
        ns = cfg.noise.noise_scales
        nl = cfg.noise.noise_level
        n  = self.num_actions  # 21

        nb    = getattr(self, 'num_bodies', 0)
        total = WALK_PROP_DIM + n + nb*3 + nb*6 + nb*10 + WALK_CLOCK

        vec = torch.zeros(total, device=self.device)
        vec[0:3]   = ns.ang_vel * nl * self.obs_scales.ang_vel
        vec[3:6]   = ns.gravity * nl
        vec[6:9]   = ns.lin_vel * nl * self.obs_scales.lin_vel   # base_lin_vel
        vec[9:10]  = ns.height_measurements * nl                  # root_height
        vec[10:31] = ns.dof_pos * nl * self.obs_scales.dof_pos
        vec[31:52] = ns.dof_vel * nl * self.obs_scales.dof_vel
        # torques, contact, body_vel, cinert, clock: no noise
        return vec

    def compute_observations(self, reset_env_ids):
        self._preprocess_obs()
        if self.add_noise:
            if self.noise_scale_vec.shape[0] != self.obs_buf.shape[-1]:
                self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.partial_obs_buf = self.obs_buf

    def get_privileged_observations(self):
        return self.obs_buf

    # ------------------------------------------------------------------
    # Command resampling: vx sampled [0.5, 2.0] but NOT observed by policy.
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            0.5, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0
        self.commands[env_ids, 3] = 2.0   # gait_frequency
        self.commands[env_ids, 4] = 0.5   # gait_phase (walking)
        self.commands[env_ids, 5] = 0.5
        self.commands[env_ids, 6] = 0.15

        standing = torch.rand(len(env_ids), device=self.device) < 0.1
        self.standing_envs_mask[env_ids] = standing
        self.commands[env_ids[standing], :3] = 0.

        self.velocity_level[env_ids] = torch.clip(
            torch.norm(self.commands[env_ids, :2], dim=-1)
            + 0.5 * torch.abs(self.commands[env_ids, 2]), min=1)

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    # ------------------------------------------------------------------
    # Termination: pelvis contact + timeout only
    # ------------------------------------------------------------------

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1)
        self.gravity_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.large_ori_buf           = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def training_curriculum(self):
        pass

    # ------------------------------------------------------------------
    # Rewards (Mania 2018)
    # ------------------------------------------------------------------

    def _reward_forward_vel(self):
        return self.base_lin_vel[:, 0]

    def _reward_control_cost(self):
        return torch.sum(self.actions ** 2, dim=1)

    def _reward_contact_cost(self):
        # Paper: -0.5e-3 * sum(clip(cfrc_ext, -1, 1)²) per step
        # Clip at 100 N (≈ 1/3 body weight per body), then normalize to [-1,1] range
        forces = self.contact_forces.reshape(self.num_envs, -1)  # (N, B*3)
        return torch.sum(torch.clamp(forces / 100.0, -1.0, 1.0) ** 2, dim=-1)
