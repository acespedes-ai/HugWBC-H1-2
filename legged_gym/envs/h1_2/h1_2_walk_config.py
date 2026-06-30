from legged_gym.envs.h1_2.h1_2_config import H1_2Cfg, H1_2CfgPPO

# H1-2 with arms (no wrists): legs×12 + torso×1 + shoulders+elbows×8 = 21 DOF controlled.
# Wrists (6 DOF) fixed to default_dof_pos — no locomotion contribution.
# Matches MuJoCo Humanoid (21 DOF) where arms are free for balance.
WALK_PROP_DIM = 69   # 3(ang_vel) + 3(gravity) + 21(dof_pos) + 21(dof_vel) + 21(actions)
WALK_CMD_DIM  = 3    # vx, vy, yaw exposed in obs
WALK_CLOCK    = 2
WALK_NUM_OBS  = WALK_PROP_DIM + WALK_CMD_DIM + WALK_CLOCK   # 74
WALK_NUM_ACT  = 21   # legs (12) + torso (1) + shoulders+elbows (4×2)


class H1_2WalkCfg(H1_2Cfg):

    class init_state(H1_2Cfg.init_state):
        # Kinematic height with default pose (hip_pitch=-0.1, knee=0.3, ankle=-0.2): ~0.855m.
        # MuJoCo Humanoid (ARS paper) starts with feet on ground, no drop.
        # Setting pos[2]=0.86 ≈ kinematic height eliminates the 0.115m drop/bounce
        # that was capping ep_len at ~53 regardless of policy quality.
        pos = [0.0, 0.0, 1.05]  # official Unitree value; kinematic height with -0.16/0.36/-0.20 ≈ 1.0m

    class env(H1_2Cfg.env):
        num_envs           = 4096
        num_observations   = WALK_NUM_OBS
        num_partial_obs    = WALK_NUM_OBS
        num_privileged_obs = None
        num_actions        = WALK_NUM_ACT
        has_privileged_info    = False
        stack_history_obs      = False
        include_history_steps  = 1
        observe_body_height    = False
        observe_body_pitch     = False
        observe_waist_roll     = False
        observe_gait_commands  = True   # keeps clock update running

    class terrain(H1_2Cfg.terrain):
        mesh_type        = 'plane'
        curriculum       = False
        measure_heights  = False
        measure_foot_scan = False

    class commands(H1_2Cfg.commands):
        curriculum      = False
        resampling_time = 10.0
        num_commands    = 7   # internal: vx, vy, yaw, freq, phase, dur, swing_h
        class ranges(H1_2Cfg.commands.ranges):
            lin_vel_x   = [-1.0, 2.0]
            lin_vel_y   = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]

    class domain_rand(H1_2Cfg.domain_rand):
        push_robots                = False  # pushes add noise to ARS gradient; paper didn't use them
        push_interval_s            = 15
        max_push_vel_xy            = 0.5
        randomize_gains            = False
        randomize_control_latency  = False  # obs es 50-dim, buffer de latencia espera 60 (6+27*2)

    class rewards(H1_2Cfg.rewards):
        penalize_curriculum    = False
        reward_curriculum_list = []

        class scales(H1_2Cfg.rewards.scales):
            # === Paper-identical reward (Mania et al. 2018) ===
            # R = forward_vel + 5*alive - 0.001*||u||²
            forward_vel       =  1.0   # raw signed vx (m/s)
            alive             =  5.0   # essential with torque control: robot falls with zero actions
            control_cost      = -0.001 # paper ctrl_cost coefficient (fn returns sum of a²)
            termination       =  0.0   # no extra spike; episode ends via check_termination
            # === everything else disabled ===
            tracking_lin_vel  =  0
            tracking_ang_vel  =  0
            no_fly            =  0
            torques           =  0
            dof_acc           =  0
            ang_vel_xy        =  0
            lin_vel_z         =  0
            feet_contact_forces = 0
            feet_slip         =  0
            feet_stumble      =  0
            action_rate       =  0
            base_height       =  0
            stand_still       =  0
            orientation_control = 0
            waist_control     =  0
            standing_air      =  0
            standing          =  0
            standing_joint_deviation = 0
            hip_deviation     =  0
            shoulder_deviation =  0
            joint_power_distribution = 0
            hopping_symmetry  =  0
            dof_pos_limits    =  0
            dof_vel_limits    =  0
            collision         =  0
            tracking_contacts_shaped_force = 0
            tracking_contacts_shaped_vel   = 0
            feet_clearance_cmd_linear      = 0
            feet_clearance_cmd_polynomial  = 0

    class curriculum_thresholds:
        class commands:
            tracking_lin_vel = 0.8


class H1_2WalkCfgPPO(H1_2CfgPPO):
    class runner(H1_2CfgPPO.runner):
        experiment_name   = 'h1_2_walk_ppo'
        max_iterations    = 3000
        save_interval     = 200
        num_steps_per_env = 24

    class policy(H1_2CfgPPO.policy):
        model_name = 'SimpleMlpModel'
        class NetModel:
            class SimpleMlpModel:
                hidden_dims = [512, 256, 128]

        critic_hidden_dims = [512, 256, 128]
        critic_obs_dim     = WALK_NUM_OBS


class H1_2WalkCfgARS:
    """Standalone config for ARS runner (not PPO-compatible).

    Paper reference (Mania et al. 2018, Table 1 — MuJoCo Humanoid-v1):
        N=230, b=115, alpha=0.015, nu=0.025
    We use N=230, b=115 to match.  alpha/nu tuned for our task.
    num_envs = 2*N = 460.
    """
    class runner:
        experiment_name = 'h1_2_walk_ars'
        max_iterations  = 2000   # paper converges ~1k; 2k gives margin without wasting compute
        save_interval   = 50
        num_pairs       = 230   # N: matches paper's Humanoid setting
        elite_pairs     = 115   # b: top-50% pairs (paper: b=N/2)
        # Paper uses alpha=0.015 for MuJoCo where sigma_r~100 (rewards not dt-scaled).
        # Isaac Gym rewards are dt-scaled → sigma_r~0.7-1.4 (100× smaller).
        # effective_lr = alpha / (sigma_r * nu).  To match paper's ~0.006:
        #   alpha = 0.006 × 1.0 × 0.025 ≈ 0.00015  → use 0.001 for some margin.
        step_size       = 0.001 # alpha: scaled for Isaac Gym dt-reward range
        noise_std       = 0.025 # nu: paper value for Humanoid
        rollout_steps   = 1000  # max steps per rollout (safety cap)
        clip_obs        = 5.0   # obs clipping after normalization
        clip_actions    = 1.0
        seed            = 1
        sigma_r_min     = 1.0   # floor on reward std: prevents update blowup when r+≈r-
        max_policy_norm = 30.0  # raised from 10: allow more exploration before capping
