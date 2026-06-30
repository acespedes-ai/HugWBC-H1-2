from legged_gym.envs.h1_2.h1_2_config import H1_2Cfg, H1_2CfgPPO

PROPRIOCEPTION_DIM = 81   # Option B: 3+3+27+27+21 (wrists excluded from NN output)
INTERRUPT_IN_CMD   = True
CMD_DIM    = 3 + 4 + 1 + 2 + INTERRUPT_IN_CMD  # = 11  (adds interrupt flag vs H1_2's 10)
TERRAIN_DIM    = 221
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 11        # = 24  same structure as H1_2
CLOCK_INPUT    = 2
DISTURB_DIM    = 14  # 7 arm joints (shoulder×3 + elbow + wrist×3) × 2 sides


class H1_2InterruptCfg(H1_2Cfg):
    class env(H1_2Cfg.env):
        num_actions      = 21  # Option B: 12 legs + 1 torso + 4 left arm + 4 right arm (no wrists)
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs  = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT

    class commands(H1_2Cfg.commands):
        num_commands = CMD_DIM

    class rewards(H1_2Cfg.rewards):
        reward_curriculum_list = [
            'action_rate_upper', 'action_rate_lower',
            'feet_stumble',
            'joint_power_distribution', 'feet_contact_forces',
            'dof_acc', 'torques',
            'base_height', 'collision', 'stand_still',
            'lin_vel_z', 'base_height_min', 'dof_vel_limits',
            'ang_vel_xy',
            'shoulder_yaw_deviation', 'shoulder_roll_deviation',
            'shoulder_pitch_deviation', 'elbow_deviation',
            'torso_deviation',
            'hopping_symmetry',
            'jump',
            'orientation_control',
            'standing_air',
            'standing_vel',
        ]
        class scales(H1_2Cfg.rewards.scales):
            # Split action_rate into lower (legs+torso) and upper (arms) like h1int
            action_rate       =  0
            action_rate_lower = -0.01
            action_rate_upper = -0.01
            wrist_deviation   = 0.0    # disabled: wrists not in NN output (Option B)
            action_rate_wrist = 0.0    # disabled: wrists not in NN output (Option B)
            base_height       = -40.0
            stand_still       = -10.0
            standing          =  2.0
            orientation_control = -10.0
            standing_air      = -2.0

    class disturb:
        max_curriculum = 1.0
        use_disturb    = True
        disturb_dim    = DISTURB_DIM
        disturb_scale  = 2

        # Joint ranges from h1_2_handless.urdf  (upper - lower).
        # Order mirrors DOF order expected at runtime (left arm first, right arm second):
        #   l_sh_pitch, l_sh_roll, l_sh_yaw, l_elbow,
        #   l_wr_roll, l_wr_pitch, l_wr_yaw,
        #   r_sh_pitch, r_sh_roll, r_sh_yaw, r_elbow,
        #   r_wr_roll, r_wr_pitch, r_wr_yaw
        noise_scale = [
            4.71,   # left_shoulder_pitch:  -3.14 ~ 1.57
            3.78,   # left_shoulder_roll:   -0.38 ~ 3.40
            5.67,   # left_shoulder_yaw:    -2.66 ~ 3.01
            4.13,   # left_elbow:           -0.95 ~ 3.18
            5.76,   # left_wrist_roll:      -3.01 ~ 2.75
            0.925,  # left_wrist_pitch:    -0.4625 ~ 0.4625
            2.54,   # left_wrist_yaw:       -1.27 ~ 1.27
            4.71,   # right_shoulder_pitch: -3.14 ~ 1.57
            3.78,   # right_shoulder_roll:  -3.40 ~ 0.38
            5.67,   # right_shoulder_yaw:   -3.01 ~ 2.66
            4.13,   # right_elbow:          -0.95 ~ 3.18
            5.76,   # right_wrist_roll:     -2.75 ~ 3.01
            0.925,  # right_wrist_pitch:   -0.4625 ~ 0.4625
            2.54,   # right_wrist_yaw:      -1.27 ~ 1.27
        ]
        noise_lowerbound = [
            -3.14,    # left_shoulder_pitch
            -0.38,    # left_shoulder_roll
            -2.66,    # left_shoulder_yaw
            -0.95,    # left_elbow
            -3.01,    # left_wrist_roll
            -0.4625,  # left_wrist_pitch
            -1.27,    # left_wrist_yaw
            -3.14,    # right_shoulder_pitch
            -3.40,    # right_shoulder_roll
            -3.01,    # right_shoulder_yaw
            -0.95,    # right_elbow
            -2.75,    # right_wrist_roll
            -0.4625,  # right_wrist_pitch
            -1.27,    # right_wrist_yaw
        ]
        uniform_scale  = 1
        uniform_noise  = True
        noise_ratio    = 1
        interrupt_action_buffer     = None
        start_by_curriculum         = True
        replace_action              = True
        disturb_rad                 = 0.2
        # Per-joint scale on the clip radius: torque_limit / max_arm_torque (shoulder=35 Nm).
        # Update wrist values when hands with higher Kp/limits are added.
        disturb_rad_scale = [1.00] * 14  # uniform across all 14 arm DOFs
        disturb_rad_curriculum      = True
        disturb_curriculum_method   = 2
        noise_update_step           = 30
        switch_prob                 = 0.005
        interrupt_in_cmd            = INTERRUPT_IN_CMD
        stand_interrupt_only        = False
        noise_curriculum_ratio      = 0.5
        disturb_in_last_action      = False
        obs_target_interrupt_in_privilege  = False
        obs_executed_actions_in_privilege  = False
        disturb_terminate_assets    = []

    class curriculum_thresholds(H1_2Cfg.curriculum_thresholds):
        class terrains_level:
            tracking_lin_vel = 0.80
            dof_vel_limits   = 0.30  # original H1-2 value; wrists excluded from reward so dov≈0, any value passes
        class disturb:
            tracking_lin_vel = 0.6   # same as H1; split ankle helps H1-2 tracking, no reason to lower


class H1_2InterruptCfgPPO(H1_2CfgPPO):
    class runner(H1_2CfgPPO.runner):
        experiment_name = 'h1_2_interrupt'
        resume          = False
        resume_path     = None
        max_iterations  = 40000
        save_interval   = 2000

    class policy(H1_2CfgPPO.policy):
        model_name = "MlpAdaptModel"
        class NetModel:
            class MlpAdaptModel:
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim            = CMD_DIM + CLOCK_INPUT  # policy sees cmd+clock together
                privileged_dim     = PRIVILEGED_DIM
                terrain_dim        = TERRAIN_DIM
                latent_dim         = 32
                privileged_recon_dim = 3
                max_length         = H1_2InterruptCfg.env.include_history_steps
                actor_hidden_dims  = [256, 128, 32]
                mlp_hidden_dims    = [256, 128]

        critic_hidden_dims = [512, 256, 128]
        # Critic sees the full obs (partial + privileged + terrain)
        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
