from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO

# H1-2 has 27 DOF: 12 legs + 1 torso + 14 arms (including 3 wrist DOF per arm)
# Proprioception = 3 (ang_vel) + 3 (gravity) + 27 (dof_pos) + 27 (dof_vel) + 27 (actions)
PROPRIOCEPTION_DIM = 87
CMD_DIM = 3 + 4 + 1 + 2
TERRAIN_DIM = 221
# Privileged: 3 (lin_vel) + 1 (jump_h_err) + 2 (foot_clearance) + 1 (friction) + 6 (contact_forces 2*3) + 11 (collision_states)
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 11
CLOCK_INPUT = 2


class H1_2Cfg(H1Cfg):
    class env(H1Cfg.env):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
        num_actions = 27

    class init_state(H1Cfg.init_state):
        pos = [0.0, 0.0, 1.05]  # official Unitree H1-2 spawn height
        default_joint_angles = {  # source: unitreerobotics/unitree_rl_gym h1_2_config.py
            # Left leg (6 DOF)
            'left_hip_yaw_joint':     0.00,
            'left_hip_roll_joint':    0.00,
            'left_hip_pitch_joint':  -0.16,
            'left_knee_joint':        0.36,
            'left_ankle_pitch_joint': -0.20,
            'left_ankle_roll_joint':   0.00,
            # Right leg (6 DOF)
            'right_hip_yaw_joint':    0.00,
            'right_hip_roll_joint':   0.00,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint':       0.36,
            'right_ankle_pitch_joint': -0.20,
            'right_ankle_roll_joint':   0.00,
            # Torso (1 DOF)
            'torso_joint': 0.00,
            # Left arm (7 DOF)
            'left_shoulder_pitch_joint':  0.40,
            'left_shoulder_roll_joint':   0.00,
            'left_shoulder_yaw_joint':    0.00,
            'left_elbow_joint':           0.30,
            'left_wrist_roll_joint':      0.00,
            'left_wrist_pitch_joint':     0.00,
            'left_wrist_yaw_joint':       0.00,
            # Right arm (7 DOF)
            'right_shoulder_pitch_joint': 0.40,
            'right_shoulder_roll_joint':  0.00,
            'right_shoulder_yaw_joint':   0.00,
            'right_elbow_joint':          0.30,
            'right_wrist_roll_joint':     0.00,
            'right_wrist_pitch_joint':    0.00,
            'right_wrist_yaw_joint':      0.00,
        }

    class control(H1Cfg.control):
        stiffness = {
            'hip_yaw':        200,
            'hip_roll':       200,
            'hip_pitch':      200,
            'knee':           300,
            'ankle_pitch':     40,  # Kp=80 (Unitree PR-mode) causes oscillation at 50Hz control
            'ankle_roll':      40,  # keeping original; high Kp unstable at sim control rate
            'torso':          300,
            'shoulder_pitch':  20,
            'shoulder_roll':   20,
            'shoulder_yaw':    20,
            'elbow':           20,
            'wrist_roll':      10,
            'wrist_pitch':     10,
            'wrist_yaw':       10,
        }
        damping = {
            'hip_yaw':        5,
            'hip_roll':       5,
            'hip_pitch':      5,
            'knee':           6,
            'ankle_pitch':    1,  # Unitree PR-mode Kd=1 (was 2, lower damping helps response)
            'ankle_roll':     1,  # same
            'torso':          6,
            'shoulder_pitch': 0.5,
            'shoulder_roll':  0.5,
            'shoulder_yaw':   0.5,
            'elbow':          0.5,
            'wrist_roll':     0.3,
            'wrist_pitch':    0.3,
            'wrist_yaw':      0.3,
        }
        torque_limits = {
            'hip_yaw':        180,
            'hip_roll':       180,
            'hip_pitch':      180,
            'knee':           280,
            'ankle_pitch':     60,  # URDF effort=60 Nm (was 38, same as H1)
            'ankle_roll':      38,  # URDF effort=40 Nm (unchanged)
            'torso':          180,
            'shoulder_pitch':  35,
            'shoulder_roll':   35,
            'shoulder_yaw':    15,
            'elbow':           18,  # URDF effort=18 Nm (was 35, copied from H1)
            'wrist_roll':      10,
            'wrist_pitch':     10,
            'wrist_yaw':       10,
        }

    class asset(H1Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/urdf/h1_2_handless.urdf'
        name = "h1_2"
        # H1-2 root body is 'pelvis'; 'torso_link' is a child connected via torso_joint
        base_name = "pelvis"
        # Most distal leg body; ankle_pitch is intermediate, ankle_roll is the foot
        foot_name = "ankle_roll"
        auxiliary_foot_link = ["left_ankle_roll_link", "right_ankle_roll_link"]
        # Penalize same body groups as H1; torso_link exists and is a valid contact body
        penalize_contacts_on = ["elbow", "torso", "hip", "knee"]
        # Terminate when the pelvis (root) hits the ground
        terminate_after_contacts_on = ["pelvis"]

    class rewards(H1Cfg.rewards):
        # H1-2 root is pelvis (not torso_link like H1). Natural standing height ≈ 0.87 m
        # with default pose (hip_pitch=-0.40, knee=0.80, ankle_pitch=-0.40).
        base_height_target = 1.0   # official Unitree value; matches natural standing height with -0.16/0.36/-0.20 angles


class H1_2CfgPPO(H1CfgPPO):
    class algorithm(H1CfgPPO.algorithm):
        robot_type = 'h1_2'

    class policy(H1CfgPPO.policy):
        class NetModel:
            class MlpAdaptModel:
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim = CMD_DIM
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM
                latent_dim = 32
                privileged_recon_dim = 3
                max_length = H1_2Cfg.env.include_history_steps
                actor_hidden_dims = [256, 128, 32]
                mlp_hidden_dims = [256, 128]

        critic_hidden_dims = [512, 256, 128]
        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + PRIVILEGED_DIM + TERRAIN_DIM

    class runner(H1CfgPPO.runner):
        experiment_name = 'h1_2_teacher'
        resume = False
        resume_path = None
        save_interval = 2000
