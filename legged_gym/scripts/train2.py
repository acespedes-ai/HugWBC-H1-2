"""train2.py — Resume training with curricula pre-set to a run's final state.

Usage:
    python legged_gym/scripts/train2.py --task h1_2int --load_run <run_name> [--num_envs 4096]

What it does vs train.py:
  - Loads network weights from the latest checkpoint of <load_run>
  - Pre-sets all curricula to their approximate final values so training
    continues at full difficulty without re-ramping from scratch:
      * curriculum_scale     = 1.0     (penalty curriculum fully mature)
      * disturb_rad_curriculum = DISTURB_VAL (arm perturbation at max)
      * terrain_curriculum_mode = False for all noise envs (disturb mode active)
      * command curriculum grid pre-warmed to CMD_X_MAX m/s
  - Continues for EXTRA_ITERS additional iterations

Edit the constants below to match the run you are resuming.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# ── Tune these to match the checkpoint being resumed ──────────────────────────
# Last updated: Jun24_09-42-27_ @ checkpoint iter 8000
# UPDATE these values before relaunching from a new checkpoint.
DISTURB_VAL  = 0.015   # disturb_rad_curriculum approx at iter 8000
CMD_X_MAX    = 1.66    # max_command_x at iter 8000 (m/s)
CMD_YAW_MAX  = 1.0     # max_command_yaw reached (rad/s)
EXTRA_ITERS  = 32000   # remaining iters (40000 - 8000)
# ──────────────────────────────────────────────────────────────────────────────


def train2(args):
    args.resume = True  # load weights from load_run checkpoint

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # ── Pre-set curricula so training resumes at full difficulty ──────────────

    # 1. Penalty curriculum — fully mature
    env.curriculum_scale = 1.0

    # 2. Arm perturbation curriculum — set to final value of source run
    env.disturb_rad_curriculum[:] = DISTURB_VAL

    # 3. Release all noise envs into disturb mode (terrain_curriculum_mode=False)
    noise_ids = torch.arange(env.noise_env_nums, device=env.device)
    env.terrain_curriculum_mode[noise_ids] = False

    # 4. Command velocity curriculum — pre-warm grid up to CMD_X_MAX
    low  = np.array([env_cfg.commands.ranges.lin_vel_x[0],
                     env_cfg.commands.ranges.ang_vel_yaw[0]])
    high = np.array([CMD_X_MAX, CMD_YAW_MAX])
    for curriculum in env.curricula:
        curriculum.set_to(low=low, high=high)

    # ─────────────────────────────────────────────────────────────────────────
    ppo_runner.learn(num_learning_iterations=EXTRA_ITERS, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train2(args)
