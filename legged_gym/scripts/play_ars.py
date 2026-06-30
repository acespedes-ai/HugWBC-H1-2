"""Evaluation / visualization script for ARS linear policies.

Usage:
    python legged_gym/scripts/play_ars.py \
        --task h1_2walk \
        --load_run Jun28_17-36-43_ \
        [--checkpoint policy_350.npz]  # default: latest available

Metrics logged per episode and summarised at the end:
    - Episode length (survival)
    - Velocity tracking error  (lin_vel XY, yaw)
    - Cost of Transport  CoT = sum|tau*dqdt| / (m*g*v*T)
    - Action smoothness  mean|a_t - a_{t-1}|
    - Base angular velocity std  (roll/pitch oscillation proxy)
    - Fall rate  (terminated by contact/orientation vs timeout)
"""

import os
import sys
sys.path.append(os.getcwd())

import glob
import argparse
from typing import Optional
import numpy as np
import tqdm
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry
from rsl_rl.algorithms.ars import LinearPolicy


ROBOT_MASS = 67.0   # kg  (H1-2 URDF total + ~0.85 kg hands)
GRAVITY    = 9.81   # m/s²


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',       type=str, default='h1_2walk')
    parser.add_argument('--load_run',   type=str, required=True,
                        help='Run folder name inside logs/<experiment_name>/')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Filename of .npz checkpoint (default: latest policy_*.npz)')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--cmd_vx',     type=float, default=1.0,
                        help='Commanded forward velocity (m/s)')
    parser.add_argument('--headless',   action='store_true')
    parser.add_argument('--device',     type=str, default='cuda:0')
    # keep for compatibility with task_registry
    parser.add_argument('--num_envs',   type=int, default=1)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--rl_device',  type=str, default='cuda:0')
    parser.add_argument('--sim_device', type=str, default='cuda:0')
    parser.add_argument('--physics_engine', type=str, default='physx')
    parser.add_argument('--pipeline',   type=str, default='gpu')
    parser.add_argument('--experiment_name', type=str, default='h1_2_walk_ars',
                        help='Log subdirectory (default: h1_2_walk_ars)')
    args = parser.parse_args()
    # gymutil.parse_arguments converts 'physx'/'flex' to gymapi constants; replicate that here
    from isaacgym import gymapi
    args.physics_engine = gymapi.SIM_PHYSX if args.physics_engine == 'physx' else gymapi.SIM_FLEX
    # attributes expected by parse_sim_params / Isaac Gym internals
    args.use_gpu          = (args.sim_device != 'cpu')
    args.use_gpu_pipeline = (args.pipeline == 'gpu')
    args.subscenes        = 0
    args.num_threads      = 0
    args.compute_device_id = 0
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_checkpoint(run_dir: str, checkpoint: Optional[str]) -> str:
    if checkpoint is not None:
        path = os.path.join(run_dir, checkpoint)
        assert os.path.exists(path), f"Checkpoint not found: {path}"
        return path
    # prefer policy_final.npz if it exists; otherwise pick highest-numbered checkpoint
    final = os.path.join(run_dir, 'policy_final.npz')
    if os.path.exists(final):
        return final
    def _step(p):
        stem = os.path.basename(p).split('_')[1].split('.')[0]
        return int(stem) if stem.isdigit() else float('inf')
    candidates = sorted(glob.glob(os.path.join(run_dir, 'policy_*.npz')), key=_step)
    assert candidates, f"No policy .npz files found in {run_dir}"
    return candidates[-1]


class EpisodeMetrics:
    """Accumulates per-step data and returns summary dict at episode end."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.steps         = 0
        self.torque_power  = 0.0   # sum |tau · dq/dt|
        self.vel_x_error   = 0.0   # sum |cmd_vx - actual_vx|
        self.vel_y_error   = 0.0
        self.yaw_error     = 0.0
        self.action_diffs  = []    # |a_t - a_{t-1}|
        self.base_ang_vels = []    # |omega_base| each step
        self.last_action   = None

    def update(self, torques, dof_vel, base_lin_vel, base_ang_vel,
               cmd_vx, cmd_vy, cmd_yaw, action):
        self.steps += 1
        self.torque_power += (torques.abs() * dof_vel.abs()).sum().item()
        self.vel_x_error  += abs(cmd_vx - base_lin_vel[0].item())
        self.vel_y_error  += abs(cmd_vy - base_lin_vel[1].item())
        self.yaw_error    += abs(cmd_yaw - base_ang_vel[2].item())
        self.base_ang_vels.append(base_ang_vel[:2].norm().item())
        if self.last_action is not None:
            self.action_diffs.append((action - self.last_action).abs().mean().item())
        self.last_action = action.clone()

    def summary(self, mean_vel_x: float, dt: float, fell: bool) -> dict:
        T = self.steps * dt
        mean_pow = self.torque_power / max(self.steps, 1)
        denom = ROBOT_MASS * GRAVITY * max(abs(mean_vel_x), 0.01) * T
        cot = (self.torque_power * dt) / denom if denom > 0 else float('nan')
        return {
            'ep_len':          self.steps,
            'fell':            int(fell),
            'tracking_vx':     self.vel_x_error  / max(self.steps, 1),
            'tracking_vy':     self.vel_y_error  / max(self.steps, 1),
            'tracking_yaw':    self.yaw_error    / max(self.steps, 1),
            'cot':             cot,
            'smoothness':      float(np.mean(self.action_diffs)) if self.action_diffs else 0.0,
            'base_ang_std':    float(np.std(self.base_ang_vels)),
            'mean_power':      mean_pow,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def play(args):
    device = args.device

    # --- Environment -------------------------------------------------------
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs          = 1
    env_cfg.env.episode_length_s  = 20          # 20s episodes
    env_cfg.terrain.curriculum    = False
    env_cfg.noise.add_noise       = False        # deterministic eval
    env_cfg.domain_rand.randomize_friction      = False
    env_cfg.domain_rand.randomize_load          = False
    env_cfg.domain_rand.randomize_gains         = False
    env_cfg.domain_rand.randomize_link_props    = False
    env_cfg.domain_rand.randomize_base_mass     = False
    env_cfg.domain_rand.randomize_control_latency = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if not args.headless:
        for i in range(env.num_bodies):
            env.gym.set_rigid_body_color(
                env.envs[0], env.actor_handles[0], i,
                gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))

    # --- Policy ------------------------------------------------------------
    experiment_name = args.experiment_name
    run_dir = os.path.join('logs', experiment_name, args.load_run)
    ckpt_path = find_checkpoint(run_dir, args.checkpoint)
    print(f"[ARS play] loading policy from: {ckpt_path}")

    policy = LinearPolicy(env.num_obs, env.num_actions, device)
    d = np.load(ckpt_path)
    policy.load_state_dict({k: d[k] for k in d.files})

    # --- Camera ------------------------------------------------------------
    if not args.headless:
        camera_rot = np.pi * 0.8
        camera_rot_per_sec = np.pi * 0.1
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        cam_pos = look_at + 2 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.4])
        env.set_camera(cam_pos, look_at, 0)

    # --- Eval loop ---------------------------------------------------------
    all_metrics = []
    max_ep_len  = int(env_cfg.env.episode_length_s / env.dt)

    print(f"\n[ARS play] evaluating {args.num_episodes} episodes  "
          f"(cmd_vx={args.cmd_vx} m/s, max_ep_len={max_ep_len})\n")

    def _set_cmds():
        env.commands[:, 0] = args.cmd_vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        # gait params are fixed by the walk env's _resample_commands (freq=2, phase=0.5, etc.)

    for ep in range(args.num_episodes):
        env.reset()
        _set_cmds()   # override BEFORE the first step so obs[0] already has correct commands
        obs, _, _, _, _ = env.step(
            torch.zeros(1, env.num_actions, device=device))

        metrics = EpisodeMetrics(device)
        fell = False

        for step in range(max_ep_len):
            _set_cmds()   # keep overriding every step (env resamples every 10 s)
            with torch.inference_mode():
                action = policy.act(obs)
            obs, _, _, dones, _ = env.step(action)

            metrics.update(
                torques      = env.torques[0],
                dof_vel      = env.dof_vel[0],
                base_lin_vel = env.base_lin_vel[0],
                base_ang_vel = env.base_ang_vel[0],
                cmd_vx       = args.cmd_vx,
                cmd_vy       = 0.0,
                cmd_yaw      = 0.0,
                action       = action[0],
            )

            base_height = env.root_states[0, 2].item()
            if dones[0] or base_height < 0.35:
                fell = True
                break

            if not args.headless:
                look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
                camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
                cam_pos = look_at + 2 * np.array(
                    [np.cos(camera_rot), np.sin(camera_rot), 0.4])
                env.set_camera(cam_pos, look_at, 0)

        ep_metrics = metrics.summary(mean_vel_x=args.cmd_vx, dt=env.dt, fell=fell)
        all_metrics.append(ep_metrics)
        print(f"  ep {ep+1:3d}: len={ep_metrics['ep_len']:4d}  "
              f"fell={ep_metrics['fell']}  "
              f"vx_err={ep_metrics['tracking_vx']:.3f}  "
              f"CoT={ep_metrics['cot']:.3f}  "
              f"smooth={ep_metrics['smoothness']:.4f}")

    # --- Summary -----------------------------------------------------------
    print("\n" + "="*60)
    print(f"SUMMARY  ({args.num_episodes} episodes,  cmd_vx={args.cmd_vx})")
    print(f"  checkpoint : {os.path.basename(ckpt_path)}")
    print("="*60)
    keys = list(all_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:20s}  mean={np.mean(vals):8.4f}   std={np.std(vals):8.4f}")
    print()
    fall_rate = np.mean([m['fell'] for m in all_metrics])
    print(f"  fall_rate            {fall_rate*100:.1f}%")


if __name__ == '__main__':
    args = get_args()
    play(args)
