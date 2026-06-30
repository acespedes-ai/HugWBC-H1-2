"""Entry point for the walk comparison project (ARS vs PPO).

Usage:
    # PPO (uses existing OnPolicyRunner)
    python legged_gym/scripts/train_walk.py --task h1_2walk --algo ppo --headless

    # ARS (uses ARSRunner, num_envs must equal 2*num_pairs = 120)
    python legged_gym/scripts/train_walk.py --task h1_2walk --algo ars --num_envs 120 --headless

    # ARS resume from checkpoint
    python legged_gym/scripts/train_walk.py --task h1_2walk --algo ars --num_envs 120 --headless \
        --load_run Jun28_17-36-43_ --checkpoint policy_350.npz
"""

import os
import sys
sys.path.append(os.getcwd())

import glob
import argparse
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def _pop_algo_and_resume():
    """Extract --algo, --load_run, --checkpoint before get_args() consumes sys.argv."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--algo',       type=str, default='ppo')
    pre.add_argument('--load_run',   type=str, default=None)
    pre.add_argument('--checkpoint', type=str, default=None)
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known.algo.lower(), known.load_run, known.checkpoint


def _find_ars_checkpoint(log_root, load_run, checkpoint):
    """Return (path, start_iter) for the chosen ARS checkpoint."""
    import numpy as np
    run_dir = os.path.join(log_root, load_run)
    assert os.path.isdir(run_dir), f"Run directory not found: {run_dir}"

    if checkpoint is not None:
        path = os.path.join(run_dir, checkpoint)
    else:
        # prefer policy_final.npz, else latest numbered one
        final = os.path.join(run_dir, 'policy_final.npz')
        if os.path.exists(final):
            path = final
        else:
            candidates = sorted(
                glob.glob(os.path.join(run_dir, 'policy_*.npz')),
                key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
            assert candidates, f"No policy .npz found in {run_dir}"
            path = candidates[-1]

    assert os.path.exists(path), f"Checkpoint not found: {path}"

    # Extract iteration number from filename (policy_350.npz → 350)
    name = os.path.basename(path)
    if name == 'policy_final.npz':
        start_iter = None   # unknown — will run full max_iterations more
    else:
        start_iter = int(name.split('_')[1].split('.')[0]) + 1

    return path, start_iter


def train(args, algo, load_run=None, checkpoint=None):

    if algo == 'ppo':
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)

    elif algo == 'ars':
        from legged_gym.envs.h1_2.h1_2_walk_config import H1_2WalkCfgARS
        from rsl_rl.runners.ars_runner import ARSRunner
        from datetime import datetime

        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        train_cfg = H1_2WalkCfgARS()

        # Verify num_envs matches 2*num_pairs
        expected = 2 * train_cfg.runner.num_pairs
        if env.num_envs != expected:
            raise ValueError(
                f"For ARS, --num_envs must be {expected} (2 × num_pairs={train_cfg.runner.num_pairs}). "
                f"Got {env.num_envs}.")

        log_root = os.path.join('logs', train_cfg.runner.experiment_name)
        log_dir  = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S_'))
        os.makedirs(log_dir, exist_ok=True)

        device = env.device
        runner = ARSRunner(env, train_cfg, log_dir=log_dir, device=device)

        if load_run is not None:
            ckpt_path, start_iter = _find_ars_checkpoint(log_root, load_run, checkpoint)
            runner.load(ckpt_path)
            if start_iter is not None:
                runner.current_iter = start_iter
                remaining = max(0, train_cfg.runner.max_iterations - start_iter)
                print(f"[ARS] resuming from iter {start_iter}  "
                      f"({remaining} iters remaining of {train_cfg.runner.max_iterations})")
            else:
                remaining = train_cfg.runner.max_iterations
                print(f"[ARS] resuming from policy_final  "
                      f"(running {remaining} additional iters)")
            runner.learn(num_iterations=remaining)
        else:
            runner.learn(num_iterations=train_cfg.runner.max_iterations)

    else:
        raise ValueError(f"Unknown --algo '{algo}'. Choose 'ppo' or 'ars'.")


if __name__ == '__main__':
    algo, load_run, checkpoint = _pop_algo_and_resume()
    args = get_args()
    train(args, algo, load_run=load_run, checkpoint=checkpoint)
