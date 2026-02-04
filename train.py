import argparse
import os
import time

import gymnasium as gym
import torch.cuda
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor


def make_env(env_id: str, rank: int, seed: int):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        return env
    return _init

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--use-subproc", type=int, default=1, help="1=SubprocVecEnv, 0=DummyVecEnv")
    p.add_argument("--logdir", type=str, default="runs/ppo_hopper_seed0")
    p.add_argument("--save", type=str, default="checkpoints/ppo_hopper_seed0.zip")

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save),exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}, env={args.env}, n_envs={args.n_envs}, total_steps={args.total_steps}")

    envs_fns = [make_env(args.env, rank=i, seed=args.seed) for i in range(args.n_envs)]
    if args.use_subproc and args.n_envs > 1:
        vec_env = SubprocVecEnv(envs_fns)
    else:
        vec_env = DummyVecEnv(envs_fns)

    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=args.logdir,
        device=device,
    )

    start = time.time()
    model.learn(total_timesteps=args.total_steps, progress_bar=True)
    print(f"[OK] training finished in {(time.time()-start)/60:.1f} min")

    model.save(args.save)
    vec_env.close()
    print(f"[OK] saved model to: {args.save}")
    print(f"[OK] tensorboard logs under: {args.logdir}")

if __name__ == "__main__":
    main()
