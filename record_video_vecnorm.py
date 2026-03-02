import argparse
import os
import cv2
import gymnasium as gym

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from register_envs import register_custom_envs
register_custom_envs()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--vecnorm", type=str, default=None, help="VecNormalize stats .pkl (required if trained with obs_norm)")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def load_model(path: str):
    if "sac" in path.lower():
        return SAC.load(path)
    return PPO.load(path)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    def make():
        return gym.make(args.env, render_mode="rgb_array")
    venv = DummyVecEnv([make])

    # If vecnorm stats provided, load it and disable training updates
    if args.vecnorm is not None:
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    model = load_model(args.model)

    obs = venv.reset()
    # DummyVecEnv returns (n_env, obs_dim)
    # seed control:
    venv.envs[0].reset(seed=args.seed)

    frame = venv.envs[0].render()
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

        frame = venv.envs[0].render()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if done[0]:
            obs = venv.reset()

    writer.release()
    venv.close()
    print(f"[OK] saved: {args.video}")


if __name__ == "__main__":
    main()