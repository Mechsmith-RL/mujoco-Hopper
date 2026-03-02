import argparse
import os
import cv2
import gymnasium as gym

from register_envs import register_custom_envs
register_custom_envs()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="HopperStairs-v0")
    p.add_argument("--model", type=str, default=None, help="SB3 .zip; if None -> random")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    model = None
    if args.model is not None:
        from stable_baselines3 import PPO, SAC
        model = SAC.load(args.model) if "sac" in args.model.lower() else PPO.load(args.model)

    env = gym.make(args.env, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    frame = env.render()
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    for _ in range(args.steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, r, term, trunc, info = env.step(action)
        frame = env.render()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if term or trunc:
            obs, _ = env.reset()

    writer.release()
    env.close()
    print(f"[OK] saved: {args.video}")

if __name__ == "__main__":
    main()