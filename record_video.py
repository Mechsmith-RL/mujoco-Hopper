import argparse
import os
import gymnasium as gym

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--model", type=str, default=None, help="SB3 model .zip (PPO/SAC). If None, random policy.")
    p.add_argument("--video", type=str, default="videos/out/hopper.mp4", help="Target video path (folder + base name used).")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()

    video_folder = os.path.dirname(args.video) or "videos"
    name_prefix = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array")

    # RecordVideo 会输出：video_folder/name_prefix-episode-0.mp4
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda ep_id: ep_id == 0,  # 只录第一个 episode，避免一堆碎片
        disable_logger=True,
    )

    obs, info = env.reset(seed=args.seed)

    model = None
    if args.model is not None:
        from stable_baselines3 import PPO, SAC
        if "sac" in args.model.lower():
            model = SAC.load(args.model)
        else:
            model = PPO.load(args.model)

    for _ in range(args.steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"[OK] video_folder={video_folder}")
    print(f"[OK] name_prefix={name_prefix}")
    print(f"[NOTE] output file will look like: {video_folder}/{name_prefix}-episode-0.mp4")

if __name__ == "__main__":
    main()
