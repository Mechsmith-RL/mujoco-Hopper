import argparse
import os
import pandas as pd
import gymnasium as gym


from stable_baselines3 import SAC, PPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--model", type=str, default=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/eval/ppo_seed0.csv")
    return p.parse_args()

def load_model(path: str):
    lower = path.lower()
    if "sac" in lower:
        return SAC.load(path)
    return PPO.load(path)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = gym.make(args.env)
    model = load_model(args.model)

    records = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_len = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_len += float(reward)
            ep_len +=1
            if terminated or truncated:
                break

        records.append({"episodes": ep, "return": ep_ret, "length": ep_len})

    env.close()

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)

    mean_ret = df["return"].mean()
    std_ret = df["return"].std(ddof=1) if len(df) > 1 else 0.0
    mean_len = df["length"].mean()

    print(f"[OK] Eval done: mean_return={mean_ret:.2f} ± {std_ret:.2f}, mean_len={mean_len:.1f}")
    print(f"[OK] CSV saved: {args.out}")


if __name__ == "__main__":
    main()