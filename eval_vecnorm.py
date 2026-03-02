import argparse
import os
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from register_envs import register_custom_envs
register_custom_envs()

def load_model(path: str):
    if "sac" in path.lower():
        return SAC.load(path)
    return PPO.load(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--vecnorm", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/eval/obsnorm_seed0_best.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def make():
        return gym.make(args.env)
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(args.vecnorm, venv)
    venv.training = False
    venv.norm_reward = False

    model = load_model(args.model)

    records = []
    for ep in range(args.episodes):
        venv.envs[0].reset(seed=args.seed + ep)
        obs = venv.reset()
        ep_ret, ep_len = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_ret += float(reward[0])
            ep_len += 1
            if done[0]:
                break
        records.append({"episode": ep, "return": ep_ret, "length": ep_len})

    venv.close()

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print("[OK] mean_return=", df["return"].mean(), "std=", df["return"].std(ddof=1))

if __name__ == "__main__":
    main()