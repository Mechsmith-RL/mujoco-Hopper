import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_model(path: str):
    if "sac" in path.lower():
        return SAC.load(path)
    return PPO.load(path)


def load_vecnorm_for_obs_only(env_id: str, vecnorm_path: str) -> VecNormalize:
    """
    Load VecNormalize stats for observation normalization only (no reward norm),
    used for offline eval to keep protocol consistent with training.
    """
    def make():
        return gym.make(env_id)
    dummy = DummyVecEnv([make])
    vn = VecNormalize.load(vecnorm_path, dummy)
    vn.training = False
    vn.norm_reward = False
    return vn


def eval_one_model(
    env_id: str,
    model_path: str,
    episodes: int,
    eval_seed_base: int,
    vecnorm_path: str | None = None,
    deterministic: bool = True,
):
    env = gym.make(env_id)
    model = load_model(model_path)

    vecnorm = None
    if vecnorm_path is not None:
        vecnorm = load_vecnorm_for_obs_only(env_id, vecnorm_path)

    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=eval_seed_base + ep)

        if vecnorm is not None:
            obs = vecnorm.normalize_obs(obs[None, :])[0]

        ep_ret = 0.0
        ep_len = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

            if vecnorm is not None:
                obs = vecnorm.normalize_obs(obs[None, :])[0]

            if terminated or truncated:
                break

        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    if vecnorm is not None:
        # close underlying dummy vecenv
        vecnorm.venv.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1) if len(returns) > 1 else 0.0)
    mean_len = float(np.mean(lengths))
    return mean_ret, std_ret, mean_len


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eval-seed-base", type=int, default=123, help="base seed for eval episode resets")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="results/summary/day4_final_table.csv")
    p.add_argument("--out-agg", type=str, default="results/summary/day4_final_agg.csv")
    p.add_argument("--plot", type=str, default="plots/day4_final_bar.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--deterministic", type=int, default=1, help="1=deterministic, 0=stochastic")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(Path(args.out_agg).parent, exist_ok=True)
    if not args.no_plot:
        os.makedirs(Path(args.plot).parent, exist_ok=True)

    ckpt_dir = Path(args.checkpoints_dir)

    records = []

    # naming convention from your Day4 train.py:
    # ppo_{env}_{variant}_seed{seed}_best.zip
    # obsnorm also has: ppo_{env}_obsnorm_seed{seed}_vecnormalize.pkl
    for variant in ["control", "obsnorm"]:
        for s in args.seeds:
            exp = f"ppo_{args.env}_{variant}_seed{s}"
            model_path = ckpt_dir / f"{exp}_best.zip"

            if not model_path.exists():
                print(f"[WARN] missing model: {model_path}")
                continue

            vecnorm_path = None
            if variant == "obsnorm":
                vecnorm_path = str(ckpt_dir / f"{exp}_vecnormalize.pkl")
                if not Path(vecnorm_path).exists():
                    print(f"[WARN] missing vecnormalize stats for obsnorm: {vecnorm_path}")
                    vecnorm_path = None  # still run, but it will likely be underestimated

            mean_ret, std_ret, mean_len = eval_one_model(
                env_id=args.env,
                model_path=str(model_path),
                episodes=args.episodes,
                eval_seed_base=args.eval_seed_base,
                vecnorm_path=vecnorm_path,
                deterministic=bool(args.deterministic),
            )

            records.append({
                "env": args.env,
                "variant": variant,
                "seed": s,
                "episodes": args.episodes,
                "eval_seed_base": args.eval_seed_base,
                "model_path": str(model_path),
                "vecnorm_path": vecnorm_path if vecnorm_path is not None else "",
                "mean_return": mean_ret,
                "std_return": std_ret,
                "mean_len": mean_len,
            })

            print(f"[OK] {variant} seed{s}: mean_return={mean_ret:.2f} ± {std_ret:.2f}, mean_len={mean_len:.1f}")

    if not records:
        raise RuntimeError("No records collected. Check checkpoints names/paths and rerun.")

    df = pd.DataFrame(records).sort_values(["variant", "seed"])
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote final table: {args.out}")

    # aggregate across seeds (mean of means; std across seeds)
    agg = df.groupby("variant").agg(
        seeds=("seed", "count"),
        mean_return=("mean_return", "mean"),
        std_across_seeds=("mean_return", "std"),
        mean_len=("mean_len", "mean"),
    ).reset_index()
    agg.to_csv(args.out_agg, index=False)
    print(f"[OK] wrote agg table: {args.out_agg}")

    # optional plot
    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        # bar of mean_return with errorbar = std across seeds
        x = np.arange(len(agg))
        means = agg["mean_return"].to_numpy()
        errs = agg["std_across_seeds"].fillna(0.0).to_numpy()
        labels = agg["variant"].tolist()

        plt.bar(x, means, yerr=errs, capsize=6)
        plt.xticks(x, labels)
        plt.ylabel("mean_return (avg over seeds)")
        plt.title("Day4 Final Comparison (best checkpoints)")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        print(f"[OK] saved plot: {args.plot}")


if __name__ == "__main__":
    main()