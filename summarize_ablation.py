import argparse
import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", type=str, default="results/eval/ppo_Hopper-v5_*_seed*_eval_log.csv",
                   help="glob pattern to match eval_log.csv files")
    p.add_argument("--out-csv", type=str, default="results/summary/day4_obsnorm_summary.csv")
    p.add_argument("--out-fig", type=str, default="plots/day4_obsnorm_curve.png")
    return p.parse_args()


def read_group(files):
    # return df with index=timesteps, columns=seed, values=mean_reward
    series = []
    for f in files:
        df = pd.read_csv(f)
        # seed from filename
        m = re.search(r"_seed(\d+)_eval_log\.csv$", f)
        seed = int(m.group(1)) if m else 0
        s = df.set_index("timesteps")["mean_reward"].rename(f"seed{seed}")
        series.append(s)
    merged = pd.concat(series, axis=1).sort_index()
    return merged


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    control = [f for f in files if "_control_" in f]
    obsnorm = [f for f in files if "_obsnorm_" in f]

    if len(control) == 0 or len(obsnorm) == 0:
        raise RuntimeError("Need both control and obsnorm eval logs. Check filenames/pattern.")

    df_c = read_group(control)
    df_o = read_group(obsnorm)

    # Align timesteps intersection to be fair
    common_steps = sorted(set(df_c.index).intersection(set(df_o.index)))
    df_c = df_c.loc[common_steps]
    df_o = df_o.loc[common_steps]

    summary = pd.DataFrame({
        "timesteps": common_steps,
        "control_mean": df_c.mean(axis=1),
        "control_std": df_c.std(axis=1, ddof=1),
        "obsnorm_mean": df_o.mean(axis=1),
        "obsnorm_std": df_o.std(axis=1, ddof=1),
    })
    summary.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote summary csv: {args.out_csv}")

    # Plot mean curves (with light std band)
    plt.figure()
    plt.plot(summary["timesteps"], summary["control_mean"], label="control (obs_norm=off)")
    plt.plot(summary["timesteps"], summary["obsnorm_mean"], label="obs_norm=on")

    # std band
    plt.fill_between(
        summary["timesteps"],
        summary["control_mean"] - summary["control_std"],
        summary["control_mean"] + summary["control_std"],
        alpha=0.2,
    )
    plt.fill_between(
        summary["timesteps"],
        summary["obsnorm_mean"] - summary["obsnorm_std"],
        summary["obsnorm_mean"] + summary["obsnorm_std"],
        alpha=0.2,
    )

    plt.xlabel("timesteps")
    plt.ylabel("eval mean_reward")
    plt.title("Day4 Ablation: Observation Normalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"[OK] saved figure: {args.out_fig}")


if __name__ == "__main__":
    main()