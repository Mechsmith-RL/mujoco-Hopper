import subprocess
import sys


def run(seed: int, obs_norm: int, total_steps: int, eval_freq: int):
    cmd = [
        sys.executable, "train.py",
        "--env", "Hopper-v5",
        "--seed", str(seed),
        "--obs-norm", str(obs_norm),
        "--total-steps", str(total_steps),
        "--n-envs", "8",
        "--use-subproc", "1",
        "--eval-freq", str(eval_freq),
        "--eval-episodes", "10",
        "--device", "cpu",
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    seeds = [0, 1, 2]
    total_steps = 500_000
    eval_freq = 100_000

    for s in seeds:
        run(seed=s, obs_norm=0, total_steps=total_steps, eval_freq=eval_freq)

    for s in seeds:
        run(seed=s, obs_norm=1, total_steps=total_steps, eval_freq=eval_freq)

if __name__ == "__main__":
    main()
