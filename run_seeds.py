import subprocess
import sys


def run(seed: int, total_steps: int = 500_000):
    cmd = [
        sys.executable, "train.py",
        "--env", "Hopper-v5",
        "--seed", str(seed),
        "--total-steps", str(total_steps),
        "--n-envs", "8",
        "--use-subproc", "1",
        "--eval-freq", "100000",
        "--eval-episodes", "10",
        "--device", "cpu"
    ]

    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    for s in [0, 1, 2]:
        run(s, total_steps=500_000) # 短跑验证，跑通后可改大到2_000_000

if __name__ == "__main__":
    main()