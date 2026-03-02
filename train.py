import argparse
import os
import time
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

from callbacks import PeriodicEvalSaveBestCallback

from register_envs import register_custom_envs
register_custom_envs()

def make_env(env_id: str, rank: int, seed: int):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Hopper-v5")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--use-subproc", type=int, default=1, help="1=SubprocVecEnv, 0=DummyVecEnv")

    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--results-dir", type=str, default="results/eval")

    # PPO hyperparams
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)

    # Eval during training
    p.add_argument("--eval-freq", type=int, default=200_000, help="eval every N steps(timesteps)")
    p.add_argument("--eval-episodes", type=int, default=10)

    # Single-variable ablation: obs normalization
    p.add_argument("--obs-norm", type=int, default=0, help="0=off, 1=on(VecNormalize norm_obs)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()

def main():
    args = parse_args()
    set_random_seed(args.seed)

    variant = "obsnorm" if args.obs_norm == 1 else "control"
    exp_name = f"ppo_{args.env}_{variant}_seed{args.seed}"

    logdir = args.logdir or os.path.join("runs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    last_path = os.path.join(args.save_dir, f"{exp_name}_last.zip")
    best_path = os.path.join(args.save_dir, f"{exp_name}_best.zip")
    eval_csv = os.path.join(args.results_dir, f"{exp_name}_eval_log.csv")
    vecnorm_path = os.path.join(args.save_dir, f"{exp_name}_vecnormalize.pkl")

    print(f"[INFO] exp={exp_name}")
    print(f"[INFO] obs_norm={args.obs_norm} (single-variable ablation)")
    print(f"[INFO] device={args.device}, n_envs={args.n_envs}, total_steps={args.total_steps}")
    print(f"[INFO] logdir={logdir}")
    print(f"[INFO] last={last_path}")
    print(f"[INFO] best={best_path}")
    print(f"[INFO] eval_log={eval_csv}")

    if args.obs_norm == 1:
        print(f"[INFO] vecnormalize={vecnorm_path}")

    envs_fns = [make_env(args.env, i, args.seed) for i in range(args.n_envs)]
    if args.use_subproc and args.n_envs > 1:
        vec_env = SubprocVecEnv(envs_fns)
    else:
        vec_env = DummyVecEnv(envs_fns)

    vec_env = VecMonitor(vec_env)

    if args.obs_norm == 1:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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
        verbose=1,
        tensorboard_log=logdir,
        device=args.device,
    )

    callback = PeriodicEvalSaveBestCallback(
        env_id=args.env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed+ 10_000,
        best_model_path=best_path,
        csv_log_path=eval_csv,
        verbose=1
    )

    start = time.time()
    model.learn(total_timesteps=args.total_steps,  callback=callback, progress_bar=True)
    print(f"[OK] training finished in {(time.time()-start)/60:.1f} min")

    model.save(last_path)

    if args.obs_norm == 1:
        vec_env.save(vecnorm_path)

    vec_env.close()
    print(f"[OK] saved last model : {last_path}")
    print(f"[OK] saved best model : {best_path}")
    print(f"[OK] eval log csv: {eval_csv}")
    if args.obs_norm == 1:
        print(f"[OK] saved vecnormalize stats: {vecnorm_path}")

if __name__ == "__main__":
    main()
