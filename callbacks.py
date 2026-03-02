import os
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback

@dataclass
class EvalResult:
    mean_reward: float
    std_reward: float
    mean_len: float

def evaluate_policy_simple(model, env_id: str, n_episodes: int, seed: int) -> EvalResult:
    env = gym.make(env_id)
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    return EvalResult(
        mean_reward=float(np.mean(returns)),
        std_reward=float(np.std(returns, ddof=1) if len(returns) > 1 else 0.0),
        mean_len=float(np.mean(lengths))
    )

class PeriodicEvalSaveBestCallback(BaseCallback):
    def __init__(
            self,
            env_id: str,
            eval_freq: int,
            n_eval_episodes,
            seed: int,
            best_model_path: str,
            csv_log_path,
            verbose: int=0
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self.best_model_path = best_model_path
        self.csv_log_path = csv_log_path
        self.best_mean_reward = -np.inf

        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)

        if not os.path.exists(csv_log_path):
            df = pd.DataFrame(columns=["timesteps", "mean_reward", "std_reward", "mean_len"])
            df.to_csv(csv_log_path, index=False)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            res = evaluate_policy_simple(
                self.model, env_id=self.env_id, n_episodes=self.n_eval_episodes, seed =self.seed
            )

            self.logger.record("eval/mean_reward", res.mean_reward)
            self.logger.record("eval/std_reward", res.std_reward)
            self.logger.record("eval/mean_len", res.mean_len)

            df = pd.DataFrame([{
                "timesteps": self.num_timesteps,
                "mean_reward": res.mean_reward,
                "std_reward": res.std_reward,
                "mean_len": res.mean_len
            }])
            df.to_csv(self.csv_log_path, mode='a', header=False, index=False)

            if res.mean_reward > self.best_mean_reward:
                self.best_mean_reward = res.mean_reward
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"[BEST] t={self.num_timesteps} "
                          f"mean_reward={res.mean_reward:.2f} -> saved {self.best_model_path}")
        return True