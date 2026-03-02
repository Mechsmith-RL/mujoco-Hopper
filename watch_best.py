import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC

def load_model(model_path: str):
    # 你也可以直接写 PPO.load(model_path)，这里做个兼容
    lower = model_path.lower()
    if "sac" in lower:
        return SAC.load(model_path)
    return PPO.load(model_path)

def main():
    env_id = "Hopper-v5"
    model_path = "checkpoints/ppo_Hopper-v5_seed2_best.zip"  # 改成你的 best 路径

    env = gym.make(env_id, render_mode="human")
    model = load_model(model_path)

    obs, info = env.reset(seed=0)

    for ep in range(5):
        terminated = truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_ret += float(reward)
            ep_len += 1

            # 可选：控制播放速度（不然可能跑得太快）
            time.sleep(1/60)

        print(f"episode {ep}: return={ep_ret:.2f}, len={ep_len}")
        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
