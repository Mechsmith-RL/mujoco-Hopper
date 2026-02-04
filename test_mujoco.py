import gymnasium as gym
import torch
import mujoco  # noqa: F401

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

env = gym.make("Hopper-v5")
obs, info = env.reset()
print("obs shape:", obs.shape)
env.close()

print("[OK] env_smoketest passed.")
