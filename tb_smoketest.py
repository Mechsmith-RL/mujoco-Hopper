from stable_baselines3.common.logger import configure
import time

logger = configure("runs/smoke_test", ["stdout", "tensorboard"])
for i in range(10):
    logger.record("smoke/step", i)
    logger.record("smoke/value", i * i)
    logger.dump(i)
    time.sleep(0.05)

print("[OK] wrote tensorboard logs to runs/smoke_test")
