from pathlib import Path
from gymnasium.envs.registration import register

def register_custom_envs():
    root = Path(__file__).resolve().parent
    xml = root / "assets" / "hopper_stairs.xml"
    if not xml.exists():
        # 没生成 xml 就不注册
        return

    try:
        register(
            id="HopperStairs-v0",
            entry_point="gymnasium.envs.mujoco.hopper_v5:HopperEnv",
            kwargs={"xml_file": str(xml)},
            max_episode_steps=1000,
        )
    except Exception:
        # 已经注册过会报错，直接忽略
        pass