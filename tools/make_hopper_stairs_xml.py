import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def find_base_hopper_xml() -> Path:
    # gymnasium/envs/mujoco/assets/hopper.xml
    import gymnasium.envs.mujoco as mujoco_pkg
    base = Path(mujoco_pkg.__file__).resolve().parent / "assets" / "hopper.xml"
    if not base.exists():
        raise FileNotFoundError(f"Cannot find hopper.xml at: {base}")
    return base

def add_stairs(worldbody: ET.Element, n_steps: int, step_h: float, step_l: float,
               width: float, start_x: float, friction: str):
    """
    Add a simple staircase made of box geoms.
    x direction is forward. floor is at z=0.
    """
    # Each step is a box. For step i:
    # top height = (i+1)*step_h
    # center z = top - step_h/2 = (i+0.5)*step_h
    # center x = start_x + i*step_l + step_l/2
    half_l = step_l / 2.0
    half_w = width / 2.0
    half_h = step_h / 2.0

    for i in range(n_steps):
        cx = start_x + i * step_l + half_l
        cz = (i + 0.5) * step_h
        geom = ET.Element("geom")
        geom.set("name", f"stair_{i}")
        geom.set("type", "box")
        geom.set("size", f"{half_l} {half_w} {half_h}")
        geom.set("pos", f"{cx} 0 {cz}")
        geom.set("rgba", "0.6 0.6 0.6 1")
        geom.set("friction", friction)
        worldbody.append(geom)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="assets/hopper_stairs.xml")
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--step-height", type=float, default=0.05)  # ✅ 先低台阶
    p.add_argument("--step-length", type=float, default=0.25)
    p.add_argument("--width", type=float, default=3.0)
    p.add_argument("--start-x", type=float, default=1.0)
    p.add_argument("--friction", type=str, default="1.0 0.1 0.1")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_xml = find_base_hopper_xml()
    tree = ET.parse(base_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> found in xml.")

    add_stairs(worldbody, args.n_steps, args.step_height, args.step_length,
               args.width, args.start_x, args.friction)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote stairs xml: {out_path}")

if __name__ == "__main__":
    main()