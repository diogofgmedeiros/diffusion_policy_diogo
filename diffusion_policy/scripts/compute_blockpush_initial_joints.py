#python diffusion_policy/scripts/compute_blockpush_initial_joints.py --x 0.3 --y 0.4 --rot 0,pi,0
import argparse
import math
import numpy as np
from scipy.spatial import transform

from diffusion_policy.env.block_pushing.block_pushing import BlockPush
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.3, help="EE initial x")
    parser.add_argument("--y", type=float, default=0.4, help="EE initial y")
    parser.add_argument("--z", type=float, default=None, help="EE initial z (default: env.effector_height)")
    parser.add_argument(
        "--rot",
        type=str,
        default="0,pi,0",
        help='rotvec as "rx,ry,rz" where each can be float or "pi" or "-pi"',
    )
    args = parser.parse_args()

    def parse_term(t: str) -> float:
        t = t.strip()
        if t == "pi":
            return math.pi
        if t == "-pi":
            return -math.pi
        return float(t)

    rx, ry, rz = [parse_term(v) for v in args.rot.split(",")]

    env = BlockPush(shared_memory=False)
    try:
        z = env.effector_height if args.z is None else args.z

        rot = transform.Rotation.from_rotvec([rx, ry, rz])
        pose = Pose3d(
            rotation=rot,
            translation=np.array([args.x, args.y, z], dtype=np.float64)
        )

        q = env.robot.inverse_kinematics(pose)

        print("\nTarget pose:")
        print(f"  translation = [{args.x}, {args.y}, {z}]")
        print(f"  rotvec      = [{rx}, {ry}, {rz}]")

        print("\nINITIAL_JOINT_POSITIONS = np.array([")
        for v in q.tolist():
            print(f"    {v},")
        print("])\n")

    finally:
        env.close()


if __name__ == "__main__":
    main()