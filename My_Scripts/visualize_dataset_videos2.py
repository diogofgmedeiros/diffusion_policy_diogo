import sys
import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

#import time
import numpy as np
import pickle
import zarr
import imageio
import argparse
from datetime import datetime
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BlockPush dataset episodes as videos."
    )
    parser.add_argument(
        "--zarr_path",
        type=pathlib.Path,
        required=True,
        help="Path to dataset zarr file",
    )
    parser.add_argument(
        "--output_root",
        type=pathlib.Path,
        default=ROOT_DIR / "data" / "dataset_videos",
        help="Root directory where one run folder will be created",
    )
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--abs_action",
        action="store_true",
        help="Set if dataset action represents absolute ee target",
    )
    parser.add_argument(
        "--seed_on_reset",
        action="store_true",
        help="Seed env with episode index when initial_states are missing",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    zarr_path = args.zarr_path
    z = zarr.open(str(zarr_path), mode="r")

    actions = np.array(z["data"]["action"])
    episode_ends = np.array(z["meta"]["episode_ends"])
    has_initial_states = "initial_states" in z
    initial_states = z["initial_states"] if has_initial_states else None

    print(f"Dataset carregado: {len(episode_ends)} episódios, {actions.shape[0]} ações")
    print(f"initial_states presente: {has_initial_states}")

    run_name = f"{zarr_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Videos vão ser guardados em: {output_dir}")

    env = BlockPushMultimodal(shared_memory=False, abs_action=args.abs_action)
    n_episodes = min(args.n_episodes, len(episode_ends))

    start = 0
    for ep in range(n_episodes):
        end = int(episode_ends[ep])
        print(f"Episode {ep} ({end - start} steps)")

        if has_initial_states:
            env.reset()
            state = pickle.loads(initial_states[ep])
            env.set_pybullet_state(state)
        else:
            if args.seed_on_reset:
                env.seed(ep)
            env.reset()

        frames = []
        for a in actions[start:end]:
            _, _, done, _ = env.step(a)
            frame = env.render()
            frames.append(frame)
            if done:
                break

        video_path = output_dir / f"episode_{ep:03d}.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps)
        print(f"Saved: {video_path}")

        start = end

    print("\nAll done!")


if __name__ == "__main__":
    main()