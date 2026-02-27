#python My_Scripts/visualize_dataset_videos.py \
#  --zarr_path data/datasets/block_pushing/single_push_seed.zarr \
#  --env_type single \
#  --output_dir data/datasets/block_pushing/videos_single

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
import click
from diffusion_policy.env.block_pushing.block_pushing import BlockPush
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal

@click.command()
@click.option('--zarr_path', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), default=ROOT_DIR / "dataset_videos")
@click.option('--n_episodes', type=int, default=10)
@click.option(
    '--env_type',
    type=click.Choice(['single', 'multimodal'], case_sensitive=False),
    default='multimodal',
    show_default=True,
)
def main(zarr_path, output_dir, n_episodes, env_type):
    z = zarr.open(str(zarr_path), mode='r')

    actions = np.array(z['data']['action'])
    episode_ends = np.array(z['meta']['episode_ends'])
    initial_states = z['initial_states']
    print(f"Uploaded dataset: {len(episode_ends)} episodes, {actions.shape[0]} actions")

    output_dir.mkdir(parents=True, exist_ok=True)

    if env_type == 'single':
        env = BlockPush(shared_memory=False)
    else:
        env = BlockPushMultimodal(shared_memory=False)

    n_episodes = min(n_episodes, len(episode_ends))
    start = 0
    for ep in range(n_episodes):
        end = episode_ends[ep]
        print(f"Episode {ep} ({end - start} steps)")

        env.seed(ep)
        env.reset(reset_poses=False)
        state = pickle.loads(initial_states[ep])
        env.set_pybullet_state(state)

        frames = []
        for _, a in enumerate(actions[start:end]):
            _, _, done, _ = env.step(a)
            frame = env.render()
            frames.append(frame)
            if done:
                break

        video_path = output_dir / f"episode_{ep:03d}.mp4"
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved: {video_path}")
        start = end

    print("\nAll done!")

if __name__ == '__main__':
    main()