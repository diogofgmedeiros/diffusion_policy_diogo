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
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal

zarr_path = ROOT_DIR / "data" / "training" / "block_pushing" / "multimodal_push_seed_new.zarr"
z = zarr.open(str(zarr_path), mode='r')

actions = np.array(z['data']['action'])
episode_ends = np.array(z['meta']['episode_ends'])
initial_states = z['initial_states']
print(f"Dataset carregado: {len(episode_ends)} episódios, {actions.shape[0]} ações")

output_dir = ROOT_DIR / "dataset_videos_new3"
output_dir.mkdir(parents=True, exist_ok=True)

env = BlockPushMultimodal(shared_memory=False)

N_EPISODES = 10

start = 0
for ep in range(N_EPISODES):
    end = episode_ends[ep]
    print(f"Episode {ep} ({end - start} steps)")

    env.seed(ep)
    initial_states = z['initial_states']
    env.reset(reset_poses=False)
    state = pickle.loads(initial_states[ep])
    env.set_pybullet_state(state)

    frames = []
    for step, a in enumerate(actions[start:end]):
        obs, reward, done, info = env.step(a)
        frame = env.render()
        frames.append(frame)
        if done:
            break

    video_path = output_dir / f"episode_{ep:03d}.mp4"
    imageio.mimsave(video_path, frames, fps=10)
    print(f"Saved: {video_path}")

    start = end

print("\nAll done!")