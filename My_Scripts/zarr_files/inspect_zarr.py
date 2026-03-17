import zarr
import numpy as np

path = "/home/medeiros/Desktop/Projects/diffusion_policy_diogo/data/datasets/block_pushing/multimodal_push_seed.zarr"

z = zarr.open(path, mode="r")

ends = z["meta/episode_ends"][:]
print("\nTotal number of episodes:", len(ends))
print("Total number of timesteps:", ends[-1])

ep_actions = z["data/action"][:ends[0]]
print("\nEpisode steps:", len(ep_actions))

def ep_slice(i):
    start = 0 if i == 0 else ends[i-1]
    end = ends[i]
    return slice(start, end)

ep0 = ep_slice(0)

acts = z["data/action"][ep0]
obs  = z["data/obs"][ep0]
print("\nEpisode 0: actions", acts.shape, "obs", obs.shape)
print("\nFirst 5 actions:\n", acts[:5])
print("\nFirst 5 observations:\n", obs[:5])
print("\nFirst 5 episode timesteps:\n", ends[:5])
print("\n")