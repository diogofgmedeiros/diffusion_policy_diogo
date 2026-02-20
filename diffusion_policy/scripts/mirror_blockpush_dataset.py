if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import click
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

# obs layout for BlockPushMultimodal lowdim:
# [0] block_x, [1] block_y, [2] block_yaw,
# [3] block2_x, [4] block2_y, [5] block2_yaw,
# [6] ee_x, [7] ee_y,
# [8] ee_target_x, [9] ee_target_y,
# [10] target_x, [11] target_y, [12] target_yaw,
# [13] target2_x, [14] target2_y, [15] target2_yaw
Y_IDX = np.array([1, 4, 7, 9, 11, 14], dtype=np.int64)
YAW_IDX = np.array([2, 5, 12, 15], dtype=np.int64)


@click.command()
@click.option('-i', '--input', required=True, help='input zarr path')
@click.option('-o', '--output', required=True, help='output mirrored zarr path')
@click.option('--abs_action', is_flag=True, default=False,
    help='Set if action stores absolute EE target (still mirrored on y-axis).')
def main(input, output, abs_action):
    src = ReplayBuffer.copy_from_path(input, keys=['obs', 'action'])

    obs_all = src['obs'].astype(np.float32).copy()
    action_all = src['action'].astype(np.float32).copy()
    ends = src.episode_ends[:]

    # Mirror horizontal axis (y -> -y)
    obs_all[:, Y_IDX] *= -1.0

    # Mirror yaw under y-axis reflection.
    # state stores yaw in [0, pi), so use modulo-pi normalization.
    obs_all[:, YAW_IDX] = np.mod(-obs_all[:, YAW_IDX], np.pi)

    # Action is 2D XY; mirror y component.
    # For both delta and absolute action representations the y component must flip.
    action_all[:, 1] *= -1.0

    dst = ReplayBuffer.create_empty_numpy()
    start = 0
    for end in ends:
        end = int(end)
        episode = {
            'obs': obs_all[start:end],
            'action': action_all[start:end],
        }
        dst.add_episode(episode)
        start = end

    dst.save_to_path(output, chunk_length=-1)

    mode = 'absolute' if abs_action else 'delta'
    print(f"Saved mirrored dataset to {output} (action mode: {mode})")


if __name__ == '__main__':
    main()