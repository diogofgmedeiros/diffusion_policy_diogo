# python diffusion_policy/scripts/generate_bet_blockpush_init_states.py -o data/datasets/block_pushing/single_push_seed2.zarr -n 10 --env_type single
# python diffusion_policy/scripts/generate_bet_blockpush_init_states.py -o data/datasets/block_pushing/multimodal_push_seed2.zarr -n 10 --env_type multimodal
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import pickle
import zarr
import numcodecs
import click
import pathlib
import numpy as np
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories.time_step import StepType
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.env.block_pushing.block_pushing import BlockPush
from diffusion_policy.env.block_pushing.oracles.oriented_push_oracle import OrientedPushOracle
from diffusion_policy.env.block_pushing.oracles.multimodal_push_oracle import MultimodalOrientedPushOracle

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-n', '--n_episodes', default=1000)
@click.option('-c', '--chunk_length', default=-1)
@click.option(
    '--save_step_states/--no-save_step_states',
    default=True,
    show_default=True,
    help='Store pybullet state before each action for exact state-based visualization replay.'
)
@click.option(
    '--env_type',
    type=click.Choice(['single', 'multimodal'], case_sensitive=False),
    default='multimodal',
    show_default=True,
    help='Dataset type: single block+target or multimodal (2 blocks+2 targets).'
)
def main(output, n_episodes, chunk_length, save_step_states, env_type):

    buffer = ReplayBuffer.create_empty_numpy()
    #env = TimeLimit(GymWrapper(BlockPushMultimodal()), duration=350)
    if env_type == 'single':
        env = TimeLimit(GymWrapper(BlockPush()), duration=150)
        policy_cls = OrientedPushOracle
    else:
        env = TimeLimit(GymWrapper(BlockPushMultimodal()), duration=350)
        policy_cls = MultimodalOrientedPushOracle

    initial_states = list()
    step_states = list()
    for i in tqdm(range(n_episodes)):
        print(i)
        if(i == 9):
            print("ola")
        obs_history = list()
        action_history = list()

        env.seed(i)
        #policy = MultimodalOrientedPushOracle(env)
        policy = policy_cls(env)
        time_step = env.reset()
        pyb_state = env.wrapped_env().gym.get_pybullet_state()
        initial_states.append(pickle.dumps(pyb_state))
        policy_state = policy.get_initial_state(1)
        while True:
            if time_step.step_type == StepType.LAST:
                break
            if save_step_states:
                pyb_step_state = env.wrapped_env().gym.get_pybullet_state()
                step_states.append(pickle.dumps(pyb_step_state))
            action_step = policy.action(time_step, policy_state)
            obs = np.concatenate(list(time_step.observation.values()), axis=-1)
            action = action_step.action
            obs_history.append(obs)
            action_history.append(action)

            if time_step.step_type == 2:
                break

            # state = env.wrapped_env().gym.get_pybullet_state()
            time_step = env.step(action)
        obs_history = np.array(obs_history)
        action_history = np.array(action_history)

        episode = {
            'obs': obs_history,
            'action': action_history
        }
        buffer.add_episode(episode)
    
    buffer.save_to_path(output, chunk_length=chunk_length)
    z = zarr.open(output, mode='a')
    obj_codec = numcodecs.Pickle()
    arr = np.array(initial_states, dtype=object)
    if 'initial_states' in z:
        del z['initial_states']
    z.array('initial_states', arr, dtype=object, object_codec=obj_codec)

    if save_step_states:
        step_states_arr = np.array(step_states, dtype=object)
        if 'step_states' in z:
            del z['step_states']
        z.array('step_states', step_states_arr, dtype=object, object_codec=obj_codec)
        
if __name__ == '__main__':
    main()