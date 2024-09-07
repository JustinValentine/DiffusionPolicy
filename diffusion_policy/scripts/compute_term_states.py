import os
import click
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np

@click.command()
@click.option('--input', '-i',  required=True)
def main(input):
    input = pathlib.Path(os.path.expanduser(input))
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    assert in_zarr_path.is_dir()

    replay_buffer = ReplayBuffer.copy_from_path(in_zarr_path)

    last_states = []
    n_episodes = replay_buffer.n_episodes
    for i in range(n_episodes):

        episode = replay_buffer.get_episode(i)
        last_states.append(episode['robot_qpos'][-1])

    last_states = np.array(last_states)
    print(repr(last_states.mean(axis=0)))
    

if __name__ == '__main__':
    main()
