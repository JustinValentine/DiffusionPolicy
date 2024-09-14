from diffusion_policy.common.replay_buffer import ReplayBuffer
import shutil
import numpy as np
import matplotlib.pyplot as plt

source_base = "/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/cleanWam"
dest_base = "/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/bhandPosWam"

replay_buffer = ReplayBuffer.create_from_path(
    zarr_path=f"{source_base}/replay_buffer.zarr", mode='r')

new_buffer = ReplayBuffer.create_empty_zarr()

ends = replay_buffer.episode_ends[:].copy()

for i in range(len(ends)):
    episode_data = replay_buffer.get_episode(i)
    new_episode_data = episode_data.copy()
    new_episode_data['action'] = np.concatenate([episode_data['action'][:, :7], episode_data['hand_qpos'][:, [0, 3]]], axis=1)
    new_buffer.add_episode(new_episode_data)

new_buffer.save_to_path(f"{dest_base}/replay_buffer.zarr", chunk_length=-1)

for i in range(len(ends)):
    shutil.copytree(f"{source_base}/videos/{i}", f"{dest_base}/videos/{i}")
