
from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt

epsisode = 19
num_actions = 15
# source_base = "/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/bhandPosWam"
source_base = "~/Projects/DiffusionPolicy/testEval"

replay_buffer = ReplayBuffer.create_from_path(
    zarr_path=f"{source_base}/replay_buffer.zarr", mode='r')

ends = replay_buffer.episode_ends[:].copy()
print(len(ends))

episode_data = replay_buffer.get_episode(epsisode)
n_plots = episode_data['action'].shape[1]
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 10))
cmds = np.load("/home/dylanmiller/Projects/DiffusionPolicy/cmd_send.npy", allow_pickle=True)
times = np.load("/home/dylanmiller/Projects/DiffusionPolicy/time_send.npy", allow_pickle=True)
for j in range(n_plots):

    axs[j].plot(episode_data['action_timestamp'], episode_data['action'][:, j], '.-', label='action')
    axs[j].plot(episode_data['action_timestamp'][::num_actions], episode_data['action'][::num_actions, j], 'r.', label='action')
    axs[j].plot(times, cmds[:, j], 'g.')
    if j == 7:
        axs[j].plot(episode_data['timestamp'], episode_data['hand_qpos'][:, 0], '.', label='robot_qpos')
    elif j == 8:
        axs[j].plot(episode_data['timestamp'], episode_data['hand_qpos'][:, 4], '.', label='robot_qpos')
    else:
        axs[j].plot(episode_data['timestamp'], episode_data['robot_qpos'][:, j], '.', label='robot_qpos')
plt.show()

