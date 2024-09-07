from diffusion_policy.common.replay_buffer import ReplayBuffer
import shutil

replay_buffer = ReplayBuffer.create_from_path(
    zarr_path="~/Projects/DiffusionPolicy/diffusion_policy/data/testWam/replay_buffer.zarr", mode='r')

new_buffer = ReplayBuffer.create_empty_zarr()

ends = replay_buffer.episode_ends[:].copy()

for i in range(2, len(ends)):
    episode_data = replay_buffer.get_episode(i)
    new_buffer.add_episode(episode_data)

print(new_buffer.episode_ends[:])
new_buffer.save_to_path("data/cleanWam/replay_buffer.zarr", chunk_length=-1)


for i in range(2, len(ends)):
    shutil.copytree(f"/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/testWam/videos/{i}", f"/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/cleanWam/videos/{i-2}")
    episode_data = replay_buffer.get_episode(i)
    new_buffer.add_episode(episode_data)
