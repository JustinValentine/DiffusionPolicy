if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.wam_pick.utils import WAM_fwd_kinematics_mat
from scipy.spatial.transform import Rotation as R



class Trajectory:
    def __init__(self, time, position, velocity, effort, xposition, orientation, z):
        self.time = time
        self.position = position
        self.velocity = velocity
        self.effort = effort
        self.xposition = xposition
        self.orientation = orientation
        self.z = z
        self.n = self.time.shape[0]
        self.dof = self.position.shape[1]
        self.T = time.shape[0] * 0.002 # estimate time based on 500Hz freq

    @staticmethod
    def from_json(joint_json):
        time = np.array([d['time'] for d in joint_json])
        position = np.array([d['position'] for d in joint_json])
        velocity = np.array([d['velocity'] for d in joint_json])
        effort = np.array([d['effort'] for d in joint_json])

        T_matrix = np.apply_along_axis(lambda x: WAM_fwd_kinematics_mat(x, 7), axis=1, arr=position) 
        xposition = T_matrix[:, :3, -1]
        r = R.from_matrix(T_matrix[:, :3, :3])
        orientation = r.as_quat()
        #z = (time - time.min()) / (time.max() - time.min())
        z = np.linspace(0, 1, time.shape[0])
        return Trajectory(time, position, velocity, effort, xposition, orientation, z)

@click.command()
@click.option('-i', '--input', default="/Users/dylanmiller/Projects/ProMPs/data/Pick and Place - Second attempt Jun 13", help='input dir contains json files')
@click.option('-o', '--output', default="/Users/dylanmiller/Projects/DiffusionPolicy/diffusion_policy/data/pick", help='output zarr path')
@click.option('--abs_action', is_flag=True, default=False)
def main(input, output, abs_action):

    data_directory = pathlib.Path(input)
    traj_dir = data_directory / "Trjs"

    with open(traj_dir.parent / "pick_points.json") as f:
        pick_points = json.load(f)


    h_transform = np.array([
        [ 5.80416032e-03,  9.86542830e-05, -1.41281154e+00],
        [ 3.49402610e-04,  3.47946765e-03, -5.78290436e-01],
        [ 7.80496521e-04,  8.63413816e-05,  1.00000000e+00]
    ])
    pick_data = []
    initial_qpos = []
    initial_qvel = []
    initial_pick_points = []
    n_demonstrations = 30
    for i in range(n_demonstrations):
        with open(traj_dir / f"DMP_data_joint_pick_{i}.json") as f:
            traj_dict = json.load(f)
        
        traj = Trajectory.from_json(traj_dict)
        pick_data.append(traj)
        initial_qpos.append(traj.position[0, :])
        initial_qvel.append(traj.velocity[0, :])


    buffer = ReplayBuffer.create_empty_numpy()

    for i, traj in enumerate(pick_data):
        down_sample_mask = np.arange(traj.position.shape[0]) % 10 == 0
        xpos = traj.xposition[down_sample_mask]
        orientation = traj.orientation[down_sample_mask]

        pick_point_hom = np.array([[pick_points[i][0], pick_points[i][1], 1]])
        pick_trans = h_transform @ pick_point_hom.T
        pick_point_norm = (pick_trans / pick_trans[-1]).T
        initial_pick_points.append(pick_point_norm[:, :2])
        pick_location = np.tile(pick_point_norm[:, :2], (xpos.shape[0], 1))
        obs = np.concatenate([xpos, orientation], axis=1)

        data = {                              
            'obs': obs,
            'position': traj.position[down_sample_mask],
            'velocity': traj.velocity[down_sample_mask],
            'pick_location': pick_location
        }
        buffer.add_episode(data)
    output_path = pathlib.Path(output)
    buffer.save_to_path(zarr_path=output, chunk_length=-1)
    with open(output_path / "all_init_qpos.npy", 'wb') as f:
        qpos_array = np.array(initial_qpos)
        np.save(f, qpos_array)

    with open(output_path / "all_init_qvel.npy", 'wb') as f:
        qvel_array = np.array(initial_qvel)
        np.save(f, qvel_array)

    with open(output_path / "all_init_pick_loc.npy", 'wb') as f:
        loc_array = np.concatenate(initial_pick_points)
        np.save(f, loc_array)


if __name__ == '__main__':
    main()
