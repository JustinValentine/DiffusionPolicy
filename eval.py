"""
Use:

Best Flow_matching model: 25 classes
python3 eval.py --checkpoint "/home/odin/DiffusionPolicy/data/outputs/2024.11.25/17.53.20_flow_matching_doodle/checkpoints/epoch_9750.ckpt" -o /tmp --save_traj

Flow Matching: 30 classes 
python3 eval.py --checkpoint "/home/odin/DiffusionPolicy/data/outputs/2024.12.02/16.30.32_flow_matching_doodle/checkpoints/epoch_1000.ckpt" -o /tmp 

Flow Matching + guidance: 30 classes - Only 32 samples each
python3 eval.py --checkpoint "/home/odin/DiffusionPolicy/data/outputs/2024.12.03/03.41.09_flow_matching_doodle/checkpoints/epoch_3500.ckpt" -o /tmp 

Flow Matching + guidance: 30 classes - Only 128 samples each
python3 eval.py --checkpoint "/home/odin/DiffusionPolicy/data/outputs/2024.12.03/05.36.44_flow_matching_doodle/checkpoints/epoch_3500.ckpt" -o /tmp 

Recent Model
python3 eval.py --checkpoint "/home/odin/DiffusionPolicy/data/outputs/2024.12.02/16.30.32_flow_matching_doodle/checkpoints/latest.ckpt" -o /tmp

Odin: Effects of W,P experiment P=0.25 
/home/odin/DiffusionPolicy/data/outputs/2024.12.13/13.41.08_flow_matching_doodle/checkpoints/epoch_2500.ckpt

Odin: Effects of W,P experiment P=0.01
/home/odin/DiffusionPolicy/data/outputs/2024.12.14/17.50.54_flow_matching_doodle/checkpoints/latest.ckpt

"""

import sys
from omegaconf import OmegaConf, open_dict
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
from io import StringIO
from csv import writer
import pandas as pd
from tqdm import tqdm
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from torch.utils.data import DataLoader, RandomSampler

extra = "_uncond"
output_file = f'./cnn/data_files/generated_data{extra}.csv'
index_file = f'./cnn/data_files/data{extra}_index.json'

class Collector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vel = []
        self.traj = []
        self.t = []

    def add(self, vel, traj, t):
        self.vel.append(vel.detach().to("cpu").numpy())
        self.traj.append(traj.detach().to("cpu").numpy())
        self.t.append(t.detach().to("cpu").numpy())


def generate(checkpoint, output_dir, save_traj, device='cuda:0', generation_file=None, num_samples=100, w=1):
    #if os.path.exists(output_dir):
    #    click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)

    # sample = train_dataloader['action']

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema.get()
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    with open(index_file, 'r') as f:
        indexes = json.load(f)

    # with open('./data/doodle/20_hot_class_index.json', 'r') as f:
    #     indexes = json.load(f)
         
    # Open the csv writer for actions
    eval = StringIO()
    csv_writer = writer(eval)
    # csv_writer.writerow(["word", "drawing"])

    # If save_traj is True, prepare csv writer for trajectories
    if save_traj:
        traj_eval = StringIO()
        traj_csv_writer = writer(traj_eval)
        # traj_csv_writer.writerow(["word", "trajectory"])

    collector = Collector()

    file_num = 4
    random.seed(10)
    query_size = 100
    manual = True
    class_num = 20
    for class_num in tqdm(range(26), desc="class"):
        for _ in tqdm(range(num_samples // query_size)):
            if not manual:
                query = random.sample(range(1, 26), query_size)
            else:
                query = [class_num] * query_size

            # Conditioned observations
            obs_dict = {
                "obs": {
                    "class_quat": torch.tensor(query), 
                }
            }

            obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(1).to(policy.device))

            collector.reset()

            with torch.no_grad():
                if type(workspace.model).__name__ == 'ConditionalFlowMatchingPolicy':
                    gen_doodle = policy.predict_action(obs_dict, collector, w=w)
                else:
                    gen_doodle = policy.predict_action(obs_dict, collector)

            # Process and save actions
            action_tensor = gen_doodle['action']  # Extract the action tensor
            action_tensor = action_tensor.cpu().numpy()  # Move to CPU and convert to NumPy (if on CUDA)

            # Save actions to csv
            for i, action in enumerate(action_tensor):
                data = action.tolist()
                csv_writer.writerow([list(indexes.keys())[query[i]], str(data)])

            # If save_traj is True, process and save trajectories
            if save_traj:
                trajs = np.array(collector.traj)  # Collector's trajs
                # trajs shape: (time_steps, batch_size, traj_dim)

                # Stack over time steps
                trajs = np.stack(collector.traj)  # Shape: (time_steps, batch_size, traj_dim)

                # For each sample in the batch, collect its trajectory over time
                batch_size = trajs.shape[1]
                for idx in range(batch_size):
                    traj_data = trajs[:, idx, :].tolist()  # Trajectory data for one sample
                    traj_csv_writer.writerow([list(indexes.keys())[query[idx]], str(traj_data)])

    # Save actions to CSV file
    eval.seek(0)  # Reset StringIO pointer to the beginning
    df = pd.read_csv(eval, header=None)
    # df.columns = ['word', 'drawing']
    if generation_file == None:
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(generation_file, index=False)

    # If save_traj is True, save trajectories to CSV file
    if save_traj:
        traj_eval.seek(0)
        traj_df = pd.read_csv(traj_eval, header=None)
        traj_df.columns = ['word', 'trajectory']
        traj_df.to_csv(f'./cnn/data_files/traj_data_{file_num}_good_figs.csv', index=False)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--save_traj', is_flag=False, help='Save trajectories to a file')
def main(checkpoint, output_dir, device, save_traj):
    generate(checkpoint, output_dir, save_traj, device=device)

if __name__ == '__main__':
    main()
