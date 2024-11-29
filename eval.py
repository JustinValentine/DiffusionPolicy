"""
Use:
python3 eval.py --checkpoint "data/outputs/2024.11.25/17.53.20_flow_matching_doodle/checkpoints/epoch_7250.ckpt" -o /tmp --save_traj
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

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--save_traj', is_flag=True, help='Save trajectories to a file')
def main(checkpoint, output_dir, device, save_traj):
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

    with open('./cnn/data_files/data_index.json', 'r') as f:
        indexes = json.load(f)
         
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

    random.seed(10)
    num_samples = 30
    query_size = 5
    for _ in tqdm(range(num_samples // query_size)):
        query = random.sample(range(1, 26), query_size)

        obs_dict = {
            "obs": {
                "class_quat": torch.tensor(query), 
            }
        }

        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(1).to(policy.device))

        collector.reset()

        with torch.no_grad():
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
    df.columns = ['word', 'drawing']
    df.to_csv(f'./cnn/data_files/action_data.csv', index=False)

    # If save_traj is True, save trajectories to CSV file
    if save_traj:
        traj_eval.seek(0)
        traj_df = pd.read_csv(traj_eval, header=None)
        traj_df.columns = ['word', 'trajectory']
        traj_df.to_csv(f'./cnn/data_files/traj_data.csv', index=False)

if __name__ == '__main__':
    main()
