
"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
from omegaconf import OmegaConf, open_dict
from diffusion_policy.common.pytorch_util import dict_apply
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
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import matplotlib.pyplot as plt
import numpy as np

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
def main(checkpoint, output_dir, device):
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
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema.get()
        if isinstance(policy, list):
            policy = policy[0]
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    policy.num_inference_steps = 20
    
    cfg.task.env_runner['n_test'] = 1
    cfg.task.env_runner['n_test_vis'] = 0
    cfg.task.env_runner['n_train'] = 0
    cfg.task.env_runner['test_start_seed'] = 49
    cfg.task.env_runner['n_envs'] = None
    # with open_dict(cfg.task.env_runner):
    #     cfg.task.env_runner['render_args'] = {"height": 256, "width": 256}
    #     cfg.task.env_runner['mode'] = 'human'
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)

    
    env_fn = env_runner.env_fns[0]

    env = env_fn()
    env.run_dill_function(env_runner.env_init_fn_dills[0])
    
    obs, info = env.reset(env_runner.env_seeds[0])

    collector = Collector()

    done = False
    while not done:
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            "obs": obs
        }
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device, dtype=torch.float32).unsqueeze(0))
        # run policy
        collector.reset()
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict, collector)

        trajs = np.array(collector.traj)
        plt.figure(figsize=(10, 6))
        dx1, dy1 = np.diff(trajs[:, 0, 0, 0]), np.diff(trajs[:, 0, 0, 1])
        plt.quiver(trajs[:-1, 0, 0, 1], trajs[:-1, 0, 0, 0], dy1, dx1, color='blue', scale=1, scale_units='xy', label='time 1', angles='xy', width=0.005)
        dx1, dy1 = np.diff(trajs[:, 0, 1, 0]), np.diff(trajs[:, 0, 1, 1])
        plt.quiver(trajs[:-1, 0, 1, 1], trajs[:-1, 0, 1, 0], dy1, dx1, color='red', scale=1, scale_units='xy', label='time 2', angles='xy', width=0.005)
        dx1, dy1 = np.diff(trajs[:, 0, -1, 0]), np.diff(trajs[:, 0, -1, 1])
        plt.quiver(trajs[:-1, 0, -1, 1], trajs[:-1, 0, -1, 0], dy1, dx1, color='green', scale=1, scale_units='xy', label='time 16', angles='xy', width=0.005)

        plt.plot(trajs[-1, 0, :, 1], trajs[-1, 0, :, 0], "-o", label='final')
        plt.axis("equal")
        plt.legend()

        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 8))
        for i in range(10):
            axes[i % 5, i // 5].plot(trajs[:, 0, 1, i], "-o", label='final')
            axes[i % 5, i // 5].set_title(f"{i}")


        plt.figure(figsize=(10, 6))
        plt.plot(trajs[:, 0, 0, -1], "-o")
        plt.plot(trajs[:, 0, -1, -1], "-ro")
        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action = np_action_dict['action']

        
        # step env
        env_action = action
        env_action = env_runner.undo_transform_action(action)

        obs, reward, terminated, truncated, info = env.step(env_action[0])
        plt.figure(figsize=(10, 6))
        plt.imshow(env.env.env.render())
        done = terminated or truncated
        plt.show()
        
    

if __name__ == '__main__':
    main()
