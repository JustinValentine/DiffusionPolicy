"""
Usage:
python eval.py --checkpoint /home/odin/DiffusionPolicy/data/outputs/2024.11.21/01.00.15_doodle_square_image/checkpoints/epoch_25.ckpt -o data/pusht_eval_output
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
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply

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
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    obs_dict = {
        "obs": {
            "class_quat": torch.eye(10)[1],  
            "on_paper_quat": torch.zeros(2),  
            "termination_quat": torch.tensor([0])  
        }
    }

    obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).unsqueeze(0).to(device))
    # obs_dict = dict_apply(obs_dict, lambda x: x.to(device))

    with torch.no_grad():
        gen_doodle = policy.predict_action(obs_dict)

    print(gen_doodle)


if __name__ == '__main__':
    main()
