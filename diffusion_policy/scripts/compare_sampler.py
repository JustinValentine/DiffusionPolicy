from diffusion_policy.noise_schedulers import LinearNoiseScheduler, ExponentialNoiseScheduler
import numpy as np
import click
from pathlib import Path
import torch
import dill
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-w', '--workspace_dir', required=True)
@click.option('-c', '--ckpt', default="latest.ckpt")
def main(workspace_dir, ckpt):
    checkpoint_dir = Path(workspace_dir) / "checkpoints"

    checkpoint = checkpoint_dir / ckpt

    assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist"

    with open(checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=workspace_dir)
    workspace.load_payload(payload, exclude_keys=['_output_dir'])

    step_list = [1, 2, 4, 8, 16, 32, 64, 100]
    results = []
    policy = workspace.ema.get()
    if isinstance(policy, list):
        policy = policy[0]
    policy.to(torch.device('cuda:0'))
    policy.eval()

    env_runner = hydra.utils.instantiate(
        workspace.cfg.task.env_runner,
        output_dir=checkpoint_dir)
    for n_steps in step_list:

        policy.num_inference_steps = n_steps
        res = env_runner.run(policy)
        results.append(res['test/mean_score'])
        print(f"n_steps: {n_steps}, score: {res['test/mean_score']}")

    print(results)

    results = np.array([step_list, results]).T
    np.save(workspace_dir + "/linear_sample.npy", results)


if __name__ == "__main__":
    main()

