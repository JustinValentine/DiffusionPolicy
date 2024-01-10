import sys
import os
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from diffusion_policy.env_runner.real_pick_lowdim_runner import RealPickLowdimRunner

def test():
    from omegaconf import OmegaConf
    cfg_path = ROOT_DIR / "diffusion_policy/config/task/real_pick_lowdim.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    cfg['n_obs_steps'] = 1
    cfg['n_action_steps'] = 1
    cfg['past_action_visible'] = False
    runner_cfg = cfg['env_runner']
    runner_cfg['n_train'] = 1
    runner_cfg['n_test'] = 0
    del runner_cfg['_target_']
    runner = RealPickLowdimRunner(
        **runner_cfg, 
        output_dir='/tmp/test')

    # import pdb; pdb.set_trace()

    self = runner
    env = self.env
    env.seed(seeds=self.env_seeds)
    obs = env.reset()
    print(obs)
    obs = env.reset()
    print(obs)

if __name__ == '__main__':
    test()
