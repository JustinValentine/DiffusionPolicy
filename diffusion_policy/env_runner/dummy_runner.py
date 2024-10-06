from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.env_runner.base_runner import BaseRunner

class DummyRunner(BaseRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BasePolicy):
        return dict()

    def close(self):
        pass
