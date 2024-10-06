from typing import Dict
from diffusion_policy.policy.base_policy import BasePolicy
from abc import ABC, abstractmethod

class BaseRunner(ABC):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @abstractmethod
    def run(self, policy: BasePolicy) -> Dict:
        pass

    @abstractmethod
    def close(self):
        pass
