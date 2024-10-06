from typing import Dict

import torch
import torch.nn
from abc import ABC, abstractmethod
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        # return an empty dataset by default
        return BaseDataset()

    @abstractmethod
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        pass

    @abstractmethod
    def get_all_actions(self) -> torch.Tensor:
        pass
    
    def __len__(self) -> int:
        return 0
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        pass
