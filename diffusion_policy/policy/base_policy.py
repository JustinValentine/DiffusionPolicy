from typing import Dict
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer
from omegaconf import DictConfig
import hydra

class BasePolicy(ModuleAttrMixin, ABC):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    @abstractmethod
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        pass

    @abstractmethod
    def compute_loss(self, batch):
        pass

    # reset state for stateful policies
    def reset(self):
        pass

    def get_optimizer(self, cfg: DictConfig):
        if "_target_" in cfg:
            return hydra.utils.instantiate(
                cfg, params=self.parameters())
        else:

            optim_groups = self.model.get_optim_groups(
                weight_decay=cfg.model_weight_decay
            )
            optim_groups.append(
                {
                    "params": self.obs_encoder.parameters(),
                    "weight_decay": cfg.obs_encoder_weight_decay,
                }
            )
            optimizer = torch.optim.AdamW(
                optim_groups, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps
            )
            
            return optimizer

    # ========== training ===========
    # no standard training interface except setting normalizer
    @abstractmethod
    def set_normalizer(self, normalizer: LinearNormalizer):
        pass
