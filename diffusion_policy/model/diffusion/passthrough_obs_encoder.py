from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from omegaconf import DictConfig
import torch

class PassThrough(ModuleAttrMixin):
    """Passes the input through unchanged."""
    def __init__(self, input_shape: DictConfig):
        self.obs_dim = self.get_input_dim(input_shape)
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            return torch.cat([val for val in x.values()], dim=2)
        else:
            return x

    @torch.no_grad()
    def output_shape(self):
        return (self.obs_dim,)

    def get_input_dim(self, shape_meta):
        total = 0
        if isinstance(shape_meta, DictConfig):
            if "shape" in shape_meta:
                total += sum(shape_meta["shape"])
            else:
                for item in shape_meta.values():
                    total += self.get_input_dim(item)

        return total

