
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch
from omegaconf import DictConfig

class FlattenTime(ModuleAttrMixin):
    """Flattens time dim into feature dim."""
    def __init__(self, input_shape: DictConfig, time_dim: int):
        super().__init__()
        self.input_dim = self.get_input_dim(input_shape)
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            return torch.cat([val.view(val.size(0), -1) for val in x.values()], dim=1)
        else: 
            return x.view(x.size(0), -1)

    @torch.no_grad()
    def output_shape(self):
        return (self.input_dim * self.time_dim,)

    def get_input_dim(self, shape_meta):
        total = 0
        if isinstance(shape_meta, DictConfig):
            if "shape" in shape_meta:
                total += sum(shape_meta["shape"])
            else:
                for item in shape_meta.values():
                    total += self.get_input_dim(item)

        return total
