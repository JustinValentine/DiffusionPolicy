
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch

class FlattenTime(ModuleAttrMixin):
    """Passes the input through unchanged."""
    def __init__(self, input_dim: int, time_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        return x.view(x.size(0), -1)

    @torch.no_grad()
    def output_shape(self):
        return (self.input_dim * self.time_dim,)
