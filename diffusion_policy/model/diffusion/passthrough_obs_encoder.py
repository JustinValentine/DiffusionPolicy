from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch

class PassThrough(ModuleAttrMixin):
    """Passes the input through unchanged."""
    def __init__(self, input_dim: int):
        self.obs_dim = input_dim
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def output_shape(self):
        return (self.obs_dim,)
