from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch
from omegaconf import DictConfig

class LabelEmb(ModuleAttrMixin):
    """Flattens time dim into feature dim."""
    def __init__(self, time_dim: int, n_classes, n_emb):
        super().__init__()
        # self.input_dim = self.get_input_dim(input_shape)
        self.time_dim = time_dim

        self.emb = torch.nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=n_emb,
            padding_idx=n_classes-1 
        )

        # self.emb=torch.nn.Embedding(n_classes, n_emb)
        self.n_emb = n_emb 
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([val.view(val.size(0), -1) for val in x.values()], dim=1)
        x = x.to(torch.long)

        # Map null tokens (e.g., -1) to the padding index
        x = torch.where(x == -1, torch.tensor(self.n_classes-1, device=x.device), x)
        x = self.emb(x)

        return x.squeeze()

    @torch.no_grad()
    def output_shape(self):
        return (self.n_emb * self.time_dim,)

    def get_input_dim(self, shape_meta):
        total = 0
        if isinstance(shape_meta, DictConfig):
            if "shape" in shape_meta:
                total += sum(shape_meta["shape"])
            else:
                for item in shape_meta.values():
                    total += self.get_input_dim(item)

        return total
