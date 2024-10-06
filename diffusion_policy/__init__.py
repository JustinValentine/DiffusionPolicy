from hydra.core.config_store import ConfigStore
from diffusion_policy.model.diffusion.ema_model import EMAConfig

cs = ConfigStore.instance()
cs.store(group="ema", name="ema", node=EMAConfig)
