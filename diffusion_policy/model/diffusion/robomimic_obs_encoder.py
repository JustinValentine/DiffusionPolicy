from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch.nn as nn
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo.algo import PolicyAlgo
from robomimic.algo import algo_factory
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.model.vision.crop_randomizer as dmvc


class RobomimicObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        crop_shape=(76, 76),
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        n_obs_steps=2,
        flatten_time=True,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        self.low_dim_keys = obs_config["low_dim"]
        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, dmvc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )
        self.obs_encoder = obs_encoder
        self.n_obs_steps = n_obs_steps
        self.flatten_time = flatten_time

    def forward(self, obs):
        B, T, _ = obs[self.low_dim_keys[0]].shape
        obs = dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))
        result = self.obs_encoder(obs)
        if self.flatten_time:
            result = result.reshape(B, -1)
        else:
            result = result.reshape(B, T, -1)
        return result

    def output_shape(self):
        if self.flatten_time:
            return (self.obs_encoder.output_shape()[0] * self.n_obs_steps,)
        else:
            return self.obs_encoder.output_shape()
