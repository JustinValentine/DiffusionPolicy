from typing import Dict

import hydra
import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import DictConfig
from typing import Union

from diffusion_policy.common.distributions import Distribution
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_policy import BasePolicy

class ConditionalFlowMatchingPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DictConfig,
        obs_encoder: DictConfig,
        inner_model: DictConfig,
        sampler: DictConfig,
        time_distribution: DictConfig,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        target_sigma_min: float,
        obs_to_action: Union[DictConfig, None] = None,
        num_inference_steps=None,
        map_from_last_obs=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.noise_scheduler = hydra.utils.instantiate(noise_scheduler)
        self.sampler = hydra.utils.instantiate(sampler)
        self.time_distribution = hydra.utils.instantiate(time_distribution)
        # get feature dim

        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        obs_feature_dim = self.obs_encoder.output_shape()[0]

        if map_from_last_obs:
            self.obs_to_action = hydra.utils.instantiate(obs_to_action)

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        if hasattr(inner_model, "input_dim"):
            inner_model.input_dim = input_dim

        if hasattr(inner_model, "global_cond_dim"):
            inner_model.global_cond_dim = global_cond_dim

        self.model = hydra.utils.instantiate(inner_model)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.sigma_min = target_sigma_min
        self.map_from_last_obs = map_from_last_obs
        self.kwargs = kwargs
        self.eps = 1e-5

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        x0,
        global_cond: torch.Tensor,
        generator=None,
        collector=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        # get batch size
        B = global_cond.shape[0]

        trajectory = x0

        times = self.noise_scheduler.get_sigmas(self.num_inference_steps).flip(dims=[0]) # reverse so time starts at 0.

        for t in range(self.num_inference_steps - 1):
            t0 = times[t]
            t_batch = t0 * torch.ones(B, device=self.device)
            pred = self.model(trajectory, t_batch, global_cond=global_cond)
            dt = times[t + 1] - t0
            trajectory = trajectory + pred * dt

            if collector:
                collector.add(pred, trajectory, t0) # Justin: This is the intermedate denoising steps 

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], collector=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)["obs"]
        if isinstance(nobs, dict):
            value = next(iter(nobs.values()))
        else:
            value = nobs
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        if self.map_from_last_obs:
            x0 = self.get_obs_action(obs_dict)
        else:
            x0 = torch.randn(
                size=(B, T, Da),
                dtype=dtype,
                device=device,
            )

        # handle different ways of passing observation
            # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
        global_cond = self.obs_encoder(this_nobs)

        # run sampling
        nsample = self.conditional_sample(
            global_cond=global_cond,
            x0=x0,
            collector=collector,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_xt(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        mu_t = t * x_1 + (1 - t) * x_0
        epsilon = torch.randn_like(x_0) * (1 - t) * self.sigma_min
        x_t = mu_t
        # x_t = mu_t + epsilon
        return x_t

    def get_obs_action(self, batch: Dict[str, torch.Tensor]):

        last_obs = dict_apply(batch["obs"], lambda x: x[:, -1, :])

        obs_action = self.obs_to_action(last_obs)
        obs_action = {"action": obs_action}
        n_obs_action = self.normalizer.normalize(obs_action)["action"]
        batch_size = n_obs_action.shape[0]
        n_obs_action = n_obs_action.unsqueeze(1)
        n_obs_action = n_obs_action.expand((batch_size, self.horizon, -1))
        return n_obs_action


    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"]
        nactions = nbatch["action"]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # Justin: add prob of using null obs for ClassIfier-Free Diffusion Guidance 
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...])
        global_cond = self.obs_encoder(this_nobs)
        
        # TODO: add uniform time distribution
        # t = self.time_distribution.sample(batch_size)
        t = torch.rand(batch_size, device=self.device)
        t = t[:, None, None].expand(nactions.shape)

        if self.map_from_last_obs:
            # TODO: confirm batch is unchanged, may need clone
            x_0 = self.get_obs_action(batch)
            x_0 = x_0 + torch.randn_like(x_0) * self.sigma_min
        else:
            x_0 = torch.randn_like(nactions)

        x_1 = nactions

        x_t = self.compute_xt(x_0, x_1, t)

        v_t = self.model(
            x_t,
            t[:, 0, 0],
            global_cond=global_cond,
        )
        # u_t = (x_1 - x_t) / (1 - t + 1e-3)
        u_t = x_1 - x_0
        scale = 1 / torch.arange(1, horizon + 1, device=self.device)
        scale = scale.reshape(1, -1, 1).expand_as(v_t)
        scaled_vel = v_t * scale

        accel_loss = torch.mean((scaled_vel[:, 1:, :3] - scaled_vel[:, :-1, :3]) ** 2)


        loss = F.mse_loss(v_t, u_t, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        # loss = loss + 0.1 * accel_loss

        return loss 

