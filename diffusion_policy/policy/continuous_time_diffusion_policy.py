from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import DictConfig
import hydra

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.distributions import Distribution
from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply


class ContinuousTimeDiffusionPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DictConfig,
        obs_encoder: DictConfig,
        inner_model: DictConfig,
        sigma_distribution: DictConfig,
        scaling: DictConfig,
        sampler: DictConfig,
        horizon: int,
        n_action_steps: int,
        n_obs_steps :int,
        sigma_min: float,
        sigma_max: float,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.noise_scheduler = hydra.utils.instantiate(noise_scheduler)
        self.sigma_distribution: Distribution = hydra.utils.instantiate(sigma_distribution)
        self.scaling = hydra.utils.instantiate(scaling)
        self.sampler = hydra.utils.instantiate(sampler)
        # get feature dim

        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        obs_feature_dim = self.obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

        local_cond_dim = None
        if obs_as_local_cond:
            local_cond_dim = obs_feature_dim

        if hasattr(inner_model, "input_dim"):
            inner_model.input_dim = input_dim

        if hasattr(inner_model, "global_cond_dim"):
            inner_model.global_cond_dim = global_cond_dim

        if hasattr(inner_model, "local_cond_dim"):
            inner_model.local_cond_dim = local_cond_dim

        self.model = hydra.utils.instantiate(inner_model)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        sigmas = scheduler.get_sigmas(self.num_inference_steps).to(self.device)
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        ) * self.sigma_max

        denoised_action = self.sampler.sample(model=self.model, state=condition_data, action=trajectory, sigmas=sigmas, local_cond=local_cond, global_cond=global_cond, scaling=self.scaling)
        

        denoised_action[condition_mask] = condition_data[condition_mask]

        return denoised_action

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet
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

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
            global_cond = self.obs_encoder(this_nobs)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
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

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"]
        nactions = nbatch["action"]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...])
            global_cond = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            # TODO: this will not work with the reshape, same goes for local_cond
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        sigma = self.sigma_distribution.sample(shape=batch_size).to(self.device)
        dims_to_add = trajectory.ndim - sigma.ndim
        for _ in range(dims_to_add):
            sigma = sigma.unsqueeze(-1)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device) * sigma
        noisy_trajectory = trajectory + noise

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        c_skip, self.c_out, c_in, c_noise = self.scaling(sigma)

        # Predict the noise residual
        pred = self.model(
            c_in * noisy_trajectory, c_noise.reshape(-1), local_cond=local_cond, global_cond=global_cond
        )

        denoised_action = c_skip * noisy_trajectory + self.c_out * pred

        loss = F.mse_loss(denoised_action, trajectory, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
