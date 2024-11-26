from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import DictConfig
import hydra

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DictConfig,
        obs_encoder: DictConfig,
        inner_model: DictConfig,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.noise_scheduler = hydra.utils.instantiate(noise_scheduler)
        # get feature dim
        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        obs_feature_dim = self.obs_encoder.output_shape()[0]

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
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        global_cond: torch.Tensor,
        generator=None,
        collector=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        # get batch size
        B = global_cond.shape[0]

        trajectory = torch.randn(
            size=(B, self.horizon, self.action_dim),
            device=self.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:

            model_output = model(
                trajectory, t, global_cond=global_cond
            )

            # compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

            if collector:
                collector.add(model_output, trajectory, t)

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], collector=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert "obs" in obs_dict

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)["obs"]

        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...])
        global_cond = self.obs_encoder(this_nobs)

        # run sampling
        nsample = self.conditional_sample(
            global_cond=global_cond,
            collector=collector,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = self.n_obs_steps - 1
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

        # handle different ways of passing observation
        trajectory = nactions

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...])
        global_cond = self.obs_encoder(this_nobs)


        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
