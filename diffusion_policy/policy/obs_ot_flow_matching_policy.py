from typing import Dict

import hydra
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import DictConfig
from torchcfm.optimal_transport import OTPlanSampler
from torchdyn.core import NeuralODE

from diffusion_policy.common.distributions import Distribution
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_policy import BasePolicy


class OTFlowMatchingPolicy(BasePolicy):
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
        self.sampler = hydra.utils.instantiate(sampler)
        self.time_distribution = hydra.utils.instantiate(time_distribution)
        # get feature dim

        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        obs_feature_dim = self.obs_encoder.output_shape()[0]

        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=1, normalize_cost=True)

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
        self.sigma_min = target_sigma_min
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        self.eps = 1e-5

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
        x0=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        batch_size = condition_data.shape[0]

        # trajectory = torch.randn(
        #     size=condition_data.shape,
        #     dtype=condition_data.dtype,
        #     device=condition_data.device,
        #     generator=generator,
        # )

        trajectory = x0

        node = NeuralODE(
            torch_wrapper(self.model, local_cond=local_cond, global_cond=global_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        trajectory = node.trajectory(trajectory, t_span=torch.linspace(0, 1, 2))
        trajectory = trajectory[-1]

        # times = self.noise_scheduler.get_sigmas(self.num_inference_steps).flip(dims=[0]) # reverse so time starts at 0.
        #
        # for t in range(self.num_inference_steps - 1):
        #     t0 = times[t]
        #     t_batch = t0 * torch.ones(batch_size, device=condition_data.device)
        #     pred = self.model(trajectory, t_batch, local_cond=local_cond, global_cond=global_cond)
        #     dt = times[t + 1] - t0
        #     trajectory = trajectory + pred * dt
        # trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet
        n_obs_action = self.get_obs_action(obs_dict)
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
            x0=n_obs_action,
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

    def psi_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        mu_t = t * x_1 + (1 - t) * x_0
        epsilon = torch.randn_like(x_0)
        return mu_t + self.sigma_min * epsilon

    def get_obs_action(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch['obs']['robot0_eef_pos'].shape[0]
        rt = RotationTransformer(
            from_rep='quaternion', to_rep='rotation_6d')
        pos = batch["obs"]["robot0_eef_pos"][:, -1]
        rot = rt.forward(batch["obs"]["robot0_eef_quat"][:, -1])
        pos = pos.unsqueeze(1)
        rot = rot.unsqueeze(1)
        gripper = torch.rand(size=(batch_size, 1, 1), device=self.device)*2 - 1 # -1 or 1
        obs_action = {"action": torch.cat([pos, rot, gripper], dim=-1)}
        n_obs_action = self.normalizer.normalize(obs_action)["action"]
        n_obs_action = n_obs_action.expand((batch_size, self.horizon, -1))
        return n_obs_action


    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        
        n_obs_action = self.get_obs_action(batch)

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
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...])
            global_cond = self.obs_encoder(this_nobs)
        else:
            # reshape B, T, ... to B*T
            # TODO: this will not work with the reshape, same goes for local_cond
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Ddsfasdfo
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting masksdjfasdf
        condition_mask = self.mask_generator(trajectory.shape)

        # t = self.time_distribution.sample(batch_size)
        t = torch.rand(batch_size, device=self.device)
        t = t[:, None, None].expand(nactions.shape)

        # x_0 = torch.randn_like(nactions)

        x_0 = n_obs_action # use last observation
        x_1 = nactions

        # x_0, x_1, _, global_cond = self.ot_sampler.sample_plan_with_labels(
        #     x_0, nactions, y1=global_cond
        # )

        # compute loss mask
        loss_mask = ~condition_mask

        # Predict the noise residual
        v_t = self.model(
            self.psi_t(x_0, x_1, t),
            t[:, 0, 0],
            local_cond=local_cond,
            global_cond=global_cond,
        )
        u_t = x_1 - x_0

        loss = F.mse_loss(v_t, u_t, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, global_cond, local_cond):
        super().__init__()
        self.model = model
        self.global_cond = global_cond
        self.local_cond = local_cond

    def forward(self, t, x, *args, **kwargs):
        return self.model(
            x,
            t.repeat(x.shape[0]),
            global_cond=self.global_cond,
            local_cond=self.local_cond,
        )
