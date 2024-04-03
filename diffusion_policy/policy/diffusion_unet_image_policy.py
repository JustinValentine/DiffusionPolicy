from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from functools import partial

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.k_diffusion.score_wrappers import Denoiser
import diffusion_policy.common.k_diffusion.utils as utils
from diffusion_policy.common.k_diffusion.gc_sampling import get_sigmas_exponential, sample_ddim, sample_euler_ancestral

# replace with direct imports
from diffusion_policy.common.k_diffusion.gc_sampling import *

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            rho: float,
            num_sampling_steps: int,
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            sigma_sample_density_type: str,
            sigma_sample_density_mean: float,
            sigma_sample_density_std: float,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,

            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        inner_model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        model = Denoiser(inner_model, sigma_data)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = "exponential"
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        # training sample density
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std        
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        sigmas = self.get_noise_schedule(self.num_sampling_steps, self.noise_scheduler)

        trajectory[condition_mask] = condition_data[condition_mask]
        trajectory = self.sample_loop(sigmas, trajectory, self.sampler_type, local_cond=local_cond, global_cond=global_cond)
    
        ## set step values
        #scheduler.set_timesteps(self.num_inference_steps)

        #for t in scheduler.timesteps:
        #    # 1. apply conditioning
        #    trajectory[condition_mask] = condition_data[condition_mask]

        #    # 2. predict model output
        #    model_output = model(trajectory, t, 
        #        local_cond=local_cond, global_cond=global_cond)

        #    # 3. compute previous image: x_t -> x_t-1
        #    trajectory = scheduler.step(
        #        model_output, t, trajectory, 
        #        generator=generator,
        #        **kwargs
        #        ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
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
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)


        noise = torch.randn_like(trajectory)
        sigma = self.make_sample_density()(shape=(batch_size,), device=self.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        #pred = self.model(trajectory, sigma, 
        #    local_cond=local_cond, global_cond=global_cond)
        
        loss = self.model.loss(trajectory, noise, sigma,
            local_cond=local_cond, global_cond=global_cond)


        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        """
        sd_config = []
        
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  
            scale = self.sigma_sample_density_std 
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self, 
            sigmas, 
            x_t: torch.Tensor,
            sampler_type: str,
            local_cond,
            global_cond,
            extra_args={}, 
            ):
            """
            Main method to generate samples depending on the chosen sampler type for rollouts
            """
            # get the s_churn 
            s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
            s_min = extra_args['s_min'] if 's_min' in extra_args else 0
            use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
            # extra_args.pop('s_churn', None)
            # extra_args.pop('use_scaler', None)
            keys = ['s_churn', 'keep_last_actions']
            if bool(extra_args):
                reduced_args = {x:extra_args[x] for x in keys}
            else:
                reduced_args = {}
            
            if use_scaler:
                scaler = self.scaler
            else:
                scaler=None
            # ODE deterministic
            #if sampler_type == 'lms':
            #    x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
            ## ODE deterministic can be made stochastic by S_churn != 0
            #elif sampler_type == 'heun':
            #    x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
            ## ODE deterministic 
            #elif sampler_type == 'euler':
            #    x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            ## SDE stochastic
            #elif sampler_type == 'ancestral':
            #    x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
            ## SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
            if sampler_type == 'euler_ancestral':
                x_0 = sample_euler_ancestral(self.model, x_t, sigmas, scaler=scaler, disable=True, local_cond=local_cond, global_cond=global_cond)
            ## ODE deterministic
            #elif sampler_type == 'dpm':
            #    x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
            elif sampler_type == 'ddim':
                x_0 = sample_ddim(self.model, x_t, sigmas, scaler=scaler, disable=True, local_cond=local_cond, global_cond=global_cond)
            ## ODE deterministic
            #elif sampler_type == 'dpm_adaptive':
            #    x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
            ## ODE deterministic
            #elif sampler_type == 'dpm_fast':
            #    x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
            ## 2nd order solver
            #elif sampler_type == 'dpmpp_2s_ancestral':
            #    x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            #elif sampler_type == 'dpmpp_2s':
            #    x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            ## 2nd order solver
            #elif sampler_type == 'dpmpp_2m':
            #    x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            #elif sampler_type == 'dpmpp_2m_sde':
            #    x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            else:
                raise ValueError('desired sampler type not found!')
            return x_0 

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')