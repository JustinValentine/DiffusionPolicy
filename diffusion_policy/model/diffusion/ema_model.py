import copy
import torch
import numpy as np

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.model = model
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    def to(self, device):
        self.averaged_model.to(device)

    @torch.no_grad()
    def get(self):
        return self.averaged_model

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        
        for p_net, p_ema in zip(new_model.buffers(), self.averaged_model.buffers()):
            p_ema.copy_(p_net)

        self.optimization_step += 1

    def state_dict(self):
        return {
            "optimization_step": self.optimization_step,
            "averaged_model": self.averaged_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimization_step = state_dict["optimization_step"]
        self.averaged_model.load_state_dict(state_dict["averaged_model"])


class PowerModel:
    """
    Power function of models weights
    """

    def __init__(
        self,
        model,
        stds=[0.050, 0.100]
    ):

        self.model = model
        self.averaged_models = [copy.deepcopy(model) for _ in stds]
        self.stds = stds
        for averaged_model in self.averaged_models:
            averaged_model.eval()
            averaged_model.requires_grad_(False)

        self.optimization_step = 1

    def std_to_exp(self, std):
        tmp = std ** -2
        exp = np.roots([1, 7, 16 - tmp, 12 -tmp]).real.max()
        return exp


    def get_decay(self, gamma):
        """
        Compute the decay factor for the exponential moving average.
        """

        beta = (1 - 1/self.optimization_step) ** (gamma + 1) 
        return beta

    def to(self, device):
        for model in self.averaged_models:
            model.to(device)

    @torch.no_grad()
    def get(self):
        return self.averaged_models

    @torch.no_grad()
    def step(self, new_model):
        for std, ema in zip(self.stds, self.averaged_models):
            gamma = self.std_to_exp(std)
            decay = self.get_decay(gamma)

            for module, ema_module in zip(new_model.modules(), ema.modules()):            
                for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                    # iterative over immediate parameters only.
                    if isinstance(param, dict):
                        raise RuntimeError('Dict parameter not supported')
                    elif not param.requires_grad:
                        ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                    else:
                        ema_param.mul_(decay)
                        ema_param.add_(param.data.to(dtype=ema_param.dtype).data, alpha=(1 - decay))
    
            for p_net, p_ema in zip(new_model.buffers(), ema.buffers()):
                p_ema.copy_(p_net)

        self.optimization_step += 1

    def state_dict(self):
        return dict(
            stds=self.stds,
            optimization_step=self.optimization_step,
            averaged_models=[model.state_dict() for model in self.averaged_models],
        )

    def load_state_dict(self, state_dict):
        self.stds = state_dict['stds']
        self.optimization_step = state_dict['optimization_step']
        for model, model_state_dict in zip(self.averaged_models, state_dict['averaged_models']):
            model.load_state_dict(model_state_dict)

