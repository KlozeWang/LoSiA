# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version
from .losia_projector_cuda import LoSiAProjector

class LoSiA(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        projector_auto_update = []

        for ith, group in enumerate(self.param_groups):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                    
                # GaLore Projection
                if "rank_factor" in group:
                    update_layer = (state["step"] // group["period"]) % group["total_id"]
                    update_type = group["update_type"]

                    if "projector" not in state:
                        if group['dim'] ==2:
                            if group['type'] == 'normal':
                                keys_shape = [int(group["rank_factor"] * dim) for dim in p.shape]
                            else:
                                # This case is for the output layer projection
                                keys_shape = [int(p.shape[0] * group["rank_factor"]), p.shape[1]]
                            state["projector"] = LoSiAProjector(
                                    shape = p.shape, 
                                    keys_shape = keys_shape, 
                                    device = p.device, 
                                    dtype = p.dtype,
                                    
                                    taylor_type = group["taylor_type"], 
                                    beta1 = group["imp_beta1"], 
                                    beta2 = group["imp_beta2"],
                                    scale = group['scale'],
                                    layer = group["layer"],
                                )
                        else:
                            raise NotImplementedError

                    grad = state["projector"].project(grad)

                    if update_type == "asy_period" and update_layer == group["id"]:
                        state["projector"].record(p, group["layer"])
                    elif update_type == "syn_period" and update_layer == 0:
                        state["projector"].record(p, group["layer"])
                    
                    if group["layer"] is not None \
                        and hasattr(group["layer"], 'setxy') \
                        and callable(getattr(group["layer"], 'setxy')):
                        if  (state["step"] % group["period"]) == group["period"] - 1 \
                        and ((state["step"]+1) // group["period"]) % group["total_id"] == group["id"]:
                            group["layer"].setxy(None, None)
                

                # State initialization
                if "exp_avg" not in state or state["exp_avg"] is None:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                if "rank_factor" in group:
                    kx, ky = state["projector"].get_key()
                
                    p.grad.zero_()
                    p.grad[kx[:, None], ky] = norm_grad
                    p.add_(p.grad, alpha=-step_size)
                
                else:
                    p.add_(norm_grad, alpha=-step_size)

                if "rank_factor" in group:
                    if group["weight_decay"] > 0.0:
                        p[kx[:, None], ky] *= (1 - group["lr"] * group["weight_decay"])
                    
                    if (state["step"] % group["period"]) == 0 \
                        and ((update_type == "syn_period" and update_layer == 0) \
                          or (update_type == "asy_period" and update_layer == group["id"])):
                            
                            state["projector"].reframing(group["layer"])

                            #clear optimizer
                            state["exp_avg"] = None
                            state["exp_avg_sq"] = None

                            #restart lr scheduler
                            self.scheduler.restart_in_next_step(ith, state["step"])
                else:
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            
        self.projector_auto_update = projector_auto_update
        return loss
