from functools import partial
import math
import math
from bisect import bisect_right
from functools import partial
from typing import (
    Any,
    Callable,
    List,
    Union,
)
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim.optimizer import Optimizer

class Recorder():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.can: bool = True
            self.lr: float = 0.0
            self.step: int = 0
            self.initialized = True

    def update(self, lr: float, step: int) -> None:
        if self.can:
            self.lr = lr
            self.step = step
            self.can = False

def get_scheculer_losia(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr_losia(
            optimizer,
            num_warmup_steps=restart_warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups_losia(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")

def get_cyclical_cosine_schedule_with_min_lr_losia(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps
    
    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda_losia,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR4LoSiA(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups_losia(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler for hdloft")
    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda_losia,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR4LoSiA(optimizer, lr_lambda, last_epoch)


def _get_cyclical_cosine_schedule_with_min_lr_lambda_losia(current_step, latest_restart_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step - latest_restart_step

    assert (current_step - latest_restart_step) % cycle_length == cycle_step, "periodic restart is a must for cyclical_cosine_schedule"

    recorder = Recorder()
    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        recorder.update(float(cycle_step) / float(max(1, num_warmup_steps)), current_step)
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    recorder.update(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay, current_step)
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda_losia(
    current_step,
    latest_restart_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step is more than full training steps"

    recorder = Recorder()
    if current_step < first_warmup_steps:
        recorder.update(float(current_step) / float(max(1, first_warmup_steps)), current_step)
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step - latest_restart_step

    if restart_step < restart_warmup_steps:
        end_of_warmup_progress = (
            float(latest_restart_step) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay

        recorder.update(float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier, current_step)
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    recorder.update(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay, current_step)
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


class LambdaLR4LoSiA(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.
    The scheduler recieve signals from optimizer when losia re-selects its subnet.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch=-1,
        verbose="deprecated",
    ):
        self.optimizer = optimizer

        self.lr_lambdas: List[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        self.latest_restart_epochs = [0] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)
    
    def restart_in_next_step(self, index, step):
        self.latest_restart_epochs[index] = step

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        return [
            base_lr * lmbda(self.last_epoch, latest_restart_epoch)
            for lmbda, latest_restart_epoch, base_lr in zip(self.lr_lambdas, self.latest_restart_epochs, self.base_lrs)
        ]
