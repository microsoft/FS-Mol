import logging
import os
import time
from functools import partial
from metamol.models.interface import AbstractTorchModel
from typing import Any, Dict, Optional, Tuple, Type

import torch


logger = logging.getLogger(__name__)


def create_optimizer(
    model: AbstractTorchModel,
    lr: float = 0.001,
    task_specific_lr: float = 0.005,
    warmup_steps: int = 1000,
    task_specific_warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    # Split parameters into shared and task-specific ones:
    shared_parameters, task_spec_parameters = [], []
    for param_name, param in model.named_parameters():
        if model.is_param_task_specific(param_name):
            task_spec_parameters.append(param)
        else:
            shared_parameters.append(param)

    opt = torch.optim.Adam(
        [
            {"params": task_spec_parameters, "lr": task_specific_lr},
            {"params": shared_parameters, "lr": lr},
        ],
    )

    def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
        if cur_step >= warmup_steps:
            return 1.0
        return cur_step / warmup_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=[
            partial(
                linear_warmup, warmup_steps=task_specific_warmup_steps
            ),  # for task specific paramters
            partial(linear_warmup, warmup_steps=warmup_steps),  # for shared paramters
        ],
    )

    return opt, scheduler


def save_model(
    path: str,
    model: AbstractTorchModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
):
    data = model.get_model_state()

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch

    torch.save(data, path)


def resolve_starting_model_file(
    model_file: str,
    model_cls: Type[AbstractTorchModel],
    out_dir: str,
    use_fresh_param_init: bool,
    config_overrides: Dict[str, Any] = {},
    device: Optional[torch.device] = None,
):
    # If we start from a fresh init, create a model, do a random init, and store that away somewhere:
    if use_fresh_param_init:
        logger.info("Using fresh model init.")
        model = model_cls.build_from_model_file(
            model_file=model_file, config_overrides=config_overrides, device=device
        )

        resolved_model_file = os.path.join(out_dir, f"fresh_init.pkl")
        save_model(resolved_model_file, model)

        # Hack to give AML some time to actually save.
        time.sleep(1)
    else:
        resolved_model_file = model_file
        logger.info(f"Using model weights loaded from {resolved_model_file}.")

    return resolved_model_file
