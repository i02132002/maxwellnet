"""Learning-rate schedule shared by train.py and train_vae.py."""
import math

import torch


def make_lr_scheduler(opt, schedule, base_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup to base_lr, then constant or cosine decay to min_lr. Step once per optimizer step."""
    if warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < total steps ({total_steps})")
    floor = min_lr / base_lr

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)
