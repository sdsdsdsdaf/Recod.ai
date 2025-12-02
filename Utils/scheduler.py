import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_with_min_lr(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        # scale between min_lr and 1.0
        return min_lr + (1 - min_lr) * cosine
    return LambdaLR(optimizer, lr_lambda)
