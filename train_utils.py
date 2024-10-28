import torch
import math

def get_best_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class CosineAnnealingLR:
    def __init__(self, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        """
            Cosine learning rate schedule
        """
        if step < self.warmup_steps:
            return self.max_lr * (step+1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)