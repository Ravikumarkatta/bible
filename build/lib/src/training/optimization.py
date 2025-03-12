# src/training/optimization.py
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

def get_optimizer_and_scheduler(parameters, lr=5e-5, warmup_steps=1000, total_steps=10000):
    optimizer = optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler