# src/training/optimization.py
import torch.optim as optim
from transformers import get_scheduler
import logging

logger = logging.getLogger(__name__)

def get_optimizer(
    parameters,
    optimizer_name="adamw",
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    **kwargs
):
    """
    Creates an optimizer based on configuration.
    
    Args:
        parameters: Model parameters to optimize
        optimizer_name: Name of the optimizer (adamw, adam, sgd)
        lr: Learning rate
        betas: Adam/AdamW beta parameters
        eps: Adam/AdamW epsilon parameter
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adamw":
        logger.info(f"Creating AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        logger.info(f"Creating Adam optimizer with lr={lr}")
        return optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        logger.info(f"Creating SGD optimizer with lr={lr}, momentum={momentum}")
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(
    optimizer,
    scheduler_type="linear_with_warmup",
    num_warmup_steps=0,
    num_training_steps=None,
    min_lr=0.0,
    last_epoch=-1,
    **kwargs
):
    """
    Creates a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to use
        scheduler_type: Type of scheduler (linear_with_warmup, cosine_with_warmup, constant_with_warmup, etc.)
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate (for applicable schedulers)
        last_epoch: The index of the last epoch when resuming training
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    # Handle percentage-based warmup
    warmup_ratio = kwargs.get("warmup_ratio", None)
    if warmup_ratio is not None and num_training_steps is not None:
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        logger.info(f"Using warmup_ratio={warmup_ratio}, computed warmup_steps={num_warmup_steps}")
    
    scheduler_args = {
        "optimizer": optimizer,
        "num_warmup_steps": num_warmup_steps,
        "last_epoch": last_epoch
    }
    
    # Add num_training_steps for schedulers that need it
    if scheduler_type in ["linear", "cosine", "cosine_with_restarts", "polynomial", 
                         "constant_with_warmup", "linear_with_warmup", 
                         "cosine_with_warmup", "cosine_with_restarts_with_warmup"]:
        if num_training_steps is None:
            raise ValueError(f"{scheduler_type} scheduler requires num_training_steps")
        scheduler_args["num_training_steps"] = num_training_steps
    
    logger.info(f"Creating {scheduler_type} scheduler with warmup_steps={num_warmup_steps}")
    
    return get_scheduler(scheduler_type, **scheduler_args)

def get_optimizer_and_scheduler(
    parameters,
    config=None,
    **kwargs
):
    """
    Sets up the optimizer and learning rate scheduler based on configuration.
    
    Args:
        parameters: Model parameters
        config: Dictionary containing optimizer and scheduler configuration
                If None, uses kwargs or defaults
        **kwargs: Override specific parameters
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    if config is None:
        config = {}
    
    # Extract optimizer and scheduler configs
    optimizer_config = config.get("optimizer", {})
    scheduler_config = optimizer_config.get("scheduler", {})
    
    # Override with kwargs if provided
    optimizer_params = {
        "optimizer_name": optimizer_config.get("name", "adamw"),
        "lr": optimizer_config.get("learning_rate", 5e-5),
        "weight_decay": optimizer_config.get("weight_decay", 0.01),
        "betas": (optimizer_config.get("beta1", 0.9), optimizer_config.get("beta2", 0.999)),
        "eps": optimizer_config.get("epsilon", 1e-8)
    }
    
    # Update with any kwargs that match optimizer params
    optimizer_params.update({k: v for k, v in kwargs.items() if k in optimizer_params})
    
    # Create optimizer
    optimizer = get_optimizer(parameters, **optimizer_params)
    
    # Calculate total steps if provided with batch info
    total_steps = kwargs.get("total_steps", None)
    if total_steps is None:
        epochs = kwargs.get("epochs", config.get("training", {}).get("max_epochs", 10))
        steps_per_epoch = kwargs.get("steps_per_epoch", None)
        if steps_per_epoch is not None:
            total_steps = steps_per_epoch * epochs
    
    # Setup scheduler params
    scheduler_params = {
        "scheduler_type": scheduler_config.get("type", "linear_with_warmup"),
        "num_warmup_steps": config.get("training", {}).get("warmup", {}).get("warmup_steps", 0),
        "num_training_steps": total_steps,
        "min_lr": scheduler_config.get("min_lr", 0)
    }
    
    # Get warmup ratio if specified
    warmup_ratio = config.get("training", {}).get("warmup", {}).get("warmup_ratio", None)
    if warmup_ratio is not None:
        scheduler_params["warmup_ratio"] = warmup_ratio
    
    # Update with any kwargs that match scheduler params
    scheduler_params.update({k: v for k, v in kwargs.items() if k in scheduler_params})
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, **scheduler_params)
    
    return optimizer, scheduler