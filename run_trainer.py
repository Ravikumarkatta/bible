#!/usr/bin/env python3
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Default config path relative to project root
DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "training_config.json")

def check_prerequisites():
    """Check if all required files and directories exist."""
    required_dirs = [
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "models")
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        print(f"Error: Config file not found at {DEFAULT_CONFIG_PATH}")
        return False
        
    return True

if __name__ == "__main__":
    if not check_prerequisites():
        print("Please run setup.py first to initialize the project")
        sys.exit(1)
        
    from src.training.trainer import Trainer
    trainer = Trainer(config_path=DEFAULT_CONFIG_PATH)
    trainer.train()
    