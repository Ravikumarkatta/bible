# scripts/generate_dummy_data.py
import torch
import os

def generate_dummy_data():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    train_inputs = torch.randint(0, 50000, (1000, 20))  # 1000 samples, seq_len=20
    train_targets = torch.randint(0, 50000, (1000, 20))
    val_inputs = torch.randint(0, 50000, (200, 20))
    val_targets = torch.randint(0, 50000, (200, 20))
    torch.save(train_inputs, "data/processed/train_inputs.pt")
    torch.save(train_targets, "data/processed/train_targets.pt")
    torch.save(val_inputs, "data/processed/val_inputs.pt")
    torch.save(val_targets, "data/processed/val_targets.pt")

if __name__ == "__main__":
    generate_dummy_data()