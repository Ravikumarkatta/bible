# src/training/evaluation.py
import torch
import math

def compute_perplexity(outputs, targets, criterion):
    loss = criterion(outputs, targets)
    return math.exp(loss.item())

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, target_ids)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": perplexity}