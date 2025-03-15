# src/training/evaluation.py
import torch
import math
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compute_perplexity(outputs, targets, criterion):
    """
    Compute perplexity given model outputs and targets.
    """
    try:
        loss = criterion(outputs, targets)
        return math.exp(loss.item())
    except Exception as e:
        logger.error(f"Error computing perplexity: {e}")
        return float('inf')

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader providing the evaluation dataset.
        criterion: Loss function to compute evaluation loss.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    
    Returns:
        A dictionary containing evaluation metrics such as loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Ensure batch contains input_ids and target_ids
                if len(batch) < 2:
                    logger.warning(f"Skipping batch {batch_idx} due to insufficient data.")
                    continue

                input_ids, target_ids = batch
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                # Forward pass
                outputs = model(input_ids)

                # Compute loss
                loss = criterion(outputs, target_ids)
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Compute average loss and perplexity
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')

        logger.info(f"Evaluation completed: Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f}")
        return {"loss": avg_loss, "perplexity": perplexity}

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"loss": float('inf'), "perplexity": float('inf')}