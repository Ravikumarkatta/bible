# src/training/loss.py
import torch
import torch.nn as nn

class TheologicalLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(TheologicalLoss, self).__init__()
        self.alpha = alpha  # Weight for language modeling loss
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    def forward(self, outputs, targets):
        # Language modeling loss (standard cross-entropy)
        lm_loss = self.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Placeholder for theological penalty (to be enhanced with Theological_Checker)
        # For now, we use a dummy penalty; integrate with Theological_Checker.py later
        theological_penalty = 0.0
        
        total_loss = self.alpha * lm_loss + (1 - self.alpha) * theological_penalty
        return total_loss