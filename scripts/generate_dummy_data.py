# scripts/generate_dummy_data.py
import torch
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
processed_dir = project_root / "data" / "processed"

def generate_dummy_data(append=True):
    """Generate dummy training data for testing.
    
    Args:
        append (bool): If True, append to existing data. If False, create new data.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate new dummy tensors
    new_batch_size = 900  # Additional samples to add
    seq_length = 512
    vocab_size = 50000
    
    # Load existing data if appending
    if append and (processed_dir / "train.pt").exists():
        existing_train = torch.load(processed_dir / "train.pt")
        existing_val = torch.load(processed_dir / "val.pt")
        
        # Generate additional data
        new_train_data = {
            'input_ids': torch.randint(0, vocab_size, (new_batch_size, seq_length)),
            'labels': torch.randint(0, vocab_size, (new_batch_size, seq_length)),
            'attention_mask': torch.ones((new_batch_size, seq_length))
        }
        
        new_val_data = {
            'input_ids': torch.randint(0, vocab_size, (new_batch_size//5, seq_length)),
            'labels': torch.randint(0, vocab_size, (new_batch_size//5, seq_length)),
            'attention_mask': torch.ones((new_batch_size//5, seq_length))
        }
        
        # Append new data to existing data
        train_data = {
            'input_ids': torch.cat((existing_train['input_ids'], new_train_data['input_ids'])),
            'labels': torch.cat((existing_train['labels'], new_train_data['labels'])),
            'attention_mask': torch.cat((existing_train['attention_mask'], new_train_data['attention_mask']))
        }
        
        val_data = {
            'input_ids': torch.cat((existing_val['input_ids'], new_val_data['input_ids'])),
            'labels': torch.cat((existing_val['labels'], new_val_data['labels'])),
            'attention_mask': torch.cat((existing_val['attention_mask'], new_val_data['attention_mask']))
        }
    else:
        # Generate dummy tensors
        batch_size = 1000  # Increased from 100
        seq_length = 512
        vocab_size = 50000
        
        # Training data
        train_data = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_length)),
            'labels': torch.randint(0, vocab_size, (batch_size, seq_length)),
            'attention_mask': torch.ones((batch_size, seq_length))
        }
        
        # Validation data 
        val_data = {
            'input_ids': torch.randint(0, vocab_size, (batch_size//5, seq_length)),
            'labels': torch.randint(0, vocab_size, (batch_size//5, seq_length)), 
            'attention_mask': torch.ones((batch_size//5, seq_length))
        }
    
    # Save tensors
    torch.save(train_data, processed_dir / "train.pt")
    torch.save(val_data, processed_dir / "val.pt")
    
    print(f"Generated dummy data in {processed_dir}")
    print(f"Train samples: {len(train_data['input_ids'])}, Val samples: {len(val_data['input_ids'])}")

if __name__ == "__main__":
    generate_dummy_data()