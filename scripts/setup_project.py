import os
from pathlib import Path
import json

from sqlalchemy import true

def create_directory_structure():
    """Create the required directory structure and config files"""
    directories = [
        "data/raw/bibles",
        "data/raw/commentaries",
        "data/raw/qa_pairs",
        "data/processed",
        "data/embeddings",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_config_files():
    """Create necessary configuration files"""
    config_files = {
        "model_config.json": {
            "model_type": "biblical-transformer",
            "vocab_size": 50000,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "max_position_embeddings": 512
        },
        "training_config.json": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "warmup_steps": 1000,
            "max_seq_length": 512,
            "gradient_accumulation_steps": 1
        },
        "data_config.json": {
            "raw_data_path": "data/raw",
            "processed_data_path": "data/processed",
            "embeddings_path": "data/embeddings",
            "tokenizer_config": {
                "vocab_size": 50000,
                "special_tokens": {
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "verse_token": "[VERSE]",
                    "ref_token": "[REF]"
                }
            },
            "augmentation_config": {
                "enabled": true,
                "techniques": [
                    "synonym_replacement",
                    "verse_reference_variation",
                    "translation_mixing"
                ],
                "aug_probability": 0.3
            }
        }
    }
    
    for filename, content in config_files.items():
        path = Path("config") / filename
        with open(path, 'w') as f:
            json.dump(content, f, indent=4)
        print(f"Created config file: {filename}")

if __name__ == "__main__":
    create_directory_structure()
    create_config_files()