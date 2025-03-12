import logging
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_data():
    """Prepare Bible texts for model training"""
    raw_dir = Path("data/raw/bibles")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Process each Bible translation
    all_verses = []
    translations = list(raw_dir.glob("*.txt"))
    
    for bible_file in translations:
        translation = bible_file.stem
        logger.info(f"Processing {translation}...")
        
        with open(bible_file, 'r', encoding='utf-8') as f:
            for line in f:
                book, chapter, verse, text = line.strip().split('|')
                all_verses.append({
                    'translation': translation,
                    'book': book,
                    'chapter': chapter,
                    'verse': verse,
                    'text': text
                })
    
    # Convert to training format
    train_data = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for verse in all_verses:
        # Tokenize verse text
        encoded = tokenizer(
            verse['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        train_data['input_ids'].append(encoded['input_ids'])
        train_data['attention_mask'].append(encoded['attention_mask'])
        train_data['labels'].append(encoded['input_ids'])  # For autoregressive training
    
    # Convert to tensors
    train_data = {
        k: torch.cat(v) for k, v in train_data.items()
    }
    
    # Save processed data
    output_file = processed_dir / "bible_data.pt"
    torch.save(train_data, output_file)
    logger.info(f"Saved {len(all_verses)} verses to {output_file}")

if __name__ == "__main__":
    prepare_training_data()