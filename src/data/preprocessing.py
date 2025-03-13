# src/data/preprocessing.py  
import re
from flask import json
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import os

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize Bible text."""
    # Remove HTML tags if present
    text = BeautifulSoup(text, 'lxml').get_text()
      
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
      
    # Handle verse numbers consistently
    text = re.sub(r'(\d+):(\d+)', r' \1:\2 ', text)
      
    # Normalize quotes
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r"['']", "'", text)
      
    return text

class BiblePreprocessor:
    """Preprocesses Bible texts for fine-tuning."""
      
    def __init__(self, config_path: str):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
          
        # Load NLTK resources  
        nltk.download('punkt')
        nltk.download('stopwords')
          
        self.book_names = self._load_book_names()
        self.verse_pattern = re.compile(
            r'(\d*\s*[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?'
        )
      
    def _load_book_names(self) -> Dict[str, List[str]]:
        """
        Load Bible book names and their variations.
        
        Returns:
            A dictionary mapping full Bible book names to a list of common abbreviations.
        """
        return {
            "Genesis": ["Gen", "Gn"],
            "Exodus": ["Exod", "Ex"],
            "Leviticus": ["Lev"],
            "Numbers": ["Num"],
            "Deuteronomy": ["Deut"],
            "Joshua": ["Josh"],
            "Judges": ["Judg"],
            "Ruth": ["Ruth"],
            "1 Samuel": ["1 Sam", "1Sm"],
            "2 Samuel": ["2 Sam", "2Sm"],
            "1 Kings": ["1 Kgs", "1Ki"],
            "2 Kings": ["2 Kgs", "2Ki"],
            "1 Chronicles": ["1 Chron", "1Ch"],
            "2 Chronicles": ["2 Chron", "2Ch"],
            "Ezra": ["Ezra"],
            "Nehemiah": ["Neh"],
            "Esther": ["Esth"],
            "Job": ["Job"],
            "Psalms": ["Ps", "Pslm"],
            "Proverbs": ["Prov"],
            "Ecclesiastes": ["Eccl"],
            "Song of Solomon": ["Song", "SoS"],
            "Isaiah": ["Isa"],
            "Jeremiah": ["Jer"],
            "Lamentations": ["Lam"],
            "Ezekiel": ["Ezek"],
            "Daniel": ["Dan"],
            "Hosea": ["Hos"],
            "Joel": ["Joel"],
            "Amos": ["Amos"],
            "Obadiah": ["Obad"],
            "Jonah": ["Jonah"],
            "Micah": ["Mic"],
            "Nahum": ["Nah"],
            "Habakkuk": ["Hab"],
            "Zephaniah": ["Zeph"],
            "Haggai": ["Hag"],
            "Zechariah": ["Zech"],
            "Malachi": ["Mal"]
        }
      
    def clean_text(self, text: str) -> str:
        """Clean and normalize Bible text."""
        return clean_text(text)
      
    def parse_verse_references(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Extract Bible verse references from text."""
        matches = self.verse_pattern.finditer(text)
        references = []
          
        for match in matches:
            book = match.group(1)
            chapter = int(match.group(2))
            verse_start = int(match.group(3))
            verse_end = int(match.group(4)) if match.group(4) else verse_start
              
            references.append({
                'book': book,
                'chapter': chapter,
                'verse_start': verse_start,
                'verse_end': verse_end,
                'reference': f"{book} {chapter}:{verse_start}" +
                             (f"-{verse_end}" if verse_end != verse_start else "")
            })
          
        return references
      
    def structure_bible_text(self, bible_text: str, translation: str) -> List[Dict[str, Union[str, int]]]:
        """Structure Bible text into verses with metadata."""
        structured_data = []
        current_book = None
        current_chapter = None
          
        for line in bible_text.strip().split('\n'):
            # Check if line is a book header
            book_match = re.match(r'^# (.+)$', line)
            if book_match:
                current_book = book_match.group(1)
                current_chapter = None
                continue
              
            # Check if line is a chapter header
            chapter_match = re.match(r'^## Chapter (\d+)$', line)
            if chapter_match:
                current_chapter = int(chapter_match.group(1))
                continue
              
            # Check if line is a verse
            verse_match = re.match(r'^(\d+)\s+(.+)$', line)
            if verse_match and current_book and current_chapter:
                verse_num = int(verse_match.group(1))
                verse_text = verse_match.group(2)
                  
                structured_data.append({
                    'book': current_book,
                    'chapter': current_chapter,
                    'verse': verse_num,
                    'text': verse_text,
                    'reference': f"{current_book} {current_chapter}:{verse_num}",
                    'translation': translation
                })
          
        return structured_data
      
    def prepare_training_examples(self, qa_pairs: List[Dict[str, str]], bible_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Prepare training examples from QA pairs and Bible data."""
        training_examples = []
          
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            answer = qa_pair['answer']
              
            # Extract verse references from question and answer
            question_refs = self.parse_verse_references(question)
            answer_refs = self.parse_verse_references(answer)
              
            # Get relevant Bible verses
            relevant_verses = []
            for ref in question_refs + answer_refs:
                verses = bible_data[
                    (bible_data['book'] == ref['book']) &
                    (bible_data['chapter'] == ref['chapter']) &
                    (bible_data['verse'] >= ref['verse_start']) &
                    (bible_data['verse'] <= ref['verse_end'])
                ]
                if not verses.empty:
                    relevant_verses.extend(verses.to_dict('records'))
              
            # Format as training example
            context = "\n".join([
                f"{v['reference']} ({v['translation']}): {v['text']}"
                for v in relevant_verses
            ])
              
            training_example = {
                'question': question,
                'answer': answer,
                'context': context,
                'references': [ref['reference'] for ref in question_refs + answer_refs]
            }
              
            training_examples.append(training_example)
          
        return training_examples
      
    def process_commentary(self, commentary_text: str, source: str) -> List[Dict[str, Union[str, List[str]]]]:
        """Process biblical commentary text."""
        commentary_sections = []
        current_section = None
        current_text = []
        current_verses = []
          
        for line in commentary_text.strip().split('\n'):
            # Check if line is a section header
            section_match = re.match(r'^# (.+)$', line)
            if section_match:
                # Save previous section if exists
                if current_section and current_text:
                    commentary_sections.append({
                        'title': current_section,
                        'text': '\n'.join(current_text),
                        'verses': current_verses,
                        'source': source
                    })
                  
                current_section = section_match.group(1)
                current_text = []
                current_verses = []
                continue
              
            # Extract verse references
            verse_refs = self.parse_verse_references(line)
            if verse_refs:
                current_verses.extend([ref['reference'] for ref in verse_refs])
              
            # Add line to current text
            current_text.append(line)
          
        # Add the last section
        if current_section and current_text:
            commentary_sections.append({
                'title': current_section,
                'text': '\n'.join(current_text),
                'verses': current_verses,
                'source': source
            })
          
        return commentary_sections
      
    def create_instruction_dataset(self, training_examples: List[Dict[str, str]]) -> pd.DataFrame:
        """Create instruction dataset for fine-tuning."""
        instructions = []
          
        for example in training_examples:
            # Format as instruction
            instruction = {
                'instruction': 'Answer the biblical question based on the provided context.',
                'input': f"Question: {example['question']}\n\nContext: {example['context']}",
                'output': example['answer'],
                'references': example['references']
            }
              
            instructions.append(instruction)
          
        return pd.DataFrame(instructions)

class BiblicalDataset(Dataset):
    """Dataset class for biblical text data."""
    
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        """
        Initialize dataset with input tensors.
        
        Args:
            input_ids: Input tensor of shape [num_samples, seq_len]
            labels: Target tensor of shape [num_samples, seq_len]
            attention_mask: Attention mask tensor of shape [num_samples, seq_len]
        """
        assert input_ids.size() == labels.size(), "Input and label tensors must have same size"
        assert input_ids.size() == attention_mask.size(), "Input and attention mask tensors must have same size"
        
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx], self.attention_mask[idx]

# ----------------- Missing Data Loading Script for the Trainer -----------------
# The following classes and functions are taken from the missing integration code (dataset.py)
# and are now included in preprocessing.py for a complete training data pipeline.

from transformers import PreTrainedTokenizer

class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize dataset from instruction data.
        
        Args:
            data_path: Path to instruction JSON file
            tokenizer: HuggingFace tokenizer to use
            max_length: Maximum sequence length
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} instruction examples")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load instruction data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example."""
        item = self.data[idx]
        
        # Format as instruction prompt
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']
        
        # Format prompt according to instruction tuning format
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "
        
        # Tokenize prompt
        prompt_tokenized = self.tokenizer(
            prompt, 
            max_length=self.max_length // 2,  # Reserve half length for output
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize output (labels)
        output_tokenized = self.tokenizer(
            output,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Combine input_ids: prompt followed by output
        input_ids = torch.cat([
            prompt_tokenized['input_ids'].squeeze(),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        # Create attention mask (1 for prompt and output tokens, 0 for padding)
        attention_mask = torch.cat([
            prompt_tokenized['attention_mask'].squeeze(),
            output_tokenized['attention_mask'].squeeze()
        ])[:self.max_length]
        
        # Create labels tensor: -100 for prompt tokens (ignored in loss), actual ids for output
        labels = torch.cat([
            torch.full_like(prompt_tokenized['input_ids'].squeeze(), -100),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BibleInstructionDataset(train_path, tokenizer, max_length)
    val_dataset = BibleInstructionDataset(val_path, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader
