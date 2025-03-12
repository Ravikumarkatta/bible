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
    text = re.sub(r'['']', "'", text)  
      
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
        """Load Bible book names and their variations."""  
        # This would load standard and alternate names for Bible books  
        # For example: {"Genesis": ["Gen", "Gn"], "Exodus": ["Exod", "Ex"], ...}  
        pass  
      
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
                'reference': f"{book} {chapter}:{verse_start}"  
                             f"{'-'+str(verse_end) if verse_end != verse_start else ''}"  
            })  
          
        return references  
      
    def structure_bible_text(self,   
                            bible_text: str,   
                            translation: str) -> List[Dict[str, Union[str, int]]]:  
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
      
    def prepare_training_examples(self,   
                                qa_pairs: List[Dict[str, str]],   
                                bible_data: pd.DataFrame) -> List[Dict[str, str]]:  
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
      
    def process_commentary(self,   
                          commentary_text: str,   
                          source: str) -> List[Dict[str, Union[str, List[str]]]]:  
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
      
    def create_instruction_dataset(self,   
                                  training_examples: List[Dict[str, str]]) -> pd.DataFrame:  
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
    
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor):
        """
        Initialize dataset with input and target tensors.
        
        Args:
            input_data: Input tensor of shape [num_samples, seq_len]
            target_data: Target tensor of shape [num_samples, seq_len]
        """
        assert input_data.size() == target_data.size(), "Input and target tensors must have same size"
        self.input_data = input_data
        self.target_data = target_data
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.target_data[idx]

def load_processed_data(data_path: str):
    """Load preprocessed training and validation data."""
    from pathlib import Path
    import torch
    from .dataset import BiblicalDataset
    
    data_dir = Path(data_path)
    
    train_data = BiblicalDataset(
        input_ids=torch.load(data_dir / "train_inputs.pt"),
        labels=torch.load(data_dir / "train_labels.pt"),
        attention_mask=torch.load(data_dir / "train_attention.pt")
    )
    
    val_data = BiblicalDataset(
        input_ids=torch.load(data_dir / "val_inputs.pt"),
        labels=torch.load(data_dir / "val_labels.pt"),
        attention_mask=torch.load(data_dir / "val_attention.pt")
    )
    
    return train_data, val_data