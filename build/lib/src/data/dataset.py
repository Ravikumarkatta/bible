import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BiblicalDataset(Dataset):
    """Base dataset class for biblical text data."""
    
    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        verse_ids: Optional[torch.Tensor] = None,
        theological_ids: Optional[torch.Tensor] = None
    ):
        """
        Initialize dataset with required tensors.
        
        Args:
            input_ids: Token IDs of shape [num_samples, seq_len]
            labels: Target token IDs of shape [num_samples, seq_len]
            attention_mask: Attention mask of shape [num_samples, seq_len]
            verse_ids: Bible verse reference IDs of shape [num_samples]
            theological_ids: Theological concept IDs of shape [num_samples, num_concepts]
        """
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.verse_ids = verse_ids
        self.theological_ids = theological_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        item = {
            "input_ids": self.input_ids[idx],
        }
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
            
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
            
        if self.verse_ids is not None:
            item["verse_ids"] = self.verse_ids[idx]
            
        if self.theological_ids is not None:
            item["theological_ids"] = self.theological_ids[idx]
            
        return item

class BibleVerseDataset(BiblicalDataset):
    """Dataset specifically for Bible verses with reference tracking."""
    
    def __init__(
        self,
        verses: Dict[str, Dict[int, Dict[int, str]]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize Bible verse dataset.
        
        Args:
            verses: Nested dict of {book: {chapter: {verse: text}}}
            tokenizer: Tokenizer instance for encoding texts
            max_length: Maximum sequence length
        """
        self.verses = verses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create verse index mapping
        self.verse_indices = []
        for book in verses:
            for chapter in verses[book]:
                for verse in verses[book][chapter]:
                    self.verse_indices.append((book, chapter, verse))
    
    def __len__(self) -> int:
        return len(self.verse_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        book, chapter, verse = self.verse_indices[idx]
        text = self.verses[book][chapter][verse]
        
        # Add reference prefix to text
        reference = f"{book} {chapter}:{verse}"
        full_text = f"{reference} {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "reference": reference
        }

class CommentaryDataset(BiblicalDataset):
    """Dataset for biblical commentaries with verse alignment."""
    
    def __init__(
        self,
        commentaries: List[Dict[str, Union[str, Dict]]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize commentary dataset.
        
        Args:
            commentaries: List of commentary entries with metadata
            tokenizer: Tokenizer instance for encoding texts
            max_length: Maximum sequence length
        """
        self.commentaries = commentaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.commentaries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.commentaries[idx]
        
        # Format reference if available
        reference = ""
        if all(k in entry for k in ["book", "chapter", "verse_start"]):
            reference = f"{entry['book']} {entry['chapter']}:{entry['verse_start']}"
            if entry.get("verse_end") and entry["verse_end"] != entry["verse_start"]:
                reference += f"-{entry['verse_end']}"
        
        # Combine reference and content
        full_text = f"{reference} {entry['content']}" if reference else entry['content']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "source": entry.get("source", "unknown"),
            "reference": reference
        }