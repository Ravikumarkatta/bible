"""
Bible verse detector and reference parser.

This module provides functionality to:
1. Detect Bible verse references in natural text
2. Parse references into standardized format
3. Validate references against the Bible canon
4. Convert between different reference formats
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.verse_utils import normalize_book_name, get_book_chapter_limits
from ..data.preprocessing import clean_text

logger = logging.getLogger(__name__)

# Load book name aliases and mappings
with open("config/bible_books.json", "r") as f:
    BIBLE_BOOKS = json.load(f)
    BOOK_ALIASES = BIBLE_BOOKS["aliases"]
    BOOK_CHAPTER_LIMITS = BIBLE_BOOKS["chapter_limits"]

# Regular expressions for verse detection
# Matches patterns like "John 3:16", "Genesis 1:1-10", "Psalm 23", etc.
VERSE_REGEX = r"((?:[1-3]\s*)?[A-Za-z]+\.?)\s*(\d+)(?:[:\.-]\s*(\d+)(?:\s*[-–—]\s*(\d+))?)?(?:\s*,\s*(\d+)(?:\s*[-–—]\s*(\d+))?)?"

class VerseReference:
    """Bible verse reference class with standardized formatting and validation."""
    
    def __init__(self, book: str, chapter: int, verse_start: Optional[int] = None, 
                 verse_end: Optional[int] = None):
        """
        Initialize a verse reference.
        
        Args:
            book: Book name (will be normalized)
            chapter: Chapter number
            verse_start: Starting verse number (optional)
            verse_end: Ending verse number (optional, for verse ranges)
        """
        self.book = normalize_book_name(book)
        self.chapter = int(chapter)
        self.verse_start = int(verse_start) if verse_start is not None else None
        self.verse_end = int(verse_end) if verse_end is not None else None
        
        if not self._validate():
            logger.warning(f"Created potentially invalid verse reference: {self}")
    
    def _validate(self) -> bool:
        """Validate that the reference points to a real Bible passage."""
        # Check if book exists
        if self.book not in BOOK_CHAPTER_LIMITS:
            return False
        
        # Check if chapter is valid for this book
        if self.chapter <= 0 or self.chapter > BOOK_CHAPTER_LIMITS[self.book]:
            return False
        
        # If verse range is specified, validate it
        if self.verse_start is not None:
            if self.verse_start <= 0:
                return False
            
            if self.verse_end is not None and self.verse_end < self.verse_start:
                return False
        
        return True
    
    def to_string(self, format_type: str = "standard") -> str:
        """
        Convert reference to string in specified format.
        
        Args:
            format_type: Format type ("standard", "short", "long")
                standard: "John 3:16-17"
                short: "Jn 3:16-17" 
                long: "The Gospel According to John, Chapter 3, verses 16 to 17"
        
        Returns:
            Formatted verse reference string
        """
        if format_type == "standard":
            if self.verse_start is None:
                return f"{self.book} {self.chapter}"
            elif self.verse_end is None:
                return f"{self.book} {self.chapter}:{self.verse_start}"
            else:
                return f"{self.book} {self.chapter}:{self.verse_start}-{self.verse_end}"
        
        elif format_type == "short":
            book_short = BOOK_ALIASES.get(self.book, {}).get("short", self.book[:3])
            if self.verse_start is None:
                return f"{book_short} {self.chapter}"
            elif self.verse_end is None:
                return f"{book_short} {self.chapter}:{self.verse_start}"
            else:
                return f"{book_short} {self.chapter}:{self.verse_start}-{self.verse_end}"
        
        elif format_type == "long":
            book_long = BOOK_ALIASES.get(self.book, {}).get("long", self.book)
            if self.verse_start is None:
                return f"{book_long}, Chapter {self.chapter}"
            elif self.verse_end is None:
                return f"{book_long}, Chapter {self.chapter}, verse {self.verse_start}"
            else:
                return f"{book_long}, Chapter {self.chapter}, verses {self.verse_start} to {self.verse_end}"
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def __str__(self) -> str:
        return self.to_string("standard")
    
    def __repr__(self) -> str:
        return f"VerseReference('{self.book}', {self.chapter}, {self.verse_start}, {self.verse_end})"


class BibleVerseDetector:
    """
    Detects Bible verse references in text using both rule-based and ML approaches.
    
    The detector combines:
    1. Regular expression matching for standard formats
    2. A fine-tuned neural model for detecting non-standard references
    3. Context-aware validation to reduce false positives
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the verse detector.
        
        Args:
            model_path: Path to pretrained neural detector model (optional)
        """
        self.regex_pattern = re.compile(VERSE_REGEX, re.IGNORECASE)
        
        # Load neural model if provided
        self.neural_model = None
        if model_path:
            self._load_neural_model(model_path)
    
    def _load_neural_model(self, model_path: str):
        """Load the neural verse detection model."""
        try:
            self.neural_model = VerseDetectionModel()
            self.neural_model.load_state_dict(torch.load(model_path))
            self.neural_model.eval()
            logger.info(f"Loaded neural verse detection model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load neural verse detection model: {e}")
            self.neural_model = None
    
    def find_references(self, text: str) -> List[VerseReference]:
        """
        Find all Bible verse references in the given text.
        
        Args:
            text: Input text to search for verse references
        
        Returns:
            List of VerseReference objects found in the text
        """
        # Clean and normalize input text
        text = clean_text(text)
        
        # Find references using regex
        regex_refs = self._find_regex_references(text)
        
        # Find references using neural model (if available)
        neural_refs = []
        if self.neural_model:
            neural_refs = self._find_neural_references(text)
        
        # Combine and deduplicate references
        all_refs = regex_refs + neural_refs
        unique_refs = self._deduplicate_references(all_refs)
        
        # Filter out invalid references
        valid_refs = [ref for ref in unique_refs if ref._validate()]
        
        return valid_refs
    
    def _find_regex_references(self, text: str) -> List[VerseReference]:
        """Find verse references using regular expressions."""
        references = []
        
        for match in self.regex_pattern.finditer(text):
            book, chapter, verse_start, verse_end, add_verse, add_verse_end = match.groups()
            
            if book and chapter:
                # Try to normalize the book name
                normalized_book = normalize_book_name(book)
                if not normalized_book:
                    continue
                
                # Handle comma-separated verses (e.g., "John 3:16, 18")
                if add_verse:
                    # First add the primary reference
                    references.append(VerseReference(
                        normalized_book, chapter, verse_start, verse_end
                    ))
                    
                    # Then add the additional verse/range
                    references.append(VerseReference(
                        normalized_book, chapter, add_verse, add_verse_end
                    ))
                else:
                    references.append(VerseReference(
                        normalized_book, chapter, verse_start, verse_end
                    ))
        
        return references
    
    def _find_neural_references(self, text: str) -> List[VerseReference]:
        """Find verse references using the neural detection model."""
        if not self.neural_model:
            return []
            
        # Implement neural model prediction logic here
        # This is a simplified placeholder
        return []
    
    def _deduplicate_references(self, references: List[VerseReference]) -> List[VerseReference]:
        """Remove duplicate references."""
        unique_refs = []
        seen = set()
        
        for ref in references:
            ref_str = str(ref)
            if ref_str not in seen:
                seen.add(ref_str)
                unique_refs.append(ref)
        
        return unique_refs


class VerseDetectionModel(nn.Module):
    """
    Neural model for detecting Bible verse references in text.
    
    This model is fine-tuned on top of a pretrained language model to identify
    non-standard references and references embedded in complex contexts.
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256, 
                 dropout: float = 0.1, context_window: int = 128):
        """
        Initialize the verse detection model.
        
        Args:
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
            context_window: Maximum context window size
        """
        super().__init__()
        
        self.context_window = context_window
        
        # BiLSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 output classes: B-REF, I-REF, O
        )
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for verse detection.
        
        Args:
            token_embeddings: Token embeddings of shape [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask of shape [batch_size, seq_len]
        
        Returns:
            Token classification logits of shape [batch_size, seq_len, 3]
        """
        # Apply BiLSTM
        lstm_out, _ = self.lstm(token_embeddings)
        
        # Apply attention if mask is provided
        if attention_mask is not None:
            # Create attention weights
            attention_weights = self.attention(lstm_out)
            
            # Apply mask
            attention_weights = attention_weights * attention_mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
            
            # Apply weighted sum
            context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
            
            # Broadcast context vector
            context_vector = context_vector.expand(-1, lstm_out.size(1), -1)
            
            # Combine context with sequence
            lstm_out = lstm_out + context_vector
        
        # Apply classifier
        logits = self.classifier(lstm_out)
        
        return logits
    
    def predict(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Predict verse reference spans.
        
        Args:
            token_embeddings: Token embeddings
            attention_mask: Attention mask
        
        Returns:
            List of token classifications (0: non-reference, 1: start of reference, 2: inside reference)
        """
        with torch.no_grad():
            logits = self.forward(token_embeddings, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to list of lists
            batch_predictions = []
            for i in range(predictions.size(0)):
                seq_preds = predictions[i].tolist()
                if attention_mask is not None:
                    # Mask out padding tokens
                    mask = attention_mask[i].bool()
                    seq_preds = [p for p, m in zip(seq_preds, mask) if m]
                batch_predictions.append(seq_preds)
            
            return batch_predictions


def parse_verse_reference(reference_text: str) -> Optional[VerseReference]:
    """
    Parse a verse reference string into a structured VerseReference object.
    
    Args:
        reference_text: Text of the verse reference
    
    Returns:
        VerseReference object or None if parsing fails
    """
    detector = BibleVerseDetector()
    refs = detector.find_references(reference_text)
    
    if refs:
        return refs[0]  # Return the first match
    return None