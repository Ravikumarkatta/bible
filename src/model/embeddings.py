"""
Token and Positional Embeddings for Biblical AI Model

This module implements specialized embedding layers for the Biblical AI model,
including token embeddings, positional encodings, and verse-aware embeddings
that capture the hierarchical structure of biblical texts.
"""

import torch
import torch.nn as nn
import math
import json
import os
from typing import Dict, Optional, Tuple

class TokenEmbeddings(nn.Module):
    """
    Token embedding layer for Biblical AI.
    
    Includes special handling for verse references and theological terms.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int,
        padding_idx: int = 0,
        theological_terms_path: Optional[str] = None
    ):
        """
        Initialize token embeddings.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index used for padding
            theological_terms_path: Path to theological terms JSON file for special initialization
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings with Xavier uniform distribution
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
        # Set padding tokens to zero
        with torch.no_grad():
            self.embedding.weight[padding_idx].fill_(0)
            
        # Load and initialize theological terms with special embeddings if provided
        if theological_terms_path and os.path.exists(theological_terms_path):
            self._initialize_theological_terms(theological_terms_path)
    
    def _initialize_theological_terms(self, theological_terms_path: str):
        """
        Initialize embeddings for theological terms with special values to
        improve model's understanding of theological concepts.
        
        Args:
            theological_terms_path: Path to JSON file with term indices and relations
        """
        with open(theological_terms_path, 'r') as f:
            theological_terms = json.load(f)
        
        # Group related theological terms and initialize them with similar vectors
        term_groups = theological_terms.get('term_groups', {})
        for group_name, term_indices in term_groups.items():
            # Create a base vector for this group
            group_embedding = torch.randn(self.embedding_dim) * 0.2
            
            # Initialize all terms in this group with small variations of the base vector
            with torch.no_grad():
                for idx in term_indices:
                    if idx < self.embedding.weight.shape[0]:
                        variation = torch.randn(self.embedding_dim) * 0.05
                        self.embedding.weight[idx] = group_embedding + variation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embeddings.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_length]
            
        Returns:
            Token embeddings [batch_size, seq_length, embedding_dim]
        """
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer using sinusoidal functions.
    
    Enhanced with verse-aware positioning to maintain structural context 
    across biblical texts.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        dropout: float = 0.1, 
        max_seq_length: int = 2048,
        use_learned: bool = False
    ):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            use_learned: Whether to use learned positional embeddings instead of fixed sinusoidal
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.use_learned = use_learned
        
        if use_learned:
            # Learned positional embeddings
            self.positional_embedding = nn.Parameter(
                torch.zeros(max_seq_length, embedding_dim)
            )
            nn.init.kaiming_normal_(self.positional_embedding)
        else:
            # Fixed sinusoidal positional encodings
            position = torch.arange(max_seq_length).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2).float() * 
                (-math.log(10000.0) / embedding_dim)
            )
            
            pe = torch.zeros(max_seq_length, embedding_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Register as buffer (not a parameter)
            self.register_buffer('pe', pe)
    
    def forward(
        self, 
        x: torch.Tensor, 
        verse_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_length, embedding_dim]
            verse_positions: Optional tensor with verse position information
                            [batch_size, seq_length]
                            
        Returns:
            Embeddings with positional information [batch_size, seq_length, embedding_dim]
        """
        seq_length = x.size(1)
        
        if self.use_learned:
            # Use learned positional embeddings
            positions = self.positional_embedding[:seq_length, :].unsqueeze(0)
        else:
            # Use fixed sinusoidal embeddings
            positions = self.pe[:seq_length, :].unsqueeze(0)
        
        # If verse positions are provided, adjust the positional encoding
        if verse_positions is not None:
            # Scale positional encoding based on verse structure
            verse_weights = self._compute_verse_weights(verse_positions)
            positions = positions * verse_weights.unsqueeze(-1)
        
        x = x + positions
        return self.dropout(x)
    
    def _compute_verse_weights(self, verse_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute weights for positional encodings based on verse positions.
        
        This helps the model understand hierarchical structure of Bible texts.
        
        Args:
            verse_positions: Tensor with verse position information
                            [batch_size, seq_length]
                            
        Returns:
            Weights for positional encodings [batch_size, seq_length]
        """
        # Simple weighting: tokens in the same verse have similar weights
        batch_size, seq_length = verse_positions.shape
        weights = torch.ones_like(verse_positions, dtype=torch.float)
        
        # Iterate through each sequence in the batch
        for b in range(batch_size):
            prev_verse = -1
            for i in range(seq_length):
                curr_verse = verse_positions[b, i].item()
                if curr_verse == 0:  # 0 means not a verse position
                    continue
                
                if prev_verse != curr_verse:
                    # Start of a new verse, slightly boost the weight
                    weights[b, i] = 1.2
                    prev_verse = curr_verse
        
        return weights


class VerseAwareEmbeddings(nn.Module):
    """
    Combined embeddings module that integrates token embeddings with 
    positional encodings and verse-aware information.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        padding_idx: int = 0,
        use_learned_positional: bool = False,
        theological_terms_path: Optional[str] = None
    ):
        """
        Initialize the combined embeddings module.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            padding_idx: Index used for padding
            use_learned_positional: Whether to use learned positional embeddings
            theological_terms_path: Path to theological terms file
        """
        super().__init__()
        
        self.token_embedding = TokenEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            theological_terms_path=theological_terms_path
        )
        
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_learned=use_learned_positional
        )
        
        # Book type embedding to differentiate OT, NT, and Commentary
        self.book_type_embedding = nn.Embedding(4, embedding_dim)  # 0=pad, 1=OT, 2=NT, 3=Commentary
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        verse_positions: Optional[torch.Tensor] = None,
        book_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combined forward pass for all embedding components.
        
        Args:
            token_ids: Input tensor of token IDs [batch_size, seq_length]
            verse_positions: Tensor with verse position information [batch_size, seq_length]
            book_types: Tensor with book type information [batch_size, seq_length]
                       (0=padding, 1=OT, 2=NT, 3=Commentary)
            
        Returns:
            Combined embeddings [batch_size, seq_length, embedding_dim]
        """
        # Get token embeddings
        embeddings = self.token_embedding(token_ids)
        
        # Add positional encodings
        embeddings = self.positional_encoding(embeddings, verse_positions)
        
        # Add book type embeddings if provided
        if book_types is not None:
            book_embeddings = self.book_type_embedding(book_types)
            embeddings = embeddings + book_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class CrossVerseAttention(nn.Module):
    """
    Specialized module for cross-verse attention to help maintain context
    across verse boundaries in biblical texts.
    """
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        """
        Initialize cross-verse attention module.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention mechanism specific for cross-verse attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Projection layer
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        verse_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-verse attention to enhance contextual understanding.
        
        Args:
            x: Input embeddings [batch_size, seq_length, embedding_dim]
            verse_positions: Tensor with verse position information [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Enhanced embeddings with cross-verse context [batch_size, seq_length, embedding_dim]
        """
        # Create cross-verse attention mask
        cross_verse_mask = self._create_cross_verse_mask(verse_positions)
        
        # Combine with standard attention mask if provided
        if attention_mask is not None:
            # Expand dims for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            cross_verse_mask = cross_verse_mask * attention_mask
        
        # Apply residual connection with cross-verse attention
        attn_output, _ = self.cross_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=cross_verse_mask
        )
        
        # Project and normalize
        output = x + self.projection(attn_output)
        output = self.layer_norm(output)
        
        return output
    
    def _create_cross_verse_mask(self, verse_positions: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for cross-verse attention that encourages tokens to attend
        to other tokens within the same verse and adjacent verses.
        
        Args:
            verse_positions: Tensor with verse position information [batch_size, seq_length]
            
        Returns:
            Cross-verse attention mask [batch_size, seq_length, seq_length]
        """
        batch_size, seq_length = verse_positions.shape
        device = verse_positions.device
        
        # Initialize with ones (allow all attention)
        mask = torch.ones(batch_size, seq_length, seq_length, device=device)
        
        # Modify mask to emphasize same-verse and adjacent-verse attention
        for b in range(batch_size):
            for i in range(seq_length):
                current_verse = verse_positions[b, i].item()
                
                # If not a verse position, continue
                if current_verse == 0:
                    continue
                
                for j in range(seq_length):
                    other_verse = verse_positions[b, j].item()
                    
                    # If not a verse position, continue
                    if other_verse == 0:
                        continue
                    
                    # Determine attention weight based on verse proximity
                    if other_verse == current_verse:
                        # Same verse: full attention
                        mask[b, i, j] = 1.0
                    elif abs(other_verse - current_verse) == 1:
                        # Adjacent verse: reduced attention
                        mask[b, i, j] = 0.7
                    else:
                        # Non-adjacent verse: further reduced attention
                        mask[b, i, j] = 0.3 / (abs(other_verse - current_verse))
        
        return mask