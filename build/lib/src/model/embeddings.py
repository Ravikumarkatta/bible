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
        if padding_idx is not None:
            self.embedding.weight.data[padding_idx].zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embedding_dim)
    
    @property
    def weight(self) -> torch.Tensor:
        """Expose the embedding weights for weight tying."""
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """Positional encoding with sinusoidal pattern."""
    
    def __init__(self, d_model: int, max_seq_length: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length to handle
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin/cos positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (won't be updated during training)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        Returns:
            Output tensor of same shape with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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