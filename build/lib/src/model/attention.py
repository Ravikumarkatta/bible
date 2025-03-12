# src/model/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with specialized biblical context awareness.
    Extends standard transformer attention with biblical reference tracking.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int, 
        dropout_prob: float = 0.1,
        cross_reference_aware: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.cross_reference_aware = cross_reference_aware
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Special component for tracking biblical cross-references
        if cross_reference_aware:
            self.reference_gate = nn.Linear(hidden_size, 1)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi-head attention."""
        batch_size, seq_length = x.size()[:2]
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        verse_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for attention calculation.
        
        Args:
            query_states: Input tensor for queries
            key_states: Input tensor for keys
            value_states: Input tensor for values
            attention_mask: Mask to avoid attending to padding tokens
            output_attentions: Whether to return attention probabilities
            verse_positions: Optional tensor indicating positions of Bible verse references
                            (Shape: [batch_size, seq_length], values: 0 for non-verse tokens, non-zero for verse tokens)
        
        Returns:
            context_layer: Output after attention mechanism
            attention_probs: Attention probability distribution (if output_attentions=True)
        """
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply verse-aware attention if verse positions are provided
        if self.cross_reference_aware and verse_positions is not None:
            # Create a binary mask where verse tokens attend to other verse tokens more strongly
            verse_mask = verse_positions.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            verse_attn_bias = (verse_mask > 0).float() * 2.0  # Boost attention to verse tokens
            attention_scores = attention_scores + verse_attn_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create a mask to avoid attending to padding tokens (1 for tokens to attend to, 0 for tokens to ignore)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            attention_mask = (1.0 - attention_mask) * -10000.0  # Convert 0s to large negative values
            attention_scores = attention_scores + attention_mask
        
        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention weights to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_length = context_layer.size()[:2]
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        output = self.output(context_layer)
        
        if output_attentions:
            return output, attention_probs
        
        return output, None


class TheologicalContextAttention(nn.Module):
    """
    Specialized attention mechanism that prioritizes theological context.
    This module helps the model maintain theological consistency in its responses.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_theological_categories: int = 50, 
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_theological_categories = num_theological_categories
        
        # Theological category embedding
        self.theological_embedding = nn.Embedding(num_theological_categories, hidden_size)
        
        # Attention components
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.theological_key_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Category classifier (to identify the theological categories present in the text)
        self.category_classifier = nn.Linear(hidden_size, num_theological_categories)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply theological context attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            
        Returns:
            Enhanced representation with theological context awareness
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Predict theological categories present in the text
        pooled_output = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        category_logits = self.category_classifier(pooled_output)  # [batch_size, num_theological_categories]
        category_probs = F.softmax(category_logits, dim=-1)  # [batch_size, num_theological_categories]
        
        # Get theological concept embeddings weighted by their probability
        # [batch_size, num_theological_categories, hidden_size] * [batch_size, num_theological_categories, 1]
        weighted_theological_embeds = self.theological_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1) * \
                                     category_probs.unsqueeze(-1)
        
        # Combine theological embeddings
        theological_context = weighted_theological_embeds.sum(dim=1)  # [batch_size, hidden_size]
        
        # Project inputs for attention
        queries = self.query_proj(hidden_states)  # [batch_size, seq_length, hidden_size]
        theological_key = self.theological_key_proj(theological_context).unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Calculate attention scores between input and theological context
        attention_scores = torch.matmul(queries, theological_key.transpose(-1, -2))  # [batch_size, seq_length, 1]
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_probs = F.softmax(attention_scores, dim=1)
        attention_probs = self.dropout(attention_probs)
        
        # Weight input by theological relevance
        theological_attention = attention_probs * hidden_states
        
        # Combine with original input using residual connection
        output = self.layer_norm(hidden_states + self.output_proj(theological_attention))
        
        return output


class CrossReferenceAttention(nn.Module):
    """
    Specialized attention mechanism for handling biblical cross-references.
    Allows the model to connect related verses and passages.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_verses: int = 31102,  # Total verses in the Bible
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_verses = max_verses
        
        # Verse embedding to represent each Bible verse
        self.verse_embedding = nn.Embedding(max_verses + 1, hidden_size)  # +1 for "no verse" token
        
        # Cross-reference matrix (learned or pre-initialized from Bible study resources)
        self.cross_reference_matrix = nn.Parameter(torch.zeros(max_verses + 1, max_verses + 1))
        
        # Projection layers
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        verse_ids: torch.Tensor  # [batch_size, seq_length], each token mapped to verse ID or 0 if not a verse
    ) -> torch.Tensor:
        """
        Apply cross-reference attention to connect related Bible verses.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            verse_ids: Tensor of verse IDs for each token position
            
        Returns:
            Enhanced representation with cross-reference awareness
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Find positions with valid verse IDs
        has_verse = (verse_ids > 0)  # [batch_size, seq_length]
        
        if not has_verse.any():
            # No verses to process, return original
            return hidden_states
        
        # Get cross-reference scores for each verse
        # For each verse ID, get its row from the cross-reference matrix
        # This gives us the cross-reference scores to all other verses
        cross_ref_scores = F.embedding(verse_ids, self.cross_reference_matrix)  # [batch_size, seq_length, max_verses+1]
        
        # For each position, compute attention to other positions based on cross-reference scores
        # We need to map from verse IDs to sequence positions
        position_scores = torch.zeros(batch_size, seq_length, seq_length, device=hidden_states.device)
        
        # This is a simplified version - in practice, would use a more optimized implementation
        for b in range(batch_size):
            for i in range(seq_length):
                if has_verse[b, i]:
                    for j in range(seq_length):
                        if has_verse[b, j]:
                            position_scores[b, i, j] = cross_ref_scores[b, i, verse_ids[b, j]]
        
        # Normalize scores
        position_scores = position_scores / math.sqrt(self.hidden_size)
        position_scores = F.softmax(position_scores, dim=-1)
        position_scores = self.dropout(position_scores)
        
        # Apply cross-reference attention
        cross_ref_context = torch.bmm(position_scores, hidden_states)  # [batch_size, seq_length, hidden_size]
        
        # Project and apply residual connection
        output = self.layer_norm(hidden_states + self.output_proj(cross_ref_context))
        
        return output