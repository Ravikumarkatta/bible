"""
Module for detecting and processing cross-references between Bible verses.
This allows the model to understand and leverage interconnected Biblical content.
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ..utils.verse_utils import normalize_verse_reference, parse_verse_reference
from ..data.preprocessing import clean_text

class CrossReferenceDetector:
    """
    Detects, processes, and retrieves cross-references for Biblical content.
    """
    
    def __init__(self, cross_reference_path: str = None, embedding_model=None):
        """
        Initialize the cross-reference detector.
        
        Args:
            cross_reference_path: Path to pre-compiled cross-reference JSON file
            embedding_model: Model to generate embeddings for semantic cross-references
        """
        self.explicit_refs = {}
        self.semantic_refs = {}
        self.embedding_model = embedding_model
        
        if cross_reference_path and Path(cross_reference_path).exists():
            self.load_cross_references(cross_reference_path)
    
    def load_cross_references(self, path: str) -> None:
        """
        Load pre-compiled cross-references from a JSON file.
        
        Args:
            path: Path to the cross-reference file
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.explicit_refs = data.get('explicit', {})
            self.semantic_refs = data.get('semantic', {})
    
    def detect_references(self, text: str) -> List[str]:
        """
        Detect any explicit Bible verse references in the text.
        
        Args:
            text: Text to search for verse references
            
        Returns:
            List of normalized verse references found in the text
        """
        # Common Bible reference patterns
        patterns = [
            r'(\d*\s*[A-Za-z]+\s+\d+:\d+(?:-\d+)?)',  # Genesis 1:1 or Genesis 1:1-5
            r'(\d*\s*[A-Za-z]+\s+\d+)',  # Genesis 1 (entire chapter)
            r'(Psalm\s+\d+:\d+(?:-\d+)?)',  # Psalm 119:1 or Psalm 119:1-5
            r'(Ps\.\s+\d+:\d+(?:-\d+)?)'   # Ps. 119:1 or Ps. 119:1-5
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized_ref = normalize_verse_reference(match)
                if normalized_ref:
                    references.append(normalized_ref)
        
        return references
    
    def get_explicit_cross_references(self, verse_ref: str) -> List[str]:
        """
        Retrieve explicit cross-references for a given verse.
        
        Args:
            verse_ref: Normalized verse reference string
            
        Returns:
            List of related verse references
        """
        return self.explicit_refs.get(verse_ref, [])
    
    def get_semantic_cross_references(self, 
                                    verse_ref: str, 
                                    verse_text: str, 
                                    top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve semantically related verses based on embedding similarity.
        
        Args:
            verse_ref: Normalized verse reference
            verse_text: Text of the verse
            top_k: Number of top related verses to return
            
        Returns:
            List of tuples containing (verse_ref, similarity_score)
        """
        if not self.embedding_model:
            return []
            
        # If we have pre-computed semantic references, use those
        if verse_ref in self.semantic_refs:
            return self.semantic_refs[verse_ref][:top_k]
            
        # Otherwise compute on the fly
        verse_embedding = self.embedding_model.encode(clean_text(verse_text))
        
        # This would require a database of verse embeddings in practice
        # For demonstration, return empty list when not pre-computed
        return []
    
    def build_cross_reference_graph(self, seed_verses: List[str], depth: int = 2) -> Dict:
        """
        Build a graph of interconnected verses up to a certain depth.
        
        Args:
            seed_verses: Starting verse references
            depth: How many levels of connections to explore
            
        Returns:
            Dictionary representing the verse reference graph
        """
        graph = {verse: set() for verse in seed_verses}
        frontier = set(seed_verses)
        
        for _ in range(depth):
            new_frontier = set()
            for verse in frontier:
                cross_refs = self.get_explicit_cross_references(verse)
                graph[verse].update(cross_refs)
                
                for ref in cross_refs:
                    if ref not in graph:
                        graph[ref] = set()
                        new_frontier.add(ref)
            
            frontier = new_frontier
            if not frontier:
                break
                
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in graph.items()}
    
    def enrich_output_with_references(self, 
                                     response: str, 
                                     context_verses: List[str], 
                                     max_refs: int = 3) -> str:
        """
        Enhance model response with relevant cross-references.
        
        Args:
            response: Model's text response
            context_verses: Verses known to be in the context
            max_refs: Maximum number of references to include
            
        Returns:
            Enhanced response with cross-references
        """
        # Detect verses already mentioned in the response
        mentioned_verses = self.detect_references(response)
        
        # Find cross-references from context verses
        potential_refs = []
        for verse in context_verses:
            refs = self.get_explicit_cross_references(verse)
            potential_refs.extend([r for r in refs if r not in mentioned_verses])
        
        # Deduplicate and limit
        potential_refs = list(set(potential_refs))[:max_refs]
        
        if potential_refs:
            response += "\n\nRelated verses you might find helpful: " + ", ".join(potential_refs)
            
        return response