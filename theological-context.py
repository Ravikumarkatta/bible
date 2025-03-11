"""
Module for managing theological context and doctrinal awareness.
This module helps the model understand different theological perspectives
and provide responses with appropriate theological framing.
"""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

class TheologicalContextManager:
    """
    Manages theological context to ensure responses are doctrinally appropriate.
    """
    
    def __init__(self, theological_data_path: str = None):
        """
        Initialize the theological context manager.
        
        Args:
            theological_data_path: Path to theological data JSON file
        """
        self.theological_traditions = {}
        self.doctrinal_positions = {}
        self.historical_contexts = {}
        
        if theological_data_path and Path(theological_data_path).exists():
            self.load_theological_data(theological_data_path)
    
    def load_theological_data(self, path: str) -> None:
        """
        Load theological data from a JSON file.
        
        Args:
            path: Path to the theological data file
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.theological_traditions = data.get('traditions', {})
            self.doctrinal_positions = data.get('doctrines', {})
            self.historical_contexts = data.get('historical_contexts', {})
    
    def identify_theological_context(self, query: str) -> List[str]:
        """
        Identify theological traditions relevant to the query.
        
        Args:
            query: User query text
            
        Returns:
            List of relevant theological tradition identifiers
        """
        relevant_traditions = []
        
        # Check for explicit mention of traditions
        for tradition, data in self.theological_traditions.items():
            keywords = data.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    relevant_traditions.append(tradition)
        
        # If no explicit mentions, return default general Christian tradition
        if not relevant_traditions:
            relevant_traditions.append('general_christian')
            
        return relevant_traditions
    
    def get_doctrinal_position(self, 
                              doctrine: str, 
                              tradition: str) -> Dict:
        """
        Get the doctrinal position for a specific theological tradition.
        
        Args:
            doctrine: The doctrine in question
            tradition: The theological tradition
            
        Returns:
            Dictionary containing the doctrinal position information
        """
        if doctrine in self.doctrinal_positions:
            positions = self.doctrinal_positions[doctrine]
            return positions.get(tradition, positions.get('general_christian', {}))
        return {}
    
    def get_historical_context(self, 
                              topic: str, 
                              period: Optional[str] = None) -> Dict:
        """
        Get historical context information for a biblical topic.
        
        Args:
            topic: The biblical topic
            period: Optional specific historical period
            
        Returns:
            Dictionary containing historical context information
        """
        if topic in self.historical_contexts:
            context = self.historical_contexts[topic]
            if period and period in context:
                return context[period]
            return context.get('general', {})
        return {}
    
    def enrich_response_with_theological_context(self,
                                               response: str,
                                               traditions: List[str],
                                               doctrines: List[str]) -> str:
        """
        Enhance response with relevant theological context.
        
        Args:
            response: The model's response text
            traditions: List of relevant theological traditions
            doctrines: List of doctrines mentioned in the response
            
        Returns:
            Enhanced response with theological context
        """
        if not traditions or not doctrines:
            return response
            
        # For primary tradition, add doctrinal context
        primary_tradition = traditions[0]
        tradition_info = self.theological_traditions.get(primary_tradition, {})
        
        # Find the most relevant doctrine to add context for
        key_doctrine = doctrines[0]
        doctrinal_position = self.get_doctrinal_position(key_doctrine, primary_tradition)
        
        if doctrinal_position:
            context_note = f"\n\nNote: {doctrinal_position.get('summary', '')}"
            
            # If there are multiple traditions, acknowledge different viewpoints
            if len(traditions) > 1:
                context_note += f" Other traditions like {traditions[1]} may have different perspectives on this matter."
                
            response += context_note
            
        return response
    
    def detect_doctrinal_topics(self, text: str) -> List[str]:
        """
        Detect doctrinal topics mentioned in text.
        
        Args:
            text: Input text
            
        Returns:
            List of doctrinal topics detected
        """
        detected_doctrines = []
        
        for doctrine, data in self.doctrinal_positions.items():
            keywords = data.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    detected_doctrines.append(doctrine)
                    break
        
        return detected_doctrines
    
    def get_tradition_representation(self, tradition: str) -> Dict:
        """
        Get the representation of a theological tradition.
        
        Args:
            tradition: Tradition identifier
            
        Returns:
            Dictionary with tradition details
        """
        return self.theological_traditions.get(tradition, {
            'name': 'General Christian',
            'description': 'Ecumenical Christian perspective'
        })
    
    def filter_response_for_tradition(self, 
                                    response: str, 
                                    traditions: List[str]) -> str:
        """
        Filter response content to be appropriate for the specified traditions.
        
        Args:
            response: Model response text
            traditions: List of theological traditions
            
        Returns:
            Filtered response text
        """
        # This would implement theological guidelines for different traditions
        # Simplified implementation for demonstration
        return response