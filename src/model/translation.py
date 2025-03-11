"""
Module for comparing and analyzing different Bible translations.
Helps the model understand nuances between translations and provide
context about translation differences.
"""

import json
import difflib
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..utils.verse_utils import normalize_verse_reference, parse_verse_reference

class TranslationComparator:
    """
    Compares different Bible translations and provides analysis of differences.
    """
    
    def __init__(self, translations_path: str = None):
        """
        Initialize the translation comparator.
        
        Args:
            translations_path: Path to directory with Bible translation files
        """
        self.translations = {}
        self.translation_metadata = {}
        
        if translations_path and Path(translations_path).exists():
            self.load_translations(translations_path)
    
    def load_translations(self, directory_path: str) -> None:
        """
        Load Bible translations from a directory.
        
        Args:
            directory_path: Path to directory containing translation files
        """
        translations_dir = Path(directory_path)
        
        # Load translation metadata
        metadata_path = translations_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.translation_metadata = json.load(f)
        
        # Load each translation file
        for translation_file in translations_dir.glob("*.json"):
            if translation_file.name == "metadata.json":
                continue
                
            translation_id = translation_file.stem
            with open(translation_file, 'r', encoding='utf-8') as f:
                self.translations[translation_id] = json.load(f)
    
    def get_verse(self, 
                verse_ref: str, 
                translation: str = "KJV") -> Optional[str]:
        """
        Get a verse from a specific translation.
        
        Args:
            verse_ref: Normalized verse reference
            translation: Translation identifier
            
        Returns:
            Verse text or None if not found
        """
        if translation not in self.translations:
            return None
            
        # Parse the reference to get book, chapter, verse
        book, chapter, verse_start, verse_end = parse_verse_reference(verse_ref)
        if not book or not chapter:
            return None
            
        # Get the verse from the translation data structure
        # Assumes JSON structure: {"Book": {"chapter": {"verse": "text"}}}
        try:
            translation_data = self.translations[translation]
            chapter_data = translation_data.get(book, {}).get(str(chapter), {})
            
            if verse_end and verse_end > verse_start:
                # Handle verse ranges
                verses = []
                for v in range(verse_start, verse_end + 1):
                    if str(v) in chapter_data:
                        verses.append(chapter_data[str(v)])
                return " ".join(verses) if verses else None
            else:
                # Single verse
                return chapter_data.get(str(verse_start))
        except Exception:
            return None
    
    def compare_translations(self, 
                           verse_ref: str, 
                           translations: List[str] = None) -> Dict[str, str]:
        """
        Compare a verse across multiple translations.
        
        Args:
            verse_ref: Normalized verse reference
            translations: List of translation identifiers to compare
            
        Returns:
            Dictionary mapping translation IDs to verse text
        """
        if translations is None:
            translations = list(self.translations.keys())
            
        result = {}
        for trans_id in translations:
            verse_text = self.get_verse(verse_ref, trans_id)
            if verse_text:
                result[trans_id] = verse_text
                
        return result
    
    def get_translation_info(self, translation_id: str) -> Dict:
        """
        Get metadata about a specific translation.
        
        Args:
            translation_id: Translation identifier
            
        Returns:
            Dictionary with translation metadata
        """
        return self.translation_metadata.get(translation_id, {
            "name": translation_id,
            "year": "Unknown",
            "description": "No description available"
        })
    
    def highlight_differences(self, 
                            verse_ref: str, 
                            translation1: str, 
                            translation2: str) -> Dict:
        """
        Highlight textual differences between two translations.
        
        Args:
            verse_ref: Normalized verse reference
            translation1: First translation identifier
            translation2: Second translation identifier
            
        Returns:
            Dictionary with diff analysis
        """
        text1 = self.get_verse(verse_ref, translation1)
        text2 = self.get_verse(verse_ref, translation2)
        
        if not text1 or not text2:
            return {"error": "One or both translations not available"}
            
        # Generate word-level diff
        words1 = text1.split()
        words2 = text2.split()
        
        differ = difflib.Differ()
        diff = list(differ.compare(words1, words2))
        
        # Process diff result
        added = [word[2:] for word in diff if word.startswith('+ ')]
        removed = [word[2:] for word in diff if word.startswith('- ')]
        
        return {
            "verse_ref": verse_ref,
            "translation1": {
                "id": translation1,
                "text": text1,
                "info": self.get_translation_info(translation1)
            },
            "translation2": {
                "id": translation2,
                "text": text2,
                "info": self.get_translation_info(translation2)
            },
            "differences": {
                "added": added,
                "removed": removed,
                "diff_text": " ".join(diff)
            }
        }
    
    def get_modern_equivalent(self, 
                            verse_ref: str, 
                            archaic_translation: str = "KJV", 
                            modern_translation: str = "NIV") -> Dict:
        """
        Get modern translation equivalent of a verse from an archaic translation.
        
        Args:
            verse_ref: Normalized verse reference
            archaic_translation: Archaic translation identifier
            modern_translation: Modern translation identifier
            
        Returns:
            Dictionary with both versions and analysis
        """
        archaic_text = self.get_verse(verse_ref, archaic_translation)
        modern_text = self.get_verse(verse_ref, modern_translation)
        
        if not archaic_text or not modern_text:
            return {"error": "One or both translations not available"}
        
        # Add simple analysis of key differences
        archaic_words = set(archaic_text.lower().split())
        modern_words = set(modern_text.lower().split())
        
        unique_to_archaic = archaic_words - modern_words
        unique_to_modern = modern_words - archaic_words
        
        return {
            "verse_ref": verse_ref,
            "archaic": {
                "translation": archaic_translation,
                "text": archaic_text
            },
            "modern": {
                "translation": modern_translation,
                "text": modern_text
            },
            "analysis": {
                "archaic_terms": list(unique_to_archaic),
                "modern_terms": list(unique_to_modern)
            }
        }
    
    def format_translation_comparison(self, 
                                    verse_ref: str, 
                                    translations: List[str] = None) -> str:
        """
        Format a user-friendly comparison of translations.
        
        Args:
            verse_ref: Normalized verse reference
            translations: List of translation identifiers
            
        Returns:
            Formatted string with comparison
        """
        comparison = self.compare_translations(verse_ref, translations)
        
        if not comparison:
            return f"No translations found for {verse_ref}"
            
        result = [f"Translation comparison for {verse_ref}:"]
        
        for trans_id, text in comparison.items():
            info = self.get_translation_info(trans_id)
            result.append(f"\n{info.get('name', trans_id)} ({info.get('year', 'Unknown')}): {text}")
            
        return "\n".join(result)