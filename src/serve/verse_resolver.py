import re
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

from src.utils.verse_utils import normalize_verse_reference, parse_verse_reference
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VerseResolver:
    """
    Service for resolving Bible verse references in text and retrieving verse content
    from various translations.
    """
    
    def __init__(self, bible_data_path: str = "data/processed/bibles", 
                 default_translation: str = "ESV"):
        """
        Initialize the VerseResolver service.
        
        Args:
            bible_data_path: Path to the processed Bible data
            default_translation: Default Bible translation to use
        """
        self.bible_data_path = Path(bible_data_path)
        self.default_translation = default_translation
        self.available_translations = self._load_available_translations()
        self.bible_data = {}
        
        # Reference pattern for detecting Bible verses
        # Matches patterns like "John 3:16", "Gen 1:1-3", "Psalm 23"
        self.verse_pattern = re.compile(
            r'\b((?:I{1,3}|IV|V|1|2|3|4|5)?\s*'  # Optional book number (I, II, III, IV, V, 1, 2, 3, 4, 5)
            r'(?:Genesis|Gen|Exodus|Ex|Exod|Leviticus|Lev|Numbers|Num|Deuteronomy|Deut|Joshua|Josh|'
            r'Judges|Judg|Ruth|Samuel|Sam|Kings|Kgs|Chronicles|Chron|Chr|Ezra|Nehemiah|Neh|Esther|Est|'
            r'Job|Psalms?|Ps|Proverbs|Prov|Ecclesiastes|Eccl|Songs?(?:\s+of\s+Songs?)?|Song|Isaiah|Isa|'
            r'Jeremiah|Jer|Lamentations|Lam|Ezekiel|Ezek|Daniel|Dan|Hosea|Hos|Joel|Amos|Obadiah|Obad|'
            r'Jonah|Jon|Micah|Mic|Nahum|Nah|Habakkuk|Hab|Zephaniah|Zeph|Haggai|Hag|Zechariah|Zech|'
            r'Malachi|Mal|Matthew|Matt|Mark|Luke|John|Acts|Romans|Rom|Corinthians|Cor|Galatians|Gal|'
            r'Ephesians|Eph|Philippians|Phil|Colossians|Col|Thessalonians|Thess|Timothy|Tim|Titus|'
            r'Philemon|Phlm|Hebrews|Heb|James|Jas|Peter|Pet|Jude|Revelation|Rev)\s+'  # Book name
            r'(\d+)(?::(\d+)(?:-(\d+))?)?)\b'  # Chapter and optional verse range
        )
        
        logger.info(f"VerseResolver initialized with default translation: {default_translation}")
        
    def _load_available_translations(self) -> List[str]:
        """
        Load the list of available Bible translations.
        
        Returns:
            List of available translation codes
        """
        try:
            translations_file = self.bible_data_path / "translations.json"
            if translations_file.exists():
                with open(translations_file, 'r', encoding='utf-8') as f:
                    translations_data = json.load(f)
                    return [t['code'] for t in translations_data]
            else:
                # Fallback to directory listing
                return [p.name for p in self.bible_data_path.glob("*") if p.is_dir()]
        except Exception as e:
            logger.error(f"Error loading available translations: {e}")
            return ["ESV", "KJV", "NIV"]  # Fallback to common translations
    
    def _load_translation(self, translation: str) -> bool:
        """
        Load a specific Bible translation into memory.
        
        Args:
            translation: Translation code to load
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if translation in self.bible_data:
            return True
        
        try:
            translation_path = self.bible_data_path / translation / "bible.json"
            if not translation_path.exists():
                logger.error(f"Translation file not found: {translation_path}")
                return False
                
            with open(translation_path, 'r', encoding='utf-8') as f:
                self.bible_data[translation] = json.load(f)
            logger.info(f"Loaded translation: {translation}")
            return True
        except Exception as e:
            logger.error(f"Error loading translation {translation}: {e}")
            return False
    
    def detect_verse_references(self, text: str) -> List[Tuple[str, dict]]:
        """
        Detect all Bible verse references in the given text.
        
        Args:
            text: Input text to scan for verse references
            
        Returns:
            List of tuples with (reference_text, parsed_reference)
        """
        matches = self.verse_pattern.finditer(text)
        references = []
        
        for match in matches:
            reference_text = match.group(1)
            parsed = parse_verse_reference(reference_text)
            if parsed:
                references.append((reference_text, parsed))
        
        return references
    
    def get_verse_text(self, 
                      reference: Union[str, dict], 
                      translation: Optional[str] = None) -> Optional[str]:
        """
        Get the text of a Bible verse by reference.
        
        Args:
            reference: Bible verse reference (string or parsed dict)
            translation: Bible translation to use (defaults to self.default_translation)
            
        Returns:
            Verse text if found, None otherwise
        """
        translation = translation or self.default_translation
        
        # Ensure the translation is loaded
        if translation not in self.bible_data:
            if not self._load_translation(translation):
                logger.warning(f"Could not load translation: {translation}")
                return None
        
        # Parse the reference if it's a string
        if isinstance(reference, str):
            parsed = parse_verse_reference(reference)
            if not parsed:
                logger.warning(f"Could not parse verse reference: {reference}")
                return None
        else:
            parsed = reference
        
        try:
            book = parsed['book']
            chapter = parsed['chapter']
            verse_start = parsed.get('verse_start')
            verse_end = parsed.get('verse_end')
            
            bible_data = self.bible_data[translation]
            
            # Check if book exists
            if book not in bible_data:
                logger.warning(f"Book '{book}' not found in translation {translation}")
                return None
            
            # Check if chapter exists
            if str(chapter) not in bible_data[book]:
                logger.warning(f"Chapter {chapter} not found in {book} ({translation})")
                return None
                
            # If no verse specified, return the entire chapter
            if verse_start is None:
                chapter_verses = bible_data[book][str(chapter)]
                return " ".join([f"{v}. {chapter_verses[v]}" for v in sorted(chapter_verses.keys(), key=int)])
            
            # Handle single verse
            if verse_end is None:
                if str(verse_start) not in bible_data[book][str(chapter)]:
                    logger.warning(f"Verse {verse_start} not found in {book} {chapter} ({translation})")
                    return None
                return bible_data[book][str(chapter)][str(verse_start)]
            
            # Handle verse range
            verses = []
            for v in range(verse_start, verse_end + 1):
                if str(v) in bible_data[book][str(chapter)]:
                    verses.append(f"{v}. {bible_data[book][str(chapter)][str(v)]}")
            
            if not verses:
                logger.warning(f"No verses found in range {verse_start}-{verse_end} in {book} {chapter} ({translation})")
                return None
                
            return " ".join(verses)
            
        except Exception as e:
            logger.error(f"Error retrieving verse text for {parsed}: {e}")
            return None
    
    def resolve_verses_in_text(self, 
                              text: str, 
                              translation: Optional[str] = None,
                              include_reference: bool = True) -> str:
        """
        Replace verse references in text with their actual content.
        
        Args:
            text: Input text with verse references
            translation: Bible translation to use
            include_reference: Whether to include the reference with the verse text
            
        Returns:
            Text with verse references replaced by their content
        """
        translation = translation or self.default_translation
        references = self.detect_verse_references(text)
        
        # Process references in reverse order to avoid messing up string indices
        for ref_text, parsed_ref in reversed(references):
            verse_text = self.get_verse_text(parsed_ref, translation)
            if verse_text:
                full_ref = normalize_verse_reference(parsed_ref)
                if include_reference:
                    replacement = f"{full_ref} ({translation}): \"{verse_text}\""
                else:
                    replacement = f"\"{verse_text}\""
                
                # Find the match position and replace it
                match_pos = text.find(ref_text)
                if match_pos >= 0:
                    text = text[:match_pos] + replacement + text[match_pos + len(ref_text):]
        
        return text
    
    def get_multiple_translations(self, 
                                 reference: Union[str, dict],
                                 translations: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get a verse in multiple translations.
        
        Args:
            reference: Bible verse reference
            translations: List of translations to include (defaults to all available)
            
        Returns:
            Dictionary mapping translation codes to verse text
        """
        if translations is None:
            translations = self.available_translations
        
        results = {}
        for translation in translations:
            verse_text = self.get_verse_text(reference, translation)
            if verse_text:
                results[translation] = verse_text
        
        return results