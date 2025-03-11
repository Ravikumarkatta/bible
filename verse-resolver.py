"""
Bible verse resolution service.
Provides utilities for:
- Retrieving verses from different translations
- Resolving verse references
- Handling cross-references
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
import sqlite3

from src.utils.logger import setup_logger
from src.utils.verse_utils import standardize_book_name, parse_verse_reference

# Set up logging
logger = setup_logger(__name__)


class VerseResolver:
    """
    Service for resolving Bible verse references and retrieving verse text.
    Uses a combination of SQLite databases and in-memory caching for efficiency.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the verse resolver with database connections.
        
        Args:
            db_path: Path to the Bible database directory. If None, uses default.
        """
        # Set default database path if not provided
        if db_path is None:
            db_path = os.environ.get("BIBLE_DB_PATH", "data/processed/bible_db")
        
        self.db_path = db_path
        self.connections = {}  # Dictionary to store database connections by translation
        self.translation_paths = self._discover_translations()
        self.verse_cache = {}  # Simple in-memory cache
        
        logger.info(f"Initialized VerseResolver with {len(self.translation_paths)} translations")
    
    def _discover_translations(self) -> Dict[str, str]:
        """
        Discover available Bible translations in the database directory.
        
        Returns:
            Dictionary mapping translation codes to database paths
        """
        translation_paths = {}
        
        try:
            # Check if directory exists
            if not os.path.exists(self.db_path):
                logger.warning(f"Bible database directory not found: {self.db_path}")
                return translation_paths
            
            # Look for SQLite database files
            for filename in os.listdir(self.db_path):
                if filename.endswith(".db"):
                    # Extract translation code from filename (e.g., "niv.db" -> "NIV")
                    translation = filename.split(".")[0].upper()
                    db_file_path = os.path.join(self.db_path, filename)
                    translation_paths[translation] = db_file_path
                    logger.debug(f"Found translation: {translation} at {db_file_path}")
        
        except Exception as e:
            logger.error(f"Error discovering translations: {e}")
        
        return translation_paths
    
    def _get_connection(self, translation: str) -> sqlite3.Connection:
        """
        Get or create a database connection for the specified translation.
        
        Args:
            translation: Bible translation code (e.g., "NIV", "KJV")
            
        Returns:
            SQLite database connection
            
        Raises:
            ValueError: If the translation is not available
        """
        translation = translation.upper()
        
        # Return existing connection if available
        if translation in self.connections:
            return self.connections[translation]
        
        # Check if translation is available
        if translation not in self.translation_paths:
            raise ValueError(f"Translation '{translation}' not available")
        
        # Create new connection
        try:
            conn = sqlite3.connect(self.translation_paths[translation])
            self.connections[translation] = conn
            return conn
        except Exception as e:
            logger.error(f"Error connecting to {translation} database: {e}")
            raise
    
    def get_verse(self, book: str, chapter: int, verse: int, 
                 translation: str = "NIV") -> str:
        """
        Get the text of a specific Bible verse.
        
        Args:
            book: Name of the Bible book
            chapter: Chapter number
            verse: Verse number
            translation: Bible translation code (default: "NIV")
            
        Returns:
            The text of the specified verse
            
        Raises:
            ValueError: If the verse is not found
        """
        # Check cache first
        cache_key = f"{book}_{chapter}_{verse}_{translation}"
        if cache_key in self.verse_cache:
            return self.verse_cache[cache_key]
        
        # Standardize book name
        std_book = standardize_book_name(book)
        
        try:
            # Get database connection
            conn = self._get_connection(translation)
            cursor = conn.cursor()
            
            # Query the verse
            query = """
            SELECT verse_text 
            FROM verses 
            WHERE book = ? AND chapter = ? AND verse = ?
            """
            cursor.execute(query, (std_book, chapter, verse))
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"Verse not found: {std_book} {chapter}:{verse}")
            
            verse_text = result[0]
            
            # Cache the result
            self.verse_cache[cache_key] = verse_text
            
            return verse_text
            
        except Exception as e:
            logger.error(f"Error retrieving verse {std_book} {chapter}:{verse}: {e}")
            raise
    
    def get_verse_by_reference(self, reference: str, translation: str = "NIV") -> str:
        """
        Get verse text by reference string (e.g., "John 3:16").
        
        Args:
            reference: Verse reference string
            translation: Bible translation code
            
        Returns:
            The text of the specified verse
        """
        book, chapter, verse = parse_verse_reference(reference)
        return self.get_verse(book, chapter, verse, translation)
    
    def get_chapter(self, book: str, chapter: int, translation: str = "NIV") -> Dict[int, str]:
        """
        Get all verses in a chapter.
        
        Args:
            book: Name of the Bible book
            chapter: Chapter number
            translation: Bible translation code
            
        Returns:
            Dictionary mapping verse numbers to verse text
        """
        # Standardize book name
        std_book = standardize_book_name(book)
        
        try:
            # Get database connection
            conn = self._get_connection(translation)
            cursor = conn.cursor()
            
            # Query the chapter
            query = """
            SELECT verse, verse_text 
            FROM verses 
            WHERE book = ? AND chapter = ?
            ORDER BY verse
            """
            cursor.execute(query, (std_book, chapter))
            results = cursor.fetchall()
            
            if not results:
                raise ValueError(f"Chapter not found: {std_book} {chapter}")
            
            # Build chapter dictionary
            chapter_dict = {verse: text for verse, text in results}
            
            return chapter_dict
            
        except Exception as e:
            logger.error(f"Error retrieving chapter {std_book} {chapter}: {e}")
            raise
    
    def search_verses(self, query: str, translation: str = "NIV", 
                     limit: int = 10) -> List[Dict[str, Union[str, int]]]:
        """
        Search for verses containing the specified text.
        
        Args:
            query: Search text
            translation: Bible translation code
            limit: Maximum number of results to return
            
        Returns:
            List of matching verses with reference information
        """
        try:
            # Get database connection
            conn = self._get_connection(translation)
            cursor = conn.cursor()
            
            # Execute search query with FTS (full-text search)
            search_query = """
            SELECT book, chapter, verse, verse_text
            FROM verses_fts
            WHERE verse_text MATCH ?
            LIMIT ?
            """
            cursor.execute(search_query, (query, limit))
            results = cursor.fetchall()
            
            # Format results
            verses = [
                {
                    "reference": f"{book} {chapter}:{verse}",
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": verse_text
                }
                for book, chapter, verse, verse_text in results
            ]
            
            return verses
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            # Fall back to simple search if FTS fails
            return self._simple_search(query, translation, limit)
    
    def _simple_search(self, query: str, translation: str, limit: int) -> List[Dict[str, Union[str, int]]]:
        """
        Fallback simple search method without FTS.
        """
        try:
            # Get database connection
            conn = self._get_connection(translation)
            cursor = conn.cursor()
            
            # Use LIKE for simple search
            search_query = """
            SELECT book, chapter, verse, verse_text
            FROM verses
            WHERE verse_text LIKE ?
            LIMIT ?
            """
            cursor.execute(search_query, (f"%{query}%", limit))
            results = cursor.fetchall()
            
            # Format results
            verses = [
                {
                    "reference": f"{book} {chapter}:{verse}",
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": verse_text
                }
                for book, chapter, verse, verse_text in results
            ]
            
            return verses
            
        except Exception as e:
            logger.error(f"Error in simple search for '{query}': {e}")
            return []
    
    def get_cross_references(self, reference: str) -> List[str]:
        """
        Get cross-references for a given verse.
        
        Args:
            reference: Verse reference string
            
        Returns:
            List of cross-referenced verse references
        """
        book, chapter, verse = parse_verse_reference(reference)
        std_book = standardize_book_name(book)
        
        try:
            # Get cross-references from database or reference data
            # This would require a separate cross-reference database
            
            # For now, return a placeholder
            # In a real implementation, this would query a cross-reference database
            return []
            
        except Exception as e:
            logger.error(f"Error getting cross-references for {reference}: {e}")
            return []
    
    def close_connections(self):
        """
        Close all database connections.
        """
        for translation, conn in self.connections.items():
            try:
                conn.close()
                logger.debug(f"Closed connection for {translation}")
            except Exception as e:
                logger.error(f"Error closing connection for {translation}: {e}")
        
        self.connections = {}
        
    def __del__(self):
        """
        Ensure connections are closed when the object is deleted.
        """
        self.close_connections()public class Main {
    public static void main(String[] args){
    //start coding
    }
}
