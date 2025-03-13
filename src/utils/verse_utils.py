# src/utils/verse_utils.py
import re
from typing import Dict, Optional, Tuple, Union

BOOK_NAME_MAPPING = {
    'gen': 'Genesis',
    'ex': 'Exodus',
    'lev': 'Leviticus',
    'num': 'Numbers',
    'deut': 'Deuteronomy',
    'josh': 'Joshua',
    'judg': 'Judges',
    'ruth': 'Ruth',
    '1 sam': '1 Samuel',
    '2 sam': '2 Samuel',
    '1 kings': '1 Kings',
    '2 kings': '2 Kings',
    '1 chron': '1 Chronicles',
    '2 chron': '2 Chronicles',
    'ezra': 'Ezra',
    'neh': 'Nehemiah',
    'esth': 'Esther',
    'job': 'Job',
    'ps': 'Psalms',
    'psa': 'Psalms',
    'psalm': 'Psalms',
    'prov': 'Proverbs',
    'eccl': 'Ecclesiastes',
    'song': 'Song of Solomon',
    'isa': 'Isaiah',
    'jer': 'Jeremiah',
    'lam': 'Lamentations',
    'ezek': 'Ezekiel',
    'dan': 'Daniel',
    'hos': 'Hosea',
    'joel': 'Joel',
    'amos': 'Amos',
    'obad': 'Obadiah',
    'jonah': 'Jonah',
    'mic': 'Micah',
    'nah': 'Nahum',
    'hab': 'Habakkuk',
    'zeph': 'Zephaniah',
    'hag': 'Haggai',
    'zech': 'Zechariah',
    'mal': 'Malachi',
    'matt': 'Matthew',
    'mark': 'Mark',
    'luke': 'Luke',
    'john': 'John',
    'acts': 'Acts',
    'rom': 'Romans',
    '1 cor': '1 Corinthians',
    '2 cor': '2 Corinthians',
    'gal': 'Galatians',
    'eph': 'Ephesians',
    'phil': 'Philippians',
    'col': 'Colossians',
    '1 thess': '1 Thessalonians',
    '2 thess': '2 Thessalonians',
    '1 tim': '1 Timothy',
    '2 tim': '2 Timothy',
    'titus': 'Titus',
    'philem': 'Philemon',
    'heb': 'Hebrews',
    'james': 'James',
    '1 pet': '1 Peter',
    '2 pet': '2 Peter',
    '1 john': '1 John',
    '2 john': '2 John',
    '3 john': '3 John',
    'jude': 'Jude',
    'rev': 'Revelation',
    # Add more mappings as needed
}

BOOK_CHAPTER_LIMITS = {
    'Genesis': 50,
    'Exodus': 40,
    'Leviticus': 27,
    'Numbers': 36,
    'Deuteronomy': 34,
    'Joshua': 24,
    'Judges': 21,
    'Ruth': 4,
    '1 Samuel': 31,
    '2 Samuel': 24,
    '1 Kings': 22,
    '2 Kings': 25,
    '1 Chronicles': 29,
    '2 Chronicles': 36,
    'Ezra': 10,
    'Nehemiah': 13,
    'Esther': 10,
    'Job': 42,
    'Psalms': 150,
    'Proverbs': 31,
    'Ecclesiastes': 12,
    'Song of Solomon': 8,
    'Isaiah': 66,
    'Jeremiah': 52,
    'Lamentations': 5,
    'Ezekiel': 48,
    'Daniel': 12,
    'Hosea': 14,
    'Joel': 3,
    'Amos': 9,
    'Obadiah': 1,
    'Jonah': 4,
    'Micah': 7,
    'Nahum': 3,
    'Habakkuk': 3,
    'Zephaniah': 3,
    'Haggai': 2,
    'Zechariah': 14,
    'Malachi': 4,
    'Matthew': 28,
    'Mark': 16,
    'Luke': 24,
    'John': 21,
    'Acts': 28,
    'Romans': 16,
    '1 Corinthians': 16,
    '2 Corinthians': 13,
    'Galatians': 6,
    'Ephesians': 6,
    'Philippians': 4,
    'Colossians': 4,
    '1 Thessalonians': 5,
    '2 Thessalonians': 3,
    '1 Timothy': 6,
    '2 Timothy': 4,
    'Titus': 3,
    'Philemon': 1,
    'Hebrews': 13,
    'James': 5,
    '1 Peter': 5,
    '2 Peter': 3,
    '1 John': 5,
    '2 John': 1,
    '3 John': 1,
    'Jude': 1,
    'Revelation': 22,
    # Add more books as needed
}

def normalize_book_name(book: str) -> str:
    """
    Normalize the Bible book name.
    
    Args:
        book: The book name to normalize
        
    Returns:
        The normalized book name
    """
    book = book.lower().strip()
    return BOOK_NAME_MAPPING.get(book, book.title())

def get_book_chapter_limits(book: str) -> int:
    """
    Get the maximum number of chapters for a given book.
    
    Args:
        book: The book name
        
    Returns:
        The maximum number of chapters in the book
    """
    normalized_book = normalize_book_name(book)
    return BOOK_CHAPTER_LIMITS.get(normalized_book, 0)

def parse_verse_reference(reference: str) -> Dict[str, Union[str, int, None]]:
    """
    Parse a Bible verse reference into its components.
    
    Examples of valid references:
    - "John 3:16"
    - "Genesis 1:1-3"
    - "Romans 8:28-30"
    - "Psalm 23"
    
    Args:
        reference: A string containing a Bible verse reference
        
    Returns:
        A dictionary with book, chapter, verse_start, and verse_end keys
    """
    # Initialize result with default values
    result = {
        "book": None,
        "chapter": None,
        "verse_start": None,
        "verse_end": None
    }
    
    # Common regex pattern for Bible references
    # Handles formats like "Book Chapter:Verse" or "Book Chapter:VerseStart-VerseEnd"
    pattern = r"([\w\s]+)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$"
    
    match = re.match(pattern, reference.strip())
    if not match:
        return result
    
    # Extract components
    book_name, chapter, verse_start, verse_end = match.groups()
    
    # Normalize book name
    normalized_book = normalize_book_name(book_name)
    
    # Fill result dictionary
    result["book"] = normalized_book
    
    if chapter:
        result["chapter"] = int(chapter)
    
    if verse_start:
        result["verse_start"] = int(verse_start)
    
    if verse_end:
        result["verse_end"] = int(verse_end)
    elif verse_start:
        # If verse_end is not specified but verse_start is, set verse_end equal to verse_start
        result["verse_end"] = int(verse_start)
    
    return result