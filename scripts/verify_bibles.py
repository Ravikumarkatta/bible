import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_bible_files():
    """Verify processed Bible files"""
    bibles_dir = Path("data/raw/bibles")
    
    for bible_file in bibles_dir.glob("*.txt"):
        verses = 0
        books = set()
        
        with open(bible_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 4:
                    books.add(parts[0])
                    verses += 1
        
        logger.info(f"{bible_file.name}:")
        logger.info(f"  Books: {len(books)}")
        logger.info(f"  Verses: {verses}")
        logger.info("---")

if __name__ == "__main__":
    verify_bible_files()