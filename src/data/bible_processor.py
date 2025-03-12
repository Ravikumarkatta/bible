import re
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibleProcessor:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.current_book = ""
        self.current_chapter = 0
        
    def process_gutenberg_bible(self):
        """Process Project Gutenberg KJV Bible format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        processed_verses = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Remove Project Gutenberg header/footer
        content = self._remove_gutenberg_artifacts(content)
        
        # Process verses
        for line in content.split('\n'):
            if verse := self._parse_verse(line):
                processed_verses.append(verse)
                
        # Save processed verses
        self._save_verses(processed_verses)
    
    def _remove_gutenberg_artifacts(self, content: str) -> str:
        """Remove Project Gutenberg header and footer."""
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            return content[start_idx:end_idx].strip()
        return content
    
    def _parse_verse(self, line: str) -> Dict:
        """Parse a single verse line into structured format."""
        # Example format: "Genesis 1:1 In the beginning..."
        verse_pattern = r"^(\d*\s*[A-Za-z]+)\s+(\d+):(\d+)\s+(.+)$"
        
        if match := re.match(verse_pattern, line.strip()):
            book, chapter, verse, text = match.groups()
            return {
                'book': book.strip(),
                'chapter': int(chapter),
                'verse': int(verse),
                'text': text.strip()
            }
        return None
    
    def _save_verses(self, verses: List[Dict]):
        """Save processed verses to output file."""
        output_file = self.output_dir / "kjv.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for verse in verses:
                f.write(f"{verse['book']}|{verse['chapter']}|{verse['verse']}|{verse['text']}\n")