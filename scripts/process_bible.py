import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibleProcessor:
    def __init__(self, input_file: str, output_dir: str, format_type: str = "gutenberg"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
    
    def process(self) -> None:
        """Process Bible text based on format type"""
        processors = {
            "gutenberg": self.process_gutenberg_bible,
            "ebible": self.process_ebible_format
        }
        
        if self.format_type not in processors:
            raise ValueError(f"Unsupported format: {self.format_type}")
            
        processors[self.format_type]()
    
    def process_gutenberg_bible(self) -> None:
        """Process Project Gutenberg KJV Bible format"""
        try:
            # Read input file
            logger.info(f"Reading Bible text from {self.input_file}")
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process content
            content = self._remove_gutenberg_artifacts(content)
            verses = self._parse_verses(content)
            
            # Save processed verses
            self._save_verses(verses)
            logger.info("Bible processing completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process Bible: {str(e)}")
            raise
    
    def process_ebible_format(self) -> None:
        """Process eBible format"""
        try:
            logger.info(f"Processing eBible format from {self.input_file}")
            verses = []
            
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('$'):
                        parts = line.strip().split('|')
                        if len(parts) >= 4:
                            verses.append({
                                'book': parts[0].lstrip('$'),
                                'chapter': int(parts[1]),
                                'verse': int(parts[2]),
                                'text': parts[3]
                            })
            
            self._save_verses(verses)
            logger.info("eBible processing completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process eBible format: {str(e)}")
            raise
    
    def _remove_gutenberg_artifacts(self, content: str) -> str:
        """Remove Project Gutenberg header and footer"""
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            return content[start_idx:end_idx].strip()
        return content
    
    def _parse_verses(self, content: str) -> List[Dict]:
        """Parse Bible content into structured verses"""
        verses = []
        current_book = ""
        verse_pattern = r"^(\d*\s*[A-Za-z]+)\s+(\d+):(\d+)\s+(.+)$"
        
        for line in content.split('\n'):
            if match := re.match(verse_pattern, line.strip()):
                book, chapter, verse, text = match.groups()
                verses.append({
                    'book': book.strip(),
                    'chapter': int(chapter),
                    'verse': int(verse),
                    'text': text.strip()
                })
        
        return verses
    
    def _save_verses(self, verses: List[Dict]) -> None:
        """Save processed verses to output file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "kjv.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for verse in verses:
                f.write(f"{verse['book']}|{verse['chapter']}|{verse['verse']}|{verse['text']}\n")
        
        logger.info(f"Saved processed verses to {output_file}")

def main():
    input_file = Path("data/downloads/kjv_bible.txt")
    
    try:
        processor = BibleProcessor(
            input_file=str(input_file),
            output_dir="data/raw/bibles",
            format_type="gutenberg"
        )
        processor.process()
        
    except Exception as e:
        logger.error(f"Failed to process Bible text: {str(e)}")

if __name__ == "__main__":
    main()