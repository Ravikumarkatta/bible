import json
from pathlib import Path
from process_bible import BibleProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_all_bibles():
    """Process all downloaded Bible translations"""
    downloads_dir = Path("data/downloads")
    output_dir = Path("data/raw/bibles")
    
    # Load configuration
    with open("config/bible_sources.json", 'r') as f:
        config = json.load(f)
    
    for translation in config['translations']:
        input_file = downloads_dir / f"{translation['id']}_bible.txt"
        
        if not input_file.exists():
            logger.warning(f"Missing Bible file for {translation['name']}")
            continue
            
        logger.info(f"Processing {translation['name']}...")
        processor = BibleProcessor(
            input_file=str(input_file),
            output_dir=str(output_dir),
            format_type=translation['format']
        )
        processor.process()

if __name__ == "__main__":
    process_all_bibles()