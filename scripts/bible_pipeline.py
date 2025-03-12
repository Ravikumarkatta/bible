import os
import requests
import logging
import json
import zipfile
from pathlib import Path
from typing import Dict, List
from process_bible import BibleProcessor
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiblePipeline:
    """Pipeline for downloading and processing Bible translations"""
    
    BIBLE_SOURCES = {
        "kjv": {
            "name": "King James Version",
            "url": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
            "format": "gutenberg"
        },
        "web": {
            "name": "World English Bible",
            "url": "https://ebible.org/Scriptures/engwebp_usfx.zip",
            "format": "xml"
        },
        "asv": {
            "name": "American Standard Version",
            "url": "https://www.gutenberg.org/cache/epub/8/pg8.txt",
            "format": "gutenberg"
        },
        "douay": {
            "name": "Douay-Rheims Bible",
            "url": "https://www.gutenberg.org/cache/epub/8300/pg8300.txt",
            "format": "gutenberg"
        },
        "niv": {
            "name": "New International Version",
            "url": "YOUR_NIV_SOURCE_URL",  # Replace with actual URL
            "format": "xml"
        },
        "nlt": {
            "name": "New Living Translation",
            "url": "YOUR_NLT_SOURCE_URL",  # Replace with actual URL
            "format": "xml"
        },
        "nkjv": {
            "name": "New King James Version",
            "url": "YOUR_NKJV_SOURCE_URL",  # Replace with actual URL
            "format": "xml"
        }
    }
    
    def __init__(self):
        self.downloads_dir = Path("data/downloads")
        self.raw_dir = Path("data/raw/bibles")
        self.processed_dir = Path("data/processed")
        
        # Create directories
        for dir_path in [self.downloads_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_url(self, url: str) -> bool:
        """Validate URL before attempting download"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _handle_zip_file(self, zip_path: Path, translation_id: str) -> Path:
        """Extract ZIP file and return path to the main Bible file"""
        extract_dir = self.downloads_dir / translation_id
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the main Bible file (assuming it's the largest .txt or .xml file)
        bible_files = []
        for ext in ['.txt', '.xml']:
            bible_files.extend(extract_dir.glob(f'**/*{ext}'))
        
        if not bible_files:
            raise ValueError(f"No suitable Bible file found in ZIP for {translation_id}")
            
        main_file = max(bible_files, key=lambda x: x.stat().st_size)
        return main_file

    def download_bible(self, translation_id: str, force: bool = False, max_retries: int = 3) -> bool:
        """Download a Bible translation with retry mechanism"""
        if translation_id not in self.BIBLE_SOURCES:
            logger.error(f"Unknown translation: {translation_id}")
            return False
            
        source = self.BIBLE_SOURCES[translation_id]
        is_zip = source['url'].endswith('.zip')
        output_file = self.downloads_dir / f"{translation_id}_bible{'_temp.zip' if is_zip else '.txt'}"
        
        if output_file.exists() and not force and not is_zip:
            logger.info(f"Skipping {source['name']}, already downloaded")
            return True
            
        if not self.validate_url(source['url']):
            logger.error(f"Invalid URL for {source['name']}: {source['url']}")
            return False

        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {source['name']}... (Attempt {attempt + 1}/{max_retries})")
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(source['url'], headers=headers, timeout=30)
                response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                if is_zip:
                    try:
                        actual_file = self._handle_zip_file(output_file, translation_id)
                        output_file.unlink()  # Remove the temp zip file
                        return True
                    except Exception as e:
                        logger.error(f"Failed to process ZIP file for {source['name']}: {str(e)}")
                        return False
                
                logger.info(f"Downloaded {source['name']}")
                return True
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download {source['name']} after {max_retries} attempts: {str(e)}")
                    return False
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue
    
    def process_bible(self, translation_id: str) -> bool:
        """Process a downloaded Bible translation"""
        if translation_id not in self.BIBLE_SOURCES:
            logger.error(f"Unknown translation: {translation_id}")
            return False
            
        source = self.BIBLE_SOURCES[translation_id]
        is_zip = source['url'].endswith('.zip')
        
        if is_zip:
            input_file = self.downloads_dir / translation_id / f"{translation_id}_bible.txt"
            if not input_file.exists():
                # Try to find any suitable file in the extracted directory
                extracted_dir = self.downloads_dir / translation_id
                bible_files = []
                for ext in ['.txt', '.xml']:
                    bible_files.extend(extracted_dir.glob(f'**/*{ext}'))
                if bible_files:
                    input_file = bible_files[0]
        else:
            input_file = self.downloads_dir / f"{translation_id}_bible.txt"
        
        if not input_file.exists():
            logger.warning(f"Missing Bible file for {source['name']}")
            return False
        
        try:
            logger.info(f"Processing {source['name']}...")
            processor = BibleProcessor(
                input_file=str(input_file),
                output_dir=str(self.raw_dir),
                format_type=source['format']
            )
            processor.process()
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {source['name']}: {str(e)}")
            return False
    
    def run_pipeline(self, translations: List[str] = None, force_download: bool = False):
        """Run full pipeline for specified translations"""
        if translations is None:
            translations = list(self.BIBLE_SOURCES.keys())
            
        results = {
            "downloads": [],
            "processed": []
        }
        
        # Download phase
        for translation_id in translations:
            if self.download_bible(translation_id, force_download):
                results["downloads"].append(translation_id)
                
        # Processing phase
        for translation_id in results["downloads"]:
            if self.process_bible(translation_id):
                results["processed"].append(translation_id)
                
        # Report results
        logger.info(f"Pipeline complete:")
        logger.info(f"Downloads: {len(results['downloads'])} successful")
        logger.info(f"Processed: {len(results['processed'])} successful")
        
        return results

if __name__ == "__main__":
    pipeline = BiblePipeline()
    pipeline.run_pipeline(force_download=True)