import json
import requests
import logging
from pathlib import Path
from zipfile import ZipFile
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibleDownloader:
    def __init__(self, config_path: str = "config/bible_sources.json"):
        self.downloads_dir = Path("data/downloads")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def download_all(self):
        """Download all configured Bible translations"""
        for translation in self.config['translations']:
            try:
                output_file = self.downloads_dir / f"{translation['id']}_bible.txt"
                
                if output_file.exists():
                    logger.info(f"Skipping {translation['name']}, already downloaded")
                    continue
                
                logger.info(f"Downloading {translation['name']}...")
                response = requests.get(translation['url'])
                response.raise_for_status()
                
                if translation['url'].endswith('.zip'):
                    self._handle_zip_download(response.content, translation)
                else:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                
                logger.info(f"Successfully downloaded {translation['name']}")
                
            except Exception as e:
                logger.error(f"Failed to download {translation['name']}: {str(e)}")

    def _handle_zip_download(self, content: bytes, translation: Dict):
        """Handle downloaded zip files"""
        zip_path = self.downloads_dir / f"{translation['id']}_temp.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(content)
        
        with ZipFile(zip_path) as zf:
            # Extract the main Bible text file
            bible_file = next(name for name in zf.namelist() if name.endswith('.txt'))
            zf.extract(bible_file, self.downloads_dir)
        
        # Rename extracted file
        extracted = self.downloads_dir / bible_file
        target = self.downloads_dir / f"{translation['id']}_bible.txt"
        extracted.rename(target)
        
        # Cleanup
        zip_path.unlink()

def main():
    downloader = BibleDownloader()
    downloader.download_all()

if __name__ == "__main__":
    main()