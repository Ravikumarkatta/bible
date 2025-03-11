# scripts/download_commentaries.py  
import requests  
import os  
import json  
import logging  
import pandas as pd  
from pathlib import Path  
from bs4 import BeautifulSoup  
from tqdm import tqdm  
from dotenv import load_dotenv  
  
load_dotenv()  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
# Directory setup  
COMMENTARY_DATA_DIR = Path(os.getenv("COMMENTARY_DATA_DIR", "./data/raw/commentaries"))  
COMMENTARY_DATA_DIR.mkdir(parents=True, exist_ok=True)  
  
# Commentary sources (public domain or with appropriate permissions)  
COMMENTARY_SOURCES = [  
    {  
        "id": "matthew_henry",  
        "name": "Matthew Henry's Complete Commentary",  
        "url": "https://www.ccel.org/ccel/henry/mhc.{book}.{chapter}.html",  
        "license": "Public Domain"  
    },  
    {  
        "id": "jamieson_fausset_brown",  
        "name": "Jamieson-Fausset-Brown Bible Commentary",  
        "url": "https://www.biblestudytools.com/commentaries/jamieson-fausset-brown/{book}/{chapter}.html",  
        "license": "Public Domain"  
    },  
    {  
        "id": "gill",  
        "name": "John Gill's Exposition of the Bible",  
        "url": "https://www.biblestudytools.com/commentaries/gills-exposition-of-the-bible/{book}/{chapter}.html",  
        "license": "Public Domain"  
    }  
    # Add more sources as needed  
]  
  
# Bible book mapping (for URL construction)  
BIBLE_BOOKS_URL_MAP = {  
    "Genesis": "gen",  
    "Exodus": "exo",  
    # ... add more mappings for all 66 books  
}  
  
def scrape_matthew_henry_commentary(book, chapter):  
    """Scrape Matthew Henry's commentary for a specific book and chapter."""  
    book_url = BIBLE_BOOKS_URL_MAP.get(book.lower(), book.lower())  
    url = COMMENTARY_SOURCES[0]["url"].format(book=book_url, chapter=chapter)  
      
    # Add delay to avoid overloading the server  
    time.sleep(1)  
      
    response = requests.get(url)  
    if response.status_code != 200:  
        logger.error(f"Failed to get commentary for {book} {chapter}: {response.status_code}")  
        return None  
      
    soup = BeautifulSoup(response.text, 'html.parser')  
      
    # Extract commentary content (specific to CCEL's structure)  
    content_div = soup.find('div', class_='ccel-content')  
    if not content_div:  
        logger.warning(f"Could not find content for {book} {chapter}")  
        return None  
      
    # Process and clean the commentary text  
    paragraphs = content_div.find_all('p')  
    commentary_text = "\n\n".join([p.get_text() for p in paragraphs])  
      
    # Extract verse references  
    verse_refs = []  
    for p in paragraphs:  
        verse_spans = p.find_all('span', class_='verse')  
        for span in verse_spans:  
            if 'id' in span.attrs:  
                verse_id = span['id']  
                if verse_id.startswith('v'):  
                    try:  
                        verse_num = int(verse_id[1:])  
                        verse_refs.append(f"{book} {chapter}:{verse_num}")  
                    except ValueError:  
                        pass  
      
    return {  
        "book": book,  
        "chapter": chapter,  
        "source": COMMENTARY_SOURCES[0]["name"],  
        "text": commentary_text,  
        "verses": verse_refs,  
        "url": url  
    }  
  
def download_all_commentaries():  
    """Download commentaries for all books and chapters of the Bible."""  
    commentaries = []  
      
    # This would loop through all books and chapters  
    # For demonstration, just doing Genesis 1-3  
    for book in ["Genesis"]:  
        for chapter in range(1, 4):  
            logger.info(f"Downloading commentary for {book} {chapter}...")  
              
            # Get commentary from each source  
            for source in COMMENTARY_SOURCES:  
                if source["id"] == "matthew_henry":  
                    commentary = scrape_matthew_henry_commentary(book, chapter)  
                    if commentary:  
                        commentaries.append(commentary)  
                # Add handlers for other commentary sources  
      
    # Convert to DataFrame and save  
    df = pd.DataFrame(commentaries)  
    csv_path = COMMENTARY_DATA_DIR / "all_commentaries.csv"  
    df.to_csv(csv_path, index=False)  
      
    # Also save as JSON  
    json_path = COMMENTARY_DATA_DIR / "all_commentaries.json"  
    df.to_json(json_path, orient="records", indent=2)  
      
    logger.info(f"Saved {len(df)} commentary entries to {csv_path} and {json_path}")  
      
    return df  
  
def main():  
    """Main function to download commentaries."""  
    download_all_commentaries()  
  
if __name__ == "__main__":  
    main()