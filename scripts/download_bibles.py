# scripts/download_bibles.py  
import requests  
import os  
import json  
import logging  
import pandas as pd  
from pathlib import Path  
from tqdm import tqdm  
from dotenv import load_dotenv  
  
load_dotenv()  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  
  
# API keys and endpoints  
BIBLE_API_KEY = os.getenv("BIBLE_API_KEY")  
BIBLE_API_URL = "https://api.scripture.api.bible/v1"  
  
# Directory setup  
BIBLE_DATA_DIR = Path(os.getenv("BIBLE_DATA_DIR", "./data/raw/bibles"))  
BIBLE_DATA_DIR.mkdir(parents=True, exist_ok=True)  
  
# Bible translations to download  
TRANSLATIONS = [  
    {"id": "kjv", "name": "King James Version"},  
    {"id": "niv", "name": "New International Version"},  
    {"id": "esv", "name": "English Standard Version"},  
    {"id": "nlt", "name": "New Living Translation"},  
    {"id": "nasb", "name": "New American Standard Bible"},  
    {"id": "nrsv", "name": "New Revised Standard Version"},  
    # Add more translations as needed  
]  
  
def get_bible_books(translation_id):  
    """Get all books for a given Bible translation."""  
    headers = {"api-key": BIBLE_API_KEY}  
    url = f"{BIBLE_API_URL}/bibles/{translation_id}/books"  
      
    response = requests.get(url, headers=headers)  
    if response.status_code != 200:  
        logger.error(f"Failed to get books for {translation_id}: {response.text}")  
        return []  
      
    return response.json()["data"]  
  
def get_bible_chapters(translation_id, book_id):  
    """Get all chapters for a given book in a Bible translation."""  
    headers = {"api-key": BIBLE_API_KEY}  
    url = f"{BIBLE_API_URL}/bibles/{translation_id}/books/{book_id}/chapters"  
      
    response = requests.get(url, headers=headers)  
    if response.status_code != 200:  
        logger.error(f"Failed to get chapters for {book_id} in {translation_id}: {response.text}")  
        return []  
      
    return response.json()["data"]  
  
def get_bible_verses(translation_id, chapter_id):  
    """Get all verses for a given chapter in a Bible translation."""  
    headers = {"api-key": BIBLE_API_KEY}  
    url = f"{BIBLE_API_URL}/bibles/{translation_id}/chapters/{chapter_id}/verses"  
      
    response = requests.get(url, headers=headers)  
    if response.status_code != 200:  
        logger.error(f"Failed to get verses for {chapter_id} in {translation_id}: {response.text}")  
        return []  
      
    return response.json()["data"]  
  
def get_verse_text(translation_id, verse_id):  
    """Get the text for a specific verse."""  
    headers = {"api-key": BIBLE_API_KEY}  
    url = f"{BIBLE_API_URL}/bibles/{translation_id}/verses/{verse_id}?content-type=text"  
      
    response = requests.get(url, headers=headers)  
    if response.status_code != 200:  
        logger.error(f"Failed to get text for verse {verse_id} in {translation_id}: {response.text}")  
        return None  
      
    return response.json()["data"]["content"]  
  
def download_bible_translation(translation):  
    """Download an entire Bible translation."""  
    translation_id = translation["id"]  
    translation_name = translation["name"]  
      
    logger.info(f"Downloading {translation_name} ({translation_id})...")  
      
    # Create directory for this translation  
    translation_dir = BIBLE_DATA_DIR / translation_id  
    translation_dir.mkdir(exist_ok=True)  
      
    # Initialize a dataframe to store all verses  
    all_verses = []  
      
    # Get all books  
    books = get_bible_books(translation_id)  
    for book in tqdm(books, desc=f"Processing books for {translation_id}"):  
        book_id = book["id"]  
        book_name = book["name"]  
          
        # Get all chapters for this book  
        chapters = get_bible_chapters(translation_id, book_id)  
        for chapter in tqdm(chapters, desc=f"Processing chapters for {book_name}", leave=False):  
            chapter_id = chapter["id"]  
            chapter_number = chapter["number"]  
              
            # Get all verses for this chapter  
            verses = get_bible_verses(translation_id, chapter_id)  
            for verse in verses:  
                verse_id = verse["id"]  
                verse_number = verse["reference"]  
                  
                # Get the verse text  
                verse_text = get_verse_text(translation_id, verse_id)  
                if verse_text:  
                    # Add to the dataframe  
                    all_verses.append({  
                        "translation": translation_id,  
                        "translation_name": translation_name,  
                        "book_id": book_id,  
                        "book_name": book_name,  
                        "chapter": int(chapter_number) if chapter_number.isdigit() else chapter_number,  
                        "verse": int(verse_number) if verse_number.isdigit() else verse_number,  
                        "reference": f"{book_name} {chapter_number}:{verse_number}",  
                        "text": verse_text  
                    })  
      
    # Save as CSV  
    df = pd.DataFrame(all_verses)  
    csv_path = translation_dir / f"{translation_id}_complete.csv"  
    df.to_csv(csv_path, index=False)  
      
    # Also save in JSON format  
    json_path = translation_dir / f"{translation_id}_complete.json"  
    df.to_json(json_path, orient="records", indent=2)  
      
    logger.info(f"Saved {len(df)} verses from {translation_name} to {csv_path} and {json_path}")  
      
    return df  
  
def main():  
    """Main function to download all specified Bible translations."""  
    all_translations = []  
      
    for translation in TRANSLATIONS:  
        df = download_bible_translation(translation)  
        all_translations.append(df)  
      
    # Combine all translations  
    combined_df = pd.concat(all_translations)  
    combined_csv_path = BIBLE_DATA_DIR / "all_translations.csv"  
    combined_df.to_csv(combined_csv_path, index=False)  
      
    logger.info(f"Saved combined dataset with {len(combined_df)} verses to {combined_csv_path}")  
  
if __name__ == "__main__":  
    main()