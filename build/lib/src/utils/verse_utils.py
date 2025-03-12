# src/utils/verse_utils.py

BOOK_NAME_MAPPING = {
    'gen': 'Genesis',
    'ex': 'Exodus',
    # Add more mappings as needed
}

BOOK_CHAPTER_LIMITS = {
    'Genesis': 50,
    'Exodus': 40,
    # Add more books as needed
}

def normalize_book_name(book: str) -> str:
    book = book.lower().strip()
    return BOOK_NAME_MAPPING.get(book, book)

def get_book_chapter_limits(book: str) -> int:
    normalized_book = normalize_book_name(book)
    return BOOK_CHAPTER_LIMITS.get(normalized_book, 0)