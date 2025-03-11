"""
API endpoints for the Biblical AI application.
Provides routes for:
- Biblical question answering
- Verse reference resolution
- Theological checking
- Cross-translation comparison
"""

import logging
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model.architecture import BiblicalAIModel
from src.utils.logger import setup_logger
from src.utils.verse_utils import parse_verse_reference, is_valid_verse_reference
from src.utils.theological_checks import check_theological_accuracy
from src.serve.middleware import add_request_metadata, log_request_response
from src.serve.verse_resolver import VerseResolver

# Set up logging
logger = setup_logger(__name__)

# Initialize the app
app = FastAPI(
    title="Biblical AI API",
    description="An API for interacting with a Bible-trained language model",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(add_request_metadata)
app.middleware("http")(log_request_response)

# Initialize model
# In production, this should be loaded once and reused
model = None
verse_resolver = VerseResolver()

# Response models
class VerseReference(BaseModel):
    book: str
    chapter: int
    verse: int
    translation: Optional[str] = "NIV"
    text: Optional[str] = None


class BibleResponse(BaseModel):
    answer: str
    confidence: float
    verse_references: List[VerseReference] = []
    theological_check: Dict[str, Union[bool, str]] = {}


class TranslationComparisonResponse(BaseModel):
    verse_reference: str
    translations: Dict[str, str]
    analysis: Optional[str] = None


# Dependency to ensure model is loaded
async def get_model():
    global model
    if model is None:
        try:
            logger.info("Loading Biblical AI model...")
            # Load model from config
            from src.model.architecture import load_model
            model = load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=503, detail="Model is not available")
    return model


@app.get("/")
async def root():
    return {"message": "Welcome to Biblical AI API"}


@app.post("/ask", response_model=BibleResponse)
async def ask_question(
    request: Request,
    question: str = Query(..., description="Biblical question to answer"),
    translation: str = Query("NIV", description="Bible translation to use"),
    theological_check: bool = Query(True, description="Whether to perform theological checking"),
    model: BiblicalAIModel = Depends(get_model)
):
    """
    Answer a biblical question with references and theological verification.
    """
    logger.info(f"Received question: {question}")
    
    try:
        # Generate answer
        answer, confidence, references = model.generate_answer(question, translation)
        
        # Resolve verse references
        verse_references = []
        for ref in references:
            try:
                book, chapter, verse = parse_verse_reference(ref)
                verse_text = verse_resolver.get_verse(book, chapter, verse, translation)
                verse_references.append(
                    VerseReference(
                        book=book,
                        chapter=chapter,
                        verse=verse,
                        translation=translation,
                        text=verse_text
                    )
                )
            except Exception as e:
                logger.warning(f"Could not resolve verse reference {ref}: {e}")
        
        # Perform theological check if requested
        theo_check = {}
        if theological_check:
            theo_check = check_theological_accuracy(answer, verse_references)
        
        return BibleResponse(
            answer=answer,
            confidence=confidence,
            verse_references=verse_references,
            theological_check=theo_check
        )
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/verse/{reference}", response_model=VerseReference)
async def get_verse(
    reference: str = Query(..., description="Bible verse reference (e.g., 'John 3:16')"),
    translation: str = Query("NIV", description="Bible translation to use")
):
    """
    Retrieve a specific Bible verse by reference.
    """
    if not is_valid_verse_reference(reference):
        raise HTTPException(status_code=400, detail="Invalid verse reference format")
    
    try:
        book, chapter, verse = parse_verse_reference(reference)
        verse_text = verse_resolver.get_verse(book, chapter, verse, translation)
        
        return VerseReference(
            book=book,
            chapter=chapter,
            verse=verse,
            translation=translation,
            text=verse_text
        )
    except Exception as e:
        logger.error(f"Error retrieving verse {reference}: {e}")
        raise HTTPException(status_code=404, detail=f"Verse not found: {e}")


@app.get("/compare/{reference}", response_model=TranslationComparisonResponse)
async def compare_translations(
    reference: str = Query(..., description="Bible verse reference to compare"),
    translations: List[str] = Query(["NIV", "KJV", "ESV"], description="Translations to compare")
):
    """
    Compare a verse across multiple translations.
    """
    if not is_valid_verse_reference(reference):
        raise HTTPException(status_code=400, detail="Invalid verse reference format")
    
    try:
        book, chapter, verse = parse_verse_reference(reference)
        result = {}
        
        for trans in translations:
            try:
                verse_text = verse_resolver.get_verse(book, chapter, verse, trans)
                result[trans] = verse_text
            except Exception as e:
                logger.warning(f"Could not retrieve verse in {trans}: {e}")
                result[trans] = f"Translation not available: {e}"
        
        # Generate analysis if model is available
        analysis = None
        if model is not None:
            analysis = model.compare_translations(reference, result)
        
        return TranslationComparisonResponse(
            verse_reference=reference,
            translations=result,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error comparing translations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/theological_check")
async def theological_check(
    text: str = Query(..., description="Text to check for theological accuracy"),
    model: BiblicalAIModel = Depends(get_model)
):
    """
    Check a piece of text for theological accuracy.
    """
    try:
        check_result = check_theological_accuracy(text)
        return check_result
    except Exception as e:
        logger.error(f"Error during theological check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)