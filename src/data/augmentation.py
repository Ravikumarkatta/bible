"""
Biblical text data augmentation techniques.

This module contains functions for augmenting biblical texts and Q&A pairs
to improve model training and generalization. Techniques are specialized for
biblical content, including:
- Verse shuffling
- Paraphrasing using different translations
- Synonym replacement for non-theological terms
- Question rephrasing
- Context window manipulation
"""

import random
import re
import logging
from typing import List, Dict, Tuple, Optional, Union, Set

import nltk
from nltk.corpus import wordnet
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class BiblicalAugmenter:
    """Class for applying various augmentation techniques to biblical text."""

    def __init__(self, config: Dict = None):
        """
        Initialize the augmenter with configuration.
        
        Args:
            config: Dictionary containing augmentation parameters
        """
        self.config = config or {}
        self.theological_terms = self._load_theological_terms()
        self.bible_translations = self._load_bible_translations()
        
        # Default probabilities for different augmentation techniques
        self.prob_synonym_replacement = self.config.get('prob_synonym_replacement', 0.1)
        self.prob_verse_shuffle = self.config.get('prob_verse_shuffle', 0.3)
        self.prob_translation_swap = self.config.get('prob_translation_swap', 0.4)
        self.max_synonym_replacements = self.config.get('max_synonym_replacements', 3)
        
        logger.info("Biblical augmenter initialized with %d theological terms and %d translations", 
                   len(self.theological_terms), len(self.bible_translations))

    def _load_theological_terms(self) -> Set[str]:
        """
        Load theological terms that should not be altered during augmentation.
        
        Returns:
            Set of theological terms
        """
        # In a real implementation, this would load from a file
        # For now, we'll include a small set of example terms
        return {
            "god", "jesus", "christ", "holy spirit", "messiah", "sin", "salvation",
            "grace", "faith", "prophet", "apostle", "gospel", "covenant", "baptism",
            "resurrection", "atonement", "redemption", "justification", "sanctification",
            "trinity", "righteousness", "glory", "kingdom", "heaven", "hell", "judgment",
            "mercy", "forgiveness", "repentance", "worship", "prayer", "scripture",
            "blessing", "sacrifice", "temple", "altar", "priest", "sabbath", "passover",
            "pentecost", "tabernacle", "ark", "spirit", "angel", "demon", "satan"
        }
    
    def _load_bible_translations(self) -> Dict[str, Dict]:
        """
        Load available Bible translations for paraphrasing.
        
        Returns:
            Dictionary mapping translation codes to translation metadata
        """
        # In a real implementation, this would discover available translations
        # For now, we'll return a placeholder structure
        return {
            "KJV": {"name": "King James Version", "path": "data/raw/bibles/kjv.txt"},
            "NIV": {"name": "New International Version", "path": "data/raw/bibles/niv.txt"},
            "ESV": {"name": "English Standard Version", "path": "data/raw/bibles/esv.txt"},
            "NASB": {"name": "New American Standard Bible", "path": "data/raw/bibles/nasb.txt"},
            "NLT": {"name": "New Living Translation", "path": "data/raw/bibles/nlt.txt"}
        }

    def augment_text(self, text: str, techniques: List[str] = None) -> str:
        """
        Apply augmentation techniques to the given text.
        
        Args:
            text: The biblical text to augment
            techniques: List of specific techniques to apply, or None for all
            
        Returns:
            Augmented text
        """
        if not text:
            return text
            
        # If no specific techniques provided, apply all with their probabilities
        if techniques is None:
            techniques = []
            if random.random() < self.prob_synonym_replacement:
                techniques.append("synonym_replacement")
            if random.random() < self.prob_verse_shuffle:
                techniques.append("verse_shuffle")
            if random.random() < self.prob_translation_swap:
                techniques.append("translation_swap")
        
        augmented_text = text
        
        # Apply selected techniques
        for technique in techniques:
            if technique == "synonym_replacement":
                augmented_text = self._apply_synonym_replacement(augmented_text)
            elif technique == "verse_shuffle":
                augmented_text = self._apply_verse_shuffle(augmented_text)
            elif technique == "translation_swap":
                augmented_text = self._apply_translation_swap(augmented_text)
                
        return augmented_text
    
    def augment_qa_pair(self, question: str, answer: str) -> Tuple[str, str]:
        """
        Augment a question-answer pair.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Tuple of (augmented_question, augmented_answer)
        """
        # For questions, apply only synonym replacement and question rephrasing
        augmented_question = self._rephrase_question(question)
        
        # For answers, apply more techniques
        augmented_answer = self.augment_text(answer)
        
        return augmented_question, augmented_answer
    
    def _apply_synonym_replacement(self, text: str) -> str:
        """
        Replace non-theological words with synonyms.
        
        Args:
            text: Text to modify
            
        Returns:
            Text with some words replaced by synonyms
        """
        words = nltk.word_tokenize(text)
        num_replacements = min(self.max_synonym_replacements, max(1, int(len(words) * 0.1)))
        replacement_indices = random.sample(range(len(words)), min(num_replacements, len(words)))
        
        for idx in replacement_indices:
            word = words[idx]
            # Skip theological terms, punctuation, and short words
            if (word.lower() in self.theological_terms or
                    not word.isalnum() or len(word) <= 3):
                continue
                
            # Get part of speech
            pos = nltk.pos_tag([word])[0][1]
            wordnet_pos = self._get_wordnet_pos(pos)
            
            if not wordnet_pos:
                continue
                
            # Find synonyms
            synonyms = []
            for syn in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym.lower() not in self.theological_terms:
                        synonyms.append(synonym)
            
            # Replace with a random synonym if any found
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """
        Convert Penn Treebank POS tags to WordNet POS tags.
        
        Args:
            treebank_tag: Penn Treebank POS tag
            
        Returns:
            WordNet POS tag or None if no match
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def _apply_verse_shuffle(self, text: str) -> str:
        """
        Shuffle the order of verses in the text while preserving verse integrity.
        
        Args:
            text: Text with Bible verses
            
        Returns:
            Text with shuffled verse order
        """
        # Try to identify verses using common patterns
        verse_pattern = r'(\d+:\d+[\-\d+]*\s+[^.!?\n]+[.!?])'
        verses = re.findall(verse_pattern, text)
        
        if len(verses) <= 1:
            # Not enough verses identified, try another approach
            sentences = re.split(r'([.!?])', text)
            if len(sentences) > 2:
                # Combine sentence with its punctuation
                formatted_sentences = []
                for i in range(0, len(sentences)-1, 2):
                    if i+1 < len(sentences):
                        formatted_sentences.append(sentences[i] + sentences[i+1])
                
                # Shuffle sentences
                random.shuffle(formatted_sentences)
                return ''.join(formatted_sentences)
            return text
        
        # Shuffle verses
        random.shuffle(verses)
        return ' '.join(verses)
    
    def _apply_translation_swap(self, text: str) -> str:
        """
        Simulate swapping text between different Bible translations.
        In a real implementation, this would look up verses in different translations.
        
        Args:
            text: Original text
            
        Returns:
            Text with simulated translation differences
        """
        # Extract potential verse references
        verse_refs = re.findall(r'([A-Za-z]+\s+\d+:\d+[\-\d+]*)', text)
        
        if not verse_refs:
            # No verse references found, just return the original text
            return text
        
        # In a real implementation, the following would look up actual translations
        # For demonstration, we'll simulate translation differences
        
        # Common KJV terms and their modern equivalents
        kjv_mappings = {
            "thee": "you", "thou": "you", "thy": "your", "thine": "yours",
            "hast": "have", "hath": "has", "ye": "you", "unto": "to",
            "verily": "truly", "begat": "became the father of",
            "behold": "look", "spake": "spoke", "saith": "says",
        }
        
        # Apply random word replacements based on translation style
        words = nltk.word_tokenize(text)
        for i in range(len(words)):
            if words[i].lower() in kjv_mappings and random.random() > 0.5:
                words[i] = kjv_mappings[words[i].lower()]
                
        return ' '.join(words)
    
    def _rephrase_question(self, question: str) -> str:
        """
        Rephrase a question while maintaining its meaning.
        
        Args:
            question: Original question
            
        Returns:
            Rephrased question
        """
        # Common question starters and alternatives
        question_starters = {
            "what is": ["what's", "could you explain", "please describe", "tell me about"],
            "who is": ["who's", "could you tell me about", "tell me about", "who was"],
            "where is": ["where's", "where can I find", "what is the location of"],
            "when did": ["when was", "at what time did", "what is the time of"],
            "why did": ["for what reason did", "what was the purpose of", "what's the reason"],
            "how can": ["in what way can", "what's the method to", "what's the process for"],
        }
        
        lower_q = question.lower()
        
        # Try to match and replace a question starter
        for starter, alternatives in question_starters.items():
            if lower_q.startswith(starter):
                alternative = random.choice(alternatives)
                return alternative + question[len(starter):]
        
        # If no starter matched, just apply synonym replacement
        return self._apply_synonym_replacement(question)
        
    def augment_batch(self, texts: List[str], techniques: List[str] = None) -> List[str]:
        """
        Apply augmentation to a batch of texts.
        
        Args:
            texts: List of texts to augment
            techniques: List of techniques to apply
            
        Returns:
            List of augmented texts
        """
        return [self.augment_text(text, techniques) for text in texts]
    
    def augment_qa_batch(self, qa_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Apply augmentation to a batch of QA pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            
        Returns:
            List of augmented (question, answer) tuples
        """
        return [self.augment_qa_pair(q, a) for q, a in qa_pairs]


class ScriptureContextAugmenter:
    """Class for augmenting biblical text by manipulating context windows."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the context augmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_context_verses = self.config.get('min_context_verses', 1)
        self.max_context_verses = self.config.get('max_context_verses', 5)
        
    def expand_context(self, verse_ref: str, current_text: str) -> str:
        """
        Expand the context by adding surrounding verses.
        
        Args:
            verse_ref: Reference like "John 3:16"
            current_text: Current verse text
            
        Returns:
            Text with expanded context
        """
        # In a real implementation, this would look up actual surrounding verses
        # For now, we'll simulate expansion with placeholder text
        
        book, chapter_verse = verse_ref.split(' ', 1)
        chapter, verse = chapter_verse.split(':')
        
        # Parse verse range if present
        if '-' in verse:
            start_verse, end_verse = map(int, verse.split('-'))
        else:
            start_verse = end_verse = int(verse)
            
        # Decide how many verses to expand by
        verses_to_add = random.randint(self.min_context_verses, self.max_context_verses)
        
        # Simulate adding verses before and after
        context_before = self._generate_simulated_verses(book, chapter, start_verse-verses_to_add, start_verse-1)
        context_after = self._generate_simulated_verses(book, chapter, end_verse+1, end_verse+verses_to_add)
        
        return context_before + current_text + context_after
    
    def _generate_simulated_verses(self, book: str, chapter: str, start: int, end: int) -> str:
        """
        Generate simulated verses for context expansion.
        
        Args:
            book: Bible book name
            chapter: Chapter number
            start: Starting verse number
            end: Ending verse number
            
        Returns:
            Simulated verse text
        """
        if start < 1:
            return ""
            
        # Placeholder text for simulated verses
        result = ""
        for verse_num in range(start, end+1):
            result += f"{book} {chapter}:{verse_num} [Simulated context verse for demonstration] "
            
        return result


def augment_theological_dataset(texts: List[str], config: Dict = None) -> List[str]:
    """
    Convenience function to augment a dataset of theological texts.
    
    Args:
        texts: List of text samples
        config: Configuration dictionary
        
    Returns:
        Augmented text samples
    """
    augmenter = BiblicalAugmenter(config)
    return augmenter.augment_batch(texts)


def augment_verse_dataset(verse_refs: List[str], verse_texts: List[str], config: Dict = None) -> Tuple[List[str], List[str]]:
    """
    Augment a dataset of Bible verses with their references.
    
    Args:
        verse_refs: List of verse references (e.g., "John 3:16")
        verse_texts: List of corresponding verse texts
        config: Configuration dictionary
        
    Returns:
        Tuple of (augmented_refs, augmented_texts)
    """
    augmenter = BiblicalAugmenter(config)
    context_augmenter = ScriptureContextAugmenter(config)
    
    augmented_texts = []
    augmented_refs = []
    
    for ref, text in zip(verse_refs, verse_texts):
        # Decide whether to expand context
        if random.random() < 0.3:
            expanded_text = context_augmenter.expand_context(ref, text)
            # Update reference to reflect expanded range
            book_chapter, verse = ref.rsplit(':', 1)
            try:
                verse_num = int(verse)
                # Add context range to reference
                expanded_ref = f"{book_chapter}:{max(1, verse_num-1)}-{verse_num+1}"
            except ValueError:
                # Handle case where verse is already a range or has letters
                expanded_ref = ref
            
            augmented_texts.append(expanded_text)
            augmented_refs.append(expanded_ref)
        else:
            # Just apply regular augmentation
            augmented_texts.append(augmenter.augment_text(text))
            augmented_refs.append(ref)
            
    return augmented_refs, augmented_texts


if __name__ == "__main__":
    # Simple demonstration
    sample_text = "For God so loved the world that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life."
    sample_question = "What does John 3:16 say about God's love?"
    sample_answer = "John 3:16 says that God loved the world so much that He gave His only Son, so that everyone who believes in Him will not perish but have eternal life."
    
    augmenter = BiblicalAugmenter()
    
    print("Original text:", sample_text)
    print("Augmented text:", augmenter.augment_text(sample_text))
    
    aug_q, aug_a = augmenter.augment_qa_pair(sample_question, sample_answer)
    print("\nOriginal question:", sample_question)
    print("Augmented question:", aug_q)
    print("\nOriginal answer:", sample_answer)
    print("Augmented answer:", aug_a)