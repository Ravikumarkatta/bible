<<<<<<< HEAD
"""
Custom tokenizer for biblical content.

This module provides specialized tokenization for biblical texts,
handling verse references, theological terms, and ancient language
elements appropriately.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class BiblicalTokenizer:
    """
    Tokenizer specialized for biblical texts and theological content.
    
    This tokenizer handles special cases in biblical text including:
    - Bible verse references (e.g., "John 3:16", "Genesis 1:1-3")
    - Ancient language terms (Hebrew, Greek, Aramaic)
    - Theological terminology
    - Special punctuation and formatting in biblical texts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the biblical tokenizer.
        
        Args:
            config_path: Path to tokenizer configuration JSON file
        """
        self.config = self._load_config(config_path)
        self._initialize_patterns()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load tokenizer configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing tokenizer configuration
        """
        default_config = {
            "preserve_verse_refs": True,
            "split_sentences": True,
            "lowercase": False,
            "remove_punctuation": False,
            "theological_terms_path": None,
            "ancient_terms_path": None,
            "special_tokens": {
                "VERSE_REF": "[VERSE_REF]",
                "HEBREW": "[HEBREW]",
                "GREEK": "[GREEK]",
                "ARAMAIC": "[ARAMAIC]",
                "THEOLOGICAL_TERM": "[THEO_TERM]",
                "UNKNOWN": "[UNK]"
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    for key, value in loaded_config.items():
                        if key in default_config:
                            default_config[key] = value
                logger.info(f"Loaded tokenizer config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default tokenizer configuration")
        
        return default_config
    
    def _initialize_patterns(self):
        """Initialize regex patterns for special token detection."""
        # Bible verse reference pattern
        # Matches patterns like "John 3:16", "Genesis 1:1-10", "1 Cor. 13:4-7", etc.
        self.verse_pattern = re.compile(
            r'\b(?:'
            r'(?:(?:1|2|3|I|II|III)\s*)?'  # Optional book number
            r'(?:Genesis|Gen|Exodus|Ex|Exod|Leviticus|Lev|Numbers|Num|Deuteronomy|Deut|Joshua|Josh|'
            r'Judges|Judg|Ruth|1\s*Samuel|1\s*Sam|2\s*Samuel|2\s*Sam|1\s*Kings|2\s*Kings|'
            r'1\s*Chronicles|1\s*Chron|2\s*Chronicles|2\s*Chron|Ezra|Nehemiah|Neh|Esther|Est|'
            r'Job|Psalms?|Ps|Proverbs|Prov|Ecclesiastes|Eccl|Song\s*of\s*Solomon|Song|Isaiah|Isa|'
            r'Jeremiah|Jer|Lamentations|Lam|Ezekiel|Ezek|Daniel|Dan|Hosea|Hos|Joel|Amos|'
            r'Obadiah|Obad|Jonah|Jon|Micah|Mic|Nahum|Nah|Habakkuk|Hab|Zephaniah|Zeph|'
            r'Haggai|Hag|Zechariah|Zech|Malachi|Mal|Matthew|Matt|Mark|Luke|John|Jn|'
            r'Acts|Romans|Rom|1\s*Corinthians|1\s*Cor|2\s*Corinthians|2\s*Cor|Galatians|Gal|'
            r'Ephesians|Eph|Philippians|Phil|Colossians|Col|1\s*Thessalonians|1\s*Thess|'
            r'2\s*Thessalonians|2\s*Thess|1\s*Timothy|1\s*Tim|2\s*Timothy|2\s*Tim|Titus|'
            r'Philemon|Phlm|Hebrews|Heb|James|Jas|1\s*Peter|1\s*Pet|2\s*Peter|2\s*Pet|'
            r'1\s*John|2\s*John|3\s*John|Jude|Revelation|Rev)'
            r'\s+'  # Space between book name and chapter
            r'(\d+):'  # Chapter number followed by colon
            r'(\d+)'  # Verse number
            r'(?:-(\d+))?'  # Optional ending verse (for ranges)
            r')\b'
        )
        
        # Load theological terms if path is provided
        self.theological_terms = set()
        if self.config.get("theological_terms_path"):
            try:
                with open(self.config["theological_terms_path"], 'r', encoding='utf-8') as f:
                    self.theological_terms = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.theological_terms)} theological terms")
            except Exception as e:
                logger.warning(f"Could not load theological terms: {e}")
        
        # Load ancient language terms if path is provided
        self.ancient_terms = {}
        if self.config.get("ancient_terms_path"):
            try:
                with open(self.config["ancient_terms_path"], 'r', encoding='utf-8') as f:
                    self.ancient_terms = json.load(f)
                logger.info(f"Loaded ancient language terms: {len(self.ancient_terms.get('hebrew', []))} Hebrew, "
                           f"{len(self.ancient_terms.get('greek', []))} Greek, "
                           f"{len(self.ancient_terms.get('aramaic', []))} Aramaic")
            except Exception as e:
                logger.warning(f"Could not load ancient language terms: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize biblical text into tokens.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # First, handle special tokens if configured to preserve them
        if self.config["preserve_verse_refs"]:
            # Replace verse references with special tokens to preserve them during tokenization
            text, verse_refs = self._extract_verse_refs(text)
        
        # Split into sentences if configured
        if self.config["split_sentences"]:
            sentences = sent_tokenize(text)
            # Tokenize each sentence
            tokens = []
            for sentence in sentences:
                tokens.extend(word_tokenize(sentence))
        else:
            # Tokenize the whole text at once
            tokens = word_tokenize(text)
        
        # Restore verse references if they were extracted
        if self.config["preserve_verse_refs"]:
            tokens = self._restore_verse_refs(tokens, verse_refs)
        
        # Handle other special cases
        tokens = self._handle_special_cases(tokens)
        
        # Apply additional processing based on config
        if self.config["lowercase"]:
            tokens = [token.lower() for token in tokens]
            
        if self.config["remove_punctuation"]:
            tokens = [token for token in tokens if not all(c in '.,;:!?"\'()[]{}' for c in token)]
        
        return tokens
    
    def _extract_verse_refs(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract verse references from text and replace with placeholders.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (modified text, list of extracted verse references)
        """
        verse_refs = []
        positions = []
        
        # Find all verse references
        for match in self.verse_pattern.finditer(text):
            verse_refs.append(match.group(0))
            positions.append((match.start(), match.end()))
        
        # Replace verse references with placeholders, starting from the end
        # to preserve the positions of earlier references
        modified_text = text
        for i in range(len(positions) - 1, -1, -1):
            start, end = positions[i]
            placeholder = f" {self.config['special_tokens']['VERSE_REF']} "
            modified_text = modified_text[:start] + placeholder + modified_text[end:]
            
        return modified_text, verse_refs
    
    def _restore_verse_refs(self, tokens: List[str], verse_refs: List[str]) -> List[str]:
        """
        Restore verse references in tokenized output.
        
        Args:
            tokens: List of tokens
            verse_refs: List of verse references to restore
            
        Returns:
            Tokens with verse references restored
        """
        restored_tokens = []
        ref_index = 0
        
        for token in tokens:
            if token == self.config['special_tokens']['VERSE_REF'] and ref_index < len(verse_refs):
                # Replace placeholder with the actual verse reference
                restored_tokens.append(verse_refs[ref_index])
                ref_index += 1
            else:
                restored_tokens.append(token)
                
        return restored_tokens
    
    def _handle_special_cases(self, tokens: List[str]) -> List[str]:
        """
        Process tokens to handle special biblical text cases.
        
        Args:
            tokens: List of tokens to process
            
        Returns:
            Processed tokens
        """
        processed_tokens = []
        
        for token in tokens:
            # Check if token is a theological term
            if self.theological_terms and token in self.theological_terms:
                if self.config.get("mark_theological_terms", False):
                    processed_tokens.append(f"{self.config['special_tokens']['THEOLOGICAL_TERM']}_{token}")
                else:
                    processed_tokens.append(token)
                continue
                
            # Check if token is an ancient language term
            is_ancient = False
            for lang, terms in self.ancient_terms.items():
                if token in terms:
                    lang_token = self.config['special_tokens'].get(lang.upper(), f"[{lang.upper()}]")
                    processed_tokens.append(f"{lang_token}_{token}")
                    is_ancient = True
                    break
            
            if not is_ancient:
                processed_tokens.append(token)
                
        return processed_tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            Reconstructed text
        """
        # Basic detokenization - join with spaces and fix punctuation
        text = ' '.join(tokens)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        
        # Fix spacing around quotes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Fix special tokens
        for token_type, token_value in self.config['special_tokens'].items():
            if token_type != 'UNKNOWN':  # Skip UNK token
                pattern = f"{token_value}_([a-zA-Z0-9]+)"
                text = re.sub(pattern, r'\1', text)
        
        return text
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized texts
        """
        return [self.tokenize(text) for text in texts]
    
    def save_vocabulary(self, output_path: str, corpus: Optional[List[str]] = None):
        """
        Generate and save vocabulary from a corpus or from loaded resources.
        
        Args:
            output_path: Path to save vocabulary file
            corpus: Optional corpus of texts to generate vocabulary from
        """
        vocabulary = set()
        
        # Add special tokens
        for token in self.config['special_tokens'].values():
            vocabulary.add(token)
            
        # Add theological terms
        vocabulary.update(self.theological_terms)
        
        # Add ancient language terms
        for terms in self.ancient_terms.values():
            vocabulary.update(terms)
            
        # Process corpus if provided
        if corpus:
            for text in corpus:
                tokens = self.tokenize(text)
                vocabulary.update(tokens)
                
        # Save vocabulary to file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in sorted(vocabulary):
                f.write(f"{word}\n")
                
        logger.info(f"Saved vocabulary with {len(vocabulary)} tokens to {output_path}")


def create_tokenizer(config_path: Optional[str] = None) -> BiblicalTokenizer:
    """
    Factory function to create a BiblicalTokenizer instance.
    
    Args:
        config_path: Optional path to tokenizer configuration
        
    Returns:
        Initialized BiblicalTokenizer instance
    """
    return BiblicalTokenizer(config_path)


if __name__ == "__main__":
    # Example usage
    tokenizer = create_tokenizer()
    
    test_text = ("In John 3:16, Jesus said 'For God so loved the world.' "
                "This illustrates the theological concept of atonement. "
                "The Hebrew word 'shalom' means peace.")
    
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    reconstructed = tokenizer.detokenize(tokens)
    print(f"Reconstructed: {reconstructed}")
=======
"""
Custom tokenizer for biblical content.

This module provides specialized tokenization for biblical texts,
handling verse references, theological terms, and ancient language
elements appropriately.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class BiblicalTokenizer:
    """
    Tokenizer specialized for biblical texts and theological content.
    
    This tokenizer handles special cases in biblical text including:
    - Bible verse references (e.g., "John 3:16", "Genesis 1:1-3")
    - Ancient language terms (Hebrew, Greek, Aramaic)
    - Theological terminology
    - Special punctuation and formatting in biblical texts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the biblical tokenizer.
        
        Args:
            config_path: Path to tokenizer configuration JSON file
        """
        self.config = self._load_config(config_path)
        self._initialize_patterns()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load tokenizer configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing tokenizer configuration
        """
        default_config = {
            "preserve_verse_refs": True,
            "split_sentences": True,
            "lowercase": False,
            "remove_punctuation": False,
            "theological_terms_path": None,
            "ancient_terms_path": None,
            "special_tokens": {
                "VERSE_REF": "[VERSE_REF]",
                "HEBREW": "[HEBREW]",
                "GREEK": "[GREEK]",
                "ARAMAIC": "[ARAMAIC]",
                "THEOLOGICAL_TERM": "[THEO_TERM]",
                "UNKNOWN": "[UNK]"
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    for key, value in loaded_config.items():
                        if key in default_config:
                            default_config[key] = value
                logger.info(f"Loaded tokenizer config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default tokenizer configuration")
        
        return default_config
    
    def _initialize_patterns(self):
        """Initialize regex patterns for special token detection."""
        # Bible verse reference pattern
        # Matches patterns like "John 3:16", "Genesis 1:1-10", "1 Cor. 13:4-7", etc.
        self.verse_pattern = re.compile(
            r'\b(?:'
            r'(?:(?:1|2|3|I|II|III)\s*)?'  # Optional book number
            r'(?:Genesis|Gen|Exodus|Ex|Exod|Leviticus|Lev|Numbers|Num|Deuteronomy|Deut|Joshua|Josh|'
            r'Judges|Judg|Ruth|1\s*Samuel|1\s*Sam|2\s*Samuel|2\s*Sam|1\s*Kings|2\s*Kings|'
            r'1\s*Chronicles|1\s*Chron|2\s*Chronicles|2\s*Chron|Ezra|Nehemiah|Neh|Esther|Est|'
            r'Job|Psalms?|Ps|Proverbs|Prov|Ecclesiastes|Eccl|Song\s*of\s*Solomon|Song|Isaiah|Isa|'
            r'Jeremiah|Jer|Lamentations|Lam|Ezekiel|Ezek|Daniel|Dan|Hosea|Hos|Joel|Amos|'
            r'Obadiah|Obad|Jonah|Jon|Micah|Mic|Nahum|Nah|Habakkuk|Hab|Zephaniah|Zeph|'
            r'Haggai|Hag|Zechariah|Zech|Malachi|Mal|Matthew|Matt|Mark|Luke|John|Jn|'
            r'Acts|Romans|Rom|1\s*Corinthians|1\s*Cor|2\s*Corinthians|2\s*Cor|Galatians|Gal|'
            r'Ephesians|Eph|Philippians|Phil|Colossians|Col|1\s*Thessalonians|1\s*Thess|'
            r'2\s*Thessalonians|2\s*Thess|1\s*Timothy|1\s*Tim|2\s*Timothy|2\s*Tim|Titus|'
            r'Philemon|Phlm|Hebrews|Heb|James|Jas|1\s*Peter|1\s*Pet|2\s*Peter|2\s*Pet|'
            r'1\s*John|2\s*John|3\s*John|Jude|Revelation|Rev)'
            r'\s+'  # Space between book name and chapter
            r'(\d+):'  # Chapter number followed by colon
            r'(\d+)'  # Verse number
            r'(?:-(\d+))?'  # Optional ending verse (for ranges)
            r')\b'
        )
        
        # Load theological terms if path is provided
        self.theological_terms = set()
        if self.config.get("theological_terms_path"):
            try:
                with open(self.config["theological_terms_path"], 'r', encoding='utf-8') as f:
                    self.theological_terms = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.theological_terms)} theological terms")
            except Exception as e:
                logger.warning(f"Could not load theological terms: {e}")
        
        # Load ancient language terms if path is provided
        self.ancient_terms = {}
        if self.config.get("ancient_terms_path"):
            try:
                with open(self.config["ancient_terms_path"], 'r', encoding='utf-8') as f:
                    self.ancient_terms = json.load(f)
                logger.info(f"Loaded ancient language terms: {len(self.ancient_terms.get('hebrew', []))} Hebrew, "
                           f"{len(self.ancient_terms.get('greek', []))} Greek, "
                           f"{len(self.ancient_terms.get('aramaic', []))} Aramaic")
            except Exception as e:
                logger.warning(f"Could not load ancient language terms: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize biblical text into tokens.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # First, handle special tokens if configured to preserve them
        if self.config["preserve_verse_refs"]:
            # Replace verse references with special tokens to preserve them during tokenization
            text, verse_refs = self._extract_verse_refs(text)
        
        # Split into sentences if configured
        if self.config["split_sentences"]:
            sentences = sent_tokenize(text)
            # Tokenize each sentence
            tokens = []
            for sentence in sentences:
                tokens.extend(word_tokenize(sentence))
        else:
            # Tokenize the whole text at once
            tokens = word_tokenize(text)
        
        # Restore verse references if they were extracted
        if self.config["preserve_verse_refs"]:
            tokens = self._restore_verse_refs(tokens, verse_refs)
        
        # Handle other special cases
        tokens = self._handle_special_cases(tokens)
        
        # Apply additional processing based on config
        if self.config["lowercase"]:
            tokens = [token.lower() for token in tokens]
            
        if self.config["remove_punctuation"]:
            tokens = [token for token in tokens if not all(c in '.,;:!?"\'()[]{}' for c in token)]
        
        return tokens
    
    def _extract_verse_refs(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract verse references from text and replace with placeholders.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (modified text, list of extracted verse references)
        """
        verse_refs = []
        positions = []
        
        # Find all verse references
        for match in self.verse_pattern.finditer(text):
            verse_refs.append(match.group(0))
            positions.append((match.start(), match.end()))
        
        # Replace verse references with placeholders, starting from the end
        # to preserve the positions of earlier references
        modified_text = text
        for i in range(len(positions) - 1, -1, -1):
            start, end = positions[i]
            placeholder = f" {self.config['special_tokens']['VERSE_REF']} "
            modified_text = modified_text[:start] + placeholder + modified_text[end:]
            
        return modified_text, verse_refs
    
    def _restore_verse_refs(self, tokens: List[str], verse_refs: List[str]) -> List[str]:
        """
        Restore verse references in tokenized output.
        
        Args:
            tokens: List of tokens
            verse_refs: List of verse references to restore
            
        Returns:
            Tokens with verse references restored
        """
        restored_tokens = []
        ref_index = 0
        
        for token in tokens:
            if token == self.config['special_tokens']['VERSE_REF'] and ref_index < len(verse_refs):
                # Replace placeholder with the actual verse reference
                restored_tokens.append(verse_refs[ref_index])
                ref_index += 1
            else:
                restored_tokens.append(token)
                
        return restored_tokens
    
    def _handle_special_cases(self, tokens: List[str]) -> List[str]:
        """
        Process tokens to handle special biblical text cases.
        
        Args:
            tokens: List of tokens to process
            
        Returns:
            Processed tokens
        """
        processed_tokens = []
        
        for token in tokens:
            # Check if token is a theological term
            if self.theological_terms and token in self.theological_terms:
                if self.config.get("mark_theological_terms", False):
                    processed_tokens.append(f"{self.config['special_tokens']['THEOLOGICAL_TERM']}_{token}")
                else:
                    processed_tokens.append(token)
                continue
                
            # Check if token is an ancient language term
            is_ancient = False
            for lang, terms in self.ancient_terms.items():
                if token in terms:
                    lang_token = self.config['special_tokens'].get(lang.upper(), f"[{lang.upper()}]")
                    processed_tokens.append(f"{lang_token}_{token}")
                    is_ancient = True
                    break
            
            if not is_ancient:
                processed_tokens.append(token)
                
        return processed_tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            Reconstructed text
        """
        # Basic detokenization - join with spaces and fix punctuation
        text = ' '.join(tokens)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        
        # Fix spacing around quotes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Fix special tokens
        for token_type, token_value in self.config['special_tokens'].items():
            if token_type != 'UNKNOWN':  # Skip UNK token
                pattern = f"{token_value}_([a-zA-Z0-9]+)"
                text = re.sub(pattern, r'\1', text)
        
        return text
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized texts
        """
        return [self.tokenize(text) for text in texts]
    
    def save_vocabulary(self, output_path: str, corpus: Optional[List[str]] = None):
        """
        Generate and save vocabulary from a corpus or from loaded resources.
        
        Args:
            output_path: Path to save vocabulary file
            corpus: Optional corpus of texts to generate vocabulary from
        """
        vocabulary = set()
        
        # Add special tokens
        for token in self.config['special_tokens'].values():
            vocabulary.add(token)
            
        # Add theological terms
        vocabulary.update(self.theological_terms)
        
        # Add ancient language terms
        for terms in self.ancient_terms.values():
            vocabulary.update(terms)
            
        # Process corpus if provided
        if corpus:
            for text in corpus:
                tokens = self.tokenize(text)
                vocabulary.update(tokens)
                
        # Save vocabulary to file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in sorted(vocabulary):
                f.write(f"{word}\n")
                
        logger.info(f"Saved vocabulary with {len(vocabulary)} tokens to {output_path}")


def create_tokenizer(config_path: Optional[str] = None) -> BiblicalTokenizer:
    """
    Factory function to create a BiblicalTokenizer instance.
    
    Args:
        config_path: Optional path to tokenizer configuration
        
    Returns:
        Initialized BiblicalTokenizer instance
    """
    return BiblicalTokenizer(config_path)


if __name__ == "__main__":
    # Example usage
    tokenizer = create_tokenizer()
    
    test_text = ("In John 3:16, Jesus said 'For God so loved the world.' "
                "This illustrates the theological concept of atonement. "
                "The Hebrew word 'shalom' means peace.")
    
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    reconstructed = tokenizer.detokenize(tokens)
    print(f"Reconstructed: {reconstructed}")
>>>>>>> b4faba0fb5c52afa16531b76e44ad827a4e5bf68
