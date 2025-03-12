import re
from typing import Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
import logging

from src.utils.logger import get_logger
from src.utils.verse_utils import parse_verse_reference

logger = get_logger(__name__)

class TheologicalChecker:
    """
    Utility for verifying theological accuracy and checking doctrinal alignment
    of generated content.
    """
    
    def __init__(self, resources_path: str = "data/processed/theological_resources"):
        """
        Initialize the theological checker.
        
        Args:
            resources_path: Path to theological resources
        """
        self.resources_path = Path(resources_path)
        
        # Load theological resources
        self.heresy_patterns = self._load_heresy_patterns()
        self.doctrinal_statements = self._load_doctrinal_statements()
        self.theological_lexicon = self._load_theological_lexicon()
        self.denominational_views = self._load_denominational_views()
        
        logger.info("TheologicalChecker initialized")
    
    def _load_heresy_patterns(self) -> List[Dict]:
        """
        Load patterns for detecting common theological heresies.
        
        Returns:
            List of heresy patterns with regex and explanation
        """
        try:
            file_path = self.resources_path / "heresy_patterns.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Heresy patterns file not found: {file_path}")
                # Fallback to basic patterns
                return [
                    {
                        "name": "Modalism",
                        "pattern": r"\b(God|Jesus|Holy Spirit)\s+(?:is|are)\s+(?:just|merely|only)\s+(?:different|various)\s+(?:modes|forms|manifestations)\b",
                        "explanation": "Modalism incorrectly teaches that Father, Son, and Holy Spirit are not distinct persons but merely different modes or manifestations of God."
                    },
                    {
                        "name": "Arianism",
                        "pattern": r"\bJesus\s+(?:is|was)\s+(?:created|made|not eternal|not fully divine)\b",
                        "explanation": "Arianism incorrectly teaches that Jesus Christ is not fully divine and was created by God the Father."
                    },
                    {
                        "name": "Pelagianism",
                        "pattern": r"\b(?:humans?|people|mankind|man)\s+can\s+(?:achieve|attain|earn|work for)\s+salvation\s+(?:without|apart from)\s+(?:God'?s)?\s+grace\b",
                        "explanation": "Pelagianism incorrectly teaches that humans can achieve salvation through their own efforts without God's grace."
                    }
                ]
        except Exception as e:
            logger.error(f"Error loading heresy patterns: {e}")
            return []
    
    def _load_doctrinal_statements(self) -> Dict[str, Dict]:
        """
        Load common doctrinal statements or creeds.
        
        Returns:
            Dictionary of doctrinal statements by tradition/denomination
        """
        try:
            file_path = self.resources_path / "doctrinal_statements.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Doctrinal statements file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading doctrinal statements: {e}")
            return {}
    
    def _load_theological_lexicon(self) -> Dict[str, Dict]:
        """
        Load theological terms and their definitions.
        
        Returns:
            Dictionary of theological terms and definitions
        """
        try:
            file_path = self.resources_path / "theological_lexicon.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Theological lexicon file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading theological lexicon: {e}")
            return {}
    
    def _load_denominational_views(self) -> Dict[str, Dict]:
        """
        Load denominational perspectives on various theological topics.
        
        Returns:
            Dictionary of denominational views by topic
        """
        try:
            file_path = self.resources_path / "denominational_views.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Denominational views file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading denominational views: {e}")
            return {}
    
    def check_for_heresies(self, text: str) -> List[Dict]:
        """
        Check for common theological heresies in the given text.
        
        Args:
            text: Text to check for heresies
            
        Returns:
            List of detected heresies with explanation and matched text
        """
        detected_heresies = []
        
        for heresy in self.heresy_patterns:
            pattern = re.compile(heresy["pattern"], re.IGNORECASE)
            matches = pattern.finditer(text)
            
            for match in matches:
                detected_heresies.append({
                    "name": heresy["name"],
                    "explanation": heresy["explanation"],
                    "matched_text": match.group(0),
                    "position": match.span()
                })
        
        return detected_heresies
    
    def validate_doctrinal_alignment(self, 
                                   text: str, 
                                   denomination: Optional[str] = None) -> Dict:
        """
        Validate if the text aligns with the doctrinal position of a specific denomination.
        
        Args:
            text: Text to validate
            denomination: Denomination to check against (if None, checks against common beliefs)
            
        Returns:
            Dictionary with validation results
        """
        if denomination and denomination not in self.denominational_views:
            logger.warning(f"Denomination not found in resources: {denomination}")
            denomination = None
        
        # Check for heresies first
        heresies = self.check_for_heresies(text)
        
        # Check doctrinal alignment
        alignment_issues = []
        
        if denomination:
            # Check against specific denominational beliefs
            for topic, view in self.denominational_views.get(denomination, {}).items():
                # Simple pattern matching for contradiction
                contradiction_pattern = re.compile(view.get("contradiction_pattern", ""), re.IGNORECASE)
                if contradiction_pattern.search(text):
                    alignment_issues.append({
                        "topic": topic,
                        "denomination_view": view.get("summary", ""),
                        "issue": f"Potential contradiction with {denomination} view on {topic}"
                    })
        
        return {
            "heresies": heresies,
            "alignment_issues": alignment_issues,
            "is_valid": len(heresies) == 0 and len(alignment_issues) == 0
        }
    
    def suggest_scripture_support(self, text: str) -> List[Dict]:
        """
        Suggest Bible verses that could support statements made in the text.
        
        Args:
            text: Text to analyze for scriptural support
            
        Returns:
            List of suggested scripture references with relevance score
        """
        # This would ideally use embeddings or a knowledge base
        # For now, implement a simple keyword-based approach
        scripture_suggestions = []
        
        # Example implementation with a few key theological concepts
        theological_concepts = {
            "salvation by grace": ["Ephesians 2:8-9", "Romans 3:23-24", "Titus 3:5"],
            "trinity": ["Matthew 28:19", "2 Corinthians 13:14", "John 1:1-14"],
            "resurrection": ["1 Corinthians 15:3-8", "Romans 6:4-5", "John 11:25-26"],
            "sin nature": ["Romans 5:12", "Psalm 51:5", "Jeremiah 17:9"],
            "redemption": ["Galatians 3:13", "Titus 2:14", "1 Peter 1:18-19"],
            "justification": ["Romans 5:1", "Galatians 2:16", "Romans 3:28"]
        }
        
        for concept, verses in theological_concepts.items():
            if re.search(r'\b' + re.escape(concept) + r'\b', text, re.IGNORECASE):
                for verse in verses:
                    scripture_suggestions.append({
                        "concept": concept,
                        "reference": verse,
                        "relevance": 0.8  # Placeholder score
                    })
        
        return scripture_suggestions
    
    def identify_theological_terms(self, text: str) -> List[Dict]:
        """
        Identify theological terms used in the text and provide definitions.
        
        Args:
            text: Text to analyze for theological terms
            
        Returns:
            List of identified terms with definitions
        """
        identified_terms = []
        
        for term, info in self.theological_lexicon.items():
            # Use word boundary to match whole words
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                identified_terms.append({
                    "term": term,
                    "definition": info.get("definition", ""),
                    "tradition": info.get("tradition", "general")
                })
        
        return identified_terms
    
    def explain_denominational_differences(self, topic: str) -> Dict[str, str]:
        """
        Explain how different denominations view a specific theological topic.
        
        Args:
            topic: Theological topic to explain
            
        Returns:
            Dictionary mapping denominations to their views on the topic
        """
        views = {}
        
        for denomination, topics in self.denominational_views.items():
            if topic in topics:
                views[denomination] = topics[topic].get("summary", "")
        
        return views
    
    def rate_theological_accuracy(self, text: str) -> Dict:
        """
        Rate the overall theological accuracy of the text.
        
        Args:
            text: Text to rate
            
        Returns:
            Dictionary with accuracy assessment
        """
        # Check for heresies
        heresies = self.check_for_heresies(text)
        
        # Identify theological terms
        terms = self.identify_theological_terms(text)
        
        # Calculate a simple accuracy score (0-100)
        score = 100
        
        # Deduct points for heresies
        score -= len(heresies) * 20
        
        # Bonus for proper use of theological terms
        score += min(len(terms) * 5, 20)
        
        # Ensure score is within 0-100
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "heresies": heresies,
            "theological_terms": terms,
            "assessment": self._get_assessment_label(score)
        }
    
    def _get_assessment_label(self, score: int) -> str:
        """Get a text label for a theological accuracy score."""
        if score >= 90:
            return "Excellent theological accuracy"
        elif score >= 80:
            return "Good theological accuracy with minor issues"
        elif score >= 70:
            return "Acceptable theological accuracy with some concerns"
        elif score >= 50:
            return "Significant theological issues present"
        else:
            return "Major theological errors or heresies detected"