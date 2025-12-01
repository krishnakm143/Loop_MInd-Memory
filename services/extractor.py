"""Entity and triple extraction service.

Extracts structured information from memory content:
- Named entities (people, locations, dates, concepts, organizations)
- Subject-predicate-object triples
- Temporal facts with time bounds
- Importance scoring based on entity density and specificity
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter
import structlog

logger = structlog.get_logger()


# Temporal predicates that represent facts that can change over time
TEMPORAL_PREDICATES = {
    "is", "was", "are", "were", "has", "have", "had",
    "works", "worked", "lives", "lived", "owns", "owned",
    "manages", "managed", "leads", "led", "reports", "reported",
    "costs", "cost", "earns", "earned", "makes", "made",
}


# Entity type patterns
ENTITY_PATTERNS = {
    # Date patterns
    "date": [
        # ISO format: 2024-01-15
        re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        # US format: 01/15/2024 or 1/15/2024
        re.compile(r'\b(\d{1,2}/\d{1,2}/\d{4})\b'),
        # Written format: January 15, 2024 or Jan 15, 2024
        re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})\b', re.IGNORECASE),
        # Relative dates
        re.compile(r'\b(yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))\b', re.IGNORECASE),
    ],
    # Time patterns
    "time": [
        # 24-hour: 14:30 or 14:30:00
        re.compile(r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b'),
        # 12-hour: 2:30 PM or 2:30pm
        re.compile(r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b'),
        # Written: 3 PM, 3pm
        re.compile(r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b'),
    ],
    # Email patterns
    "email": [
        re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),
    ],
    # URL patterns
    "url": [
        re.compile(r'(https?://[^\s<>"{}|\\^`\[\]]+)'),
    ],
    # Phone patterns
    "phone": [
        re.compile(r'\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
    ],
    # Money patterns
    "money": [
        re.compile(r'(\$\d+(?:,\d{3})*(?:\.\d{2})?)'),
        re.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP|euros?|pounds?))', re.IGNORECASE),
    ],
    # Percentage patterns
    "percentage": [
        re.compile(r'\b(\d+(?:\.\d+)?%)\b'),
    ],
}

# Patterns for extracting proper nouns (potential people/organizations/locations)
PROPER_NOUN_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

# Common words that look like proper nouns but aren't
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night',
}

# Location indicators (words that suggest following word is a location)
LOCATION_INDICATORS = {'in', 'at', 'to', 'from', 'near', 'around', 'visited', 'went'}

# Person indicators (words that suggest following word is a person)
PERSON_INDICATORS = {'met', 'with', 'called', 'emailed', 'told', 'asked', 'spoke', 'mr', 'mrs', 'ms', 'dr'}

# Organization indicators
ORG_INDICATORS = {'company', 'corporation', 'inc', 'ltd', 'llc', 'corp', 'organization', 'institute', 'university'}

# Triple extraction patterns (subject-predicate-object)
TRIPLE_PATTERNS = [
    # "X is Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*([^.!?]+)', re.IGNORECASE),
    # "X has Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:has|have|had)\s+(?:a|an|the)?\s*([^.!?]+)', re.IGNORECASE),
    # "X works at Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works?|worked)\s+(?:at|for)\s+([^.!?]+)', re.IGNORECASE),
    # "X lives in Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:lives?|lived)\s+in\s+([^.!?]+)', re.IGNORECASE),
    # "X went to Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:went|goes?|going)\s+to\s+([^.!?]+)', re.IGNORECASE),
    # "X met Y" pattern
    re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+met\s+(?:with\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
]


class Extractor:
    """Entity and triple extractor from text content."""
    
    def __init__(self):
        """Initialize extractor."""
        self.entity_patterns = ENTITY_PATTERNS
        self.triple_patterns = TRIPLE_PATTERNS
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text.
        
        Args:
            text: Text content to extract from
            
        Returns:
            List of entity dictionaries with name, type, and confidence
        """
        if not text:
            return []
        
        entities = []
        seen_normalized = set()
        
        # Extract pattern-based entities (dates, emails, URLs, etc.)
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    # Handle tuple matches from groups
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                    
                    if not match:
                        continue
                    
                    normalized = match.strip().lower()
                    if normalized not in seen_normalized:
                        seen_normalized.add(normalized)
                        entities.append({
                            "name": match.strip(),
                            "type": entity_type,
                            "confidence": 0.9,  # Pattern-based = high confidence
                        })
        
        # Extract proper nouns (potential people/organizations/locations)
        proper_nouns = PROPER_NOUN_PATTERN.findall(text)
        
        for noun in proper_nouns:
            normalized = noun.strip().lower()
            
            # Skip common words and already extracted entities
            if normalized in COMMON_WORDS or normalized in seen_normalized:
                continue
            
            # Skip single-letter matches
            if len(noun.strip()) < 2:
                continue
            
            # Determine entity type based on context
            entity_type = self._classify_proper_noun(text, noun)
            
            seen_normalized.add(normalized)
            entities.append({
                "name": noun.strip(),
                "type": entity_type,
                "confidence": 0.7,  # Proper noun = medium confidence
            })
        
        logger.debug(
            "Entities extracted",
            text_length=len(text),
            entity_count=len(entities),
        )
        
        return entities
    
    def _classify_proper_noun(self, text: str, noun: str) -> str:
        """Classify a proper noun as person, location, or organization.
        
        Args:
            text: Full text for context
            noun: The proper noun to classify
            
        Returns:
            Entity type string
        """
        text_lower = text.lower()
        noun_lower = noun.lower()
        
        # Find context around the noun
        noun_pos = text_lower.find(noun_lower)
        if noun_pos > 0:
            # Get word before the noun
            words_before = text_lower[:noun_pos].split()
            if words_before:
                last_word = words_before[-1].strip('.,!?:;')
                
                if last_word in LOCATION_INDICATORS:
                    return "location"
                if last_word in PERSON_INDICATORS:
                    return "person"
        
        # Check for organization indicators in the noun itself
        if any(ind in noun_lower for ind in ORG_INDICATORS):
            return "organization"
        
        # Default to concept (general named entity)
        return "concept"
    
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """Extract subject-predicate-object triples from text.
        
        Args:
            text: Text content to extract from
            
        Returns:
            List of triple dictionaries with subject, predicate, object
        """
        if not text:
            return []
        
        triples = []
        seen_keys = set()
        
        for pattern in self.triple_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    subject = match[0].strip()
                    obj = match[1].strip()[:100]  # Limit object length
                    
                    # Determine predicate from pattern
                    predicate = self._extract_predicate(text, subject, obj)
                    
                    # Create unique key
                    key = f"{subject.lower()}:{predicate}:{obj.lower()}"
                    
                    if key not in seen_keys and subject and obj:
                        seen_keys.add(key)
                        triples.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                        })
        
        logger.debug(
            "Triples extracted",
            text_length=len(text),
            triple_count=len(triples),
        )
        
        return triples
    
    def _extract_predicate(self, text: str, subject: str, obj: str) -> str:
        """Extract the predicate between subject and object.
        
        Args:
            text: Full text
            subject: Subject entity
            obj: Object entity
            
        Returns:
            Predicate string
        """
        text_lower = text.lower()
        
        # Find the text between subject and object
        subj_pos = text_lower.find(subject.lower())
        obj_pos = text_lower.find(obj.lower())
        
        if subj_pos >= 0 and obj_pos > subj_pos:
            between = text[subj_pos + len(subject):obj_pos].strip()
            
            # Extract verb phrase
            words = between.split()
            verbs = []
            for word in words:
                word = word.strip('.,!?:;').lower()
                if word and word not in {'a', 'an', 'the', 'to', 'at', 'in', 'on', 'for', 'with'}:
                    verbs.append(word)
            
            if verbs:
                return "_".join(verbs[:3])  # Limit to first 3 words
        
        return "related_to"
    
    def extract_temporal_facts(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Extract temporal facts from triples and context.
        
        Identifies facts that represent time-bound truths that can evolve.
        
        Args:
            text: Original text for context
            entities: Extracted entities (for dates)
            triples: Extracted triples
            
        Returns:
            List of temporal fact candidates
        """
        if not triples:
            return []
        
        temporal_facts = []
        
        # Extract date entities for temporal context
        date_entities = [e for e in entities if e.get("type") == "date"]
        reference_date = None
        
        # Parse first date found as reference
        if date_entities:
            date_str = date_entities[0].get("name", "")
            reference_date = self._parse_date(date_str)
        
        for triple in triples:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            
            if not (subject and predicate and obj):
                continue
            
            # Check if this is a temporal predicate
            predicate_words = predicate.replace("_", " ").split()
            is_temporal = any(w in TEMPORAL_PREDICATES for w in predicate_words)
            
            # Normalize predicate to standard form
            normalized_predicate = self._normalize_predicate(predicate)
            
            # Determine confidence based on extraction quality
            confidence = 0.8 if is_temporal else 0.6
            
            temporal_facts.append({
                "subject": subject,
                "predicate": normalized_predicate,
                "object": obj,
                "valid_from": reference_date or datetime.utcnow(),
                "confidence": confidence,
                "is_temporal": is_temporal,
                "source_type": "extracted",
                "metadata": {
                    "original_predicate": predicate,
                    "extraction_method": "triple_to_fact",
                },
            })
        
        logger.debug(
            "Temporal facts extracted",
            triple_count=len(triples),
            temporal_fact_count=len(temporal_facts),
        )
        
        return temporal_facts
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Try to parse a date string into a datetime object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        # Handle relative dates
        date_lower = date_str.lower().strip()
        now = datetime.utcnow()
        
        if date_lower == "today":
            return now
        elif date_lower == "yesterday":
            return now.replace(day=now.day - 1)
        elif date_lower == "tomorrow":
            return now.replace(day=now.day + 1)
        
        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to a standard form.
        
        Args:
            predicate: Raw predicate string
            
        Returns:
            Normalized predicate (e.g., 'has_CEO', 'works_at')
        """
        predicate = predicate.lower().strip()
        
        # Map common variations to standard forms
        mappings = {
            "is": "is_a",
            "was": "was_a",
            "works": "works_at",
            "worked": "worked_at",
            "lives": "lives_in",
            "lived": "lived_in",
            "has": "has",
            "had": "had",
            "owns": "owns",
            "owned": "owned",
            "manages": "manages",
            "managed": "managed",
            "leads": "leads",
            "led": "led",
        }
        
        # Check for direct mapping
        words = predicate.replace("_", " ").split()
        if words and words[0] in mappings:
            return mappings[words[0]] + ("_" + "_".join(words[1:]) if len(words) > 1 else "")
        
        return predicate.replace(" ", "_")
    
    def calculate_importance(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, str]],
    ) -> float:
        """Calculate importance score based on content richness.
        
        Args:
            text: Original text
            entities: Extracted entities
            triples: Extracted triples
            
        Returns:
            Importance score (0.0-1.0)
        """
        if not text:
            return 0.3
        
        score = 0.3  # Base score
        
        # Entity density bonus (more entities = more important)
        entity_count = len(entities)
        if entity_count > 0:
            # Logarithmic scaling to avoid over-weighting
            entity_bonus = min(0.3, 0.1 * (1 + entity_count / 5))
            score += entity_bonus
        
        # Entity type diversity bonus
        entity_types = set(e.get("type") for e in entities)
        type_bonus = min(0.15, 0.05 * len(entity_types))
        score += type_bonus
        
        # Triple bonus (facts are valuable)
        triple_count = len(triples)
        if triple_count > 0:
            triple_bonus = min(0.2, 0.1 * triple_count)
            score += triple_bonus
        
        # Specific entity types bonus
        for entity in entities:
            e_type = entity.get("type", "")
            if e_type == "date":
                score += 0.05  # Dated memories are useful
            elif e_type == "person":
                score += 0.03  # People references matter
            elif e_type == "location":
                score += 0.02  # Location context
        
        # Cap at 1.0
        score = min(1.0, score)
        
        logger.debug(
            "Importance calculated",
            entity_count=entity_count,
            triple_count=triple_count,
            score=score,
        )
        
        return round(score, 3)


# Global extractor instance
_extractor = Extractor()


async def extract_entities_and_triples(
    text: str,
    calculate_importance_score: bool = True,
    extract_temporal: bool = True,
) -> Dict[str, Any]:
    """Extract entities, triples, temporal facts, and importance from text.
    
    Args:
        text: Text content to extract from
        calculate_importance_score: Whether to calculate importance
        extract_temporal: Whether to extract temporal facts
        
    Returns:
        Extraction results with entities, triples, temporal_facts, and importance
    """
    entities = _extractor.extract_entities(text)
    triples = _extractor.extract_triples(text)
    
    importance = 0.5
    if calculate_importance_score:
        importance = _extractor.calculate_importance(text, entities, triples)
    
    temporal_facts = []
    if extract_temporal:
        temporal_facts = _extractor.extract_temporal_facts(text, entities, triples)
    
    return {
        "entities": entities,
        "triples": triples,
        "temporal_facts": temporal_facts,
        "importance": importance,
    }


async def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities from text.
    
    Args:
        text: Text content
        
    Returns:
        List of entity dictionaries
    """
    return _extractor.extract_entities(text)


async def extract_triples(text: str) -> List[Dict[str, str]]:
    """Extract triples from text.
    
    Args:
        text: Text content
        
    Returns:
        List of triple dictionaries
    """
    return _extractor.extract_triples(text)


def calculate_importance(
    text: str,
    entities: List[Dict[str, Any]],
    triples: List[Dict[str, str]],
) -> float:
    """Calculate importance score.
    
    Args:
        text: Original text
        entities: Extracted entities
        triples: Extracted triples
        
    Returns:
        Importance score (0.0-1.0)
    """
    return _extractor.calculate_importance(text, entities, triples)


async def extract_temporal_facts(
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    triples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Extract temporal facts from text.
    
    Args:
        text: Text content
        entities: Pre-extracted entities (optional, will be extracted if None)
        triples: Pre-extracted triples (optional, will be extracted if None)
        
    Returns:
        List of temporal fact dictionaries
    """
    if entities is None:
        entities = _extractor.extract_entities(text)
    if triples is None:
        triples = _extractor.extract_triples(text)
    
    return _extractor.extract_temporal_facts(text, entities, triples)

