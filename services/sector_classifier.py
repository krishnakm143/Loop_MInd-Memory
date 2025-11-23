"""Sector classification service for memory type detection."""

import re
from typing import Dict, Any, List, Optional
from core.constants import MemoryType
import structlog

logger = structlog.get_logger()

# Rule-based patterns for each sector
EPISODIC_PATTERNS = [
    # Personal experiences and events
    re.compile(r'\b(i|we|they)\s+(went|did|saw|met|visited|attended|experienced)', re.IGNORECASE),
    re.compile(r'\b(today|yesterday|last\s+(week|month|year)|on\s+\w+day)', re.IGNORECASE),
    re.compile(r'\b(happened|occurred|took\s+place|event)', re.IGNORECASE),
    re.compile(r'\b(remember|recall|recollect)\s+(that|when|how)', re.IGNORECASE),
    re.compile(r'\b(met|talked|spoke|conversation)\s+(with|to)', re.IGNORECASE),
]

SEMANTIC_PATTERNS = [
    # Facts and knowledge
    re.compile(r'\b(is|are|was|were)\s+(a|an|the)\s+\w+', re.IGNORECASE),
    re.compile(r'\b(means?|definition|refers?\s+to|is\s+defined\s+as)', re.IGNORECASE),
    re.compile(r'\b(fact|knowledge|information|data|concept)', re.IGNORECASE),
    re.compile(r'\b(according\s+to|states?|says?|defines?)\s+(that|as)', re.IGNORECASE),
    re.compile(r'\b(example|instance|case)\s+of', re.IGNORECASE),
]

EMOTIONAL_PATTERNS = [
    # Feelings and sentiments
    re.compile(r'\b(feel|feeling|felt|emotion|emotional)', re.IGNORECASE),
    re.compile(r'\b(happy|sad|angry|excited|frustrated|anxious|calm|stressed)', re.IGNORECASE),
    re.compile(r'\b(love|hate|like|dislike|enjoy|prefer)', re.IGNORECASE),
    re.compile(r'\b(mood|sentiment|feeling|emotion)', re.IGNORECASE),
    re.compile(r'\b(make\s+me|makes?\s+me\s+feel)', re.IGNORECASE),
]

PROCEDURAL_PATTERNS = [
    # How-to and processes
    re.compile(r'\b(how\s+to|steps?|process|procedure|method)', re.IGNORECASE),
    re.compile(r'\b(first|second|third|then|next|finally|step\s+\d+)', re.IGNORECASE),
    re.compile(r'\b(instructions?|guide|tutorial|walkthrough)', re.IGNORECASE),
    re.compile(r'\b(to\s+do|to\s+make|to\s+create|to\s+build)', re.IGNORECASE),
    re.compile(r'\b(follow|follows?|following)\s+(these|the|these\s+steps)', re.IGNORECASE),
]

WORKSPACE_PATTERNS = [
    # Project and workspace context
    re.compile(r'\b(project|workspace|task|issue|ticket|bug)', re.IGNORECASE),
    re.compile(r'\b(meeting|standup|sprint|milestone|deadline)', re.IGNORECASE),
    re.compile(r'\b(code|repository|repo|branch|commit|pr|pull\s+request)', re.IGNORECASE),
    re.compile(r'\b(team|colleague|collaborator|stakeholder)', re.IGNORECASE),
    re.compile(r'\b(jira|github|gitlab|slack|trello|asana)', re.IGNORECASE),
]


class SectorClassifier:
    """Rule-based sector classifier for memory type detection."""
    
    def __init__(self):
        """Initialize sector classifier."""
        self.patterns = {
            MemoryType.EPISODIC: EPISODIC_PATTERNS,
            MemoryType.SEMANTIC: SEMANTIC_PATTERNS,
            MemoryType.EMOTIONAL: EMOTIONAL_PATTERNS,
            MemoryType.PROCEDURAL: PROCEDURAL_PATTERNS,
            MemoryType.WORKSPACE: WORKSPACE_PATTERNS,
        }
    
    def rule_based_classify(self, content: str) -> Dict[str, Any]:
        """Classify sector using rule-based patterns.
        
        Args:
            content: Memory content to classify
            
        Returns:
            Classification result with sector and confidence
        """
        if not content:
            return {
                "sector": MemoryType.default(),
                "confidence": 0.0,
                "scores": {},
            }
        
        content_lower = content.lower()
        scores: Dict[str, float] = {}
        
        # Count matches for each sector
        for sector, patterns in self.patterns.items():
            match_count = 0
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    match_count += len(matches)
            
            # Normalize score by content length and pattern count
            normalized_score = min(match_count / max(len(content.split()) / 10, 1), 1.0)
            scores[sector.value] = normalized_score
        
        # Find sector with highest score
        if not scores or max(scores.values()) == 0:
            # Default to episodic if no matches
            sector = MemoryType.EPISODIC.value
            confidence = 0.3  # Low confidence for default
        else:
            sector = max(scores, key=scores.get)  # type: ignore
            max_score = scores[sector]
            
            # Calculate confidence based on score dominance
            total_score = sum(scores.values())
            if total_score > 0:
                confidence = max_score / total_score
            else:
                confidence = 0.3
            
            # Boost confidence if score is significantly higher than others
            other_scores = [s for k, s in scores.items() if k != sector]
            if other_scores:
                second_highest = max(other_scores)
                if max_score > second_highest * 1.5:  # 50% higher than second
                    confidence = min(confidence * 1.2, 0.95)
        
        return {
            "sector": sector,
            "confidence": round(confidence, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
        }
    
    def embedding_based_classify(
        self, content: str, embedding: List[float]
    ) -> Dict[str, Any]:
        """Classify sector using embedding similarity (stub for future ML model).
        
        Args:
            content: Memory content
            embedding: Embedding vector
            
        Returns:
            Classification result (currently falls back to rule-based)
        """
        # TODO: Implement embedding-based classification
        # For now, use rule-based classification
        return self.rule_based_classify(content)


# Global classifier instance
_classifier = SectorClassifier()


async def classify_sector(
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Classify memory into sector (episodic, semantic, emotional, procedural, workspace).
    
    Uses rule-first approach with optional embedding-based refinement.
    
    Args:
        content: Memory content to classify
        embedding: Optional embedding vector (for future ML-based classification)
        metadata: Optional metadata (can provide hints like tags, content_type)
        
    Returns:
        Classification result with:
        - sector: Detected sector type
        - confidence: Confidence score [0, 1]
        - scores: Score breakdown for each sector
        - method: Classification method used
    """
    if not content:
        return {
            "sector": MemoryType.default(),
            "confidence": 0.0,
            "scores": {},
            "method": "default",
        }
    
    # Use metadata hints if available
    if metadata:
        # Check if content_type suggests a sector
        content_type = metadata.get("content_type", "")
        if content_type == "code":
            # Code is likely workspace-related
            return {
                "sector": MemoryType.WORKSPACE.value,
                "confidence": 0.7,
                "scores": {MemoryType.WORKSPACE.value: 0.7},
                "method": "metadata_hint",
            }
    
    # Rule-based classification (primary method)
    result = _classifier.rule_based_classify(content)
    result["method"] = "rule_based"
    
    # If embedding is provided, can refine classification (future enhancement)
    if embedding:
        # TODO: Use embedding for refinement
        # For now, embedding-based classification is a stub
        pass
    
    logger.info(
        "Sector classified",
        sector=result["sector"],
        confidence=result["confidence"],
        method=result["method"]
    )
    
    return result
