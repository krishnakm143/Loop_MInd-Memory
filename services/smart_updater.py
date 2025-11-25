"""Smart Updater service for memory deduplication and linking.

Implements the MERGE/LINK/CREATE/FLAG decision logic:
- Uses hybrid scoring (vector + graph + recency + trust + importance)
- Decides whether to merge, link, or create new memories
- Handles conflicts and contradictions
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import structlog

from core.storage.vector import VectorStore
from core.storage.graph import get_graph_store
from services.embedding_router import get_embeddings_batch
from api.config import settings

logger = structlog.get_logger()


# Thresholds for decision making
MERGE_THRESHOLD = 0.86      # Nearly identical - merge content
LINK_THRESHOLD = 0.72       # Related - create link
FLAG_THRESHOLD = 0.90       # Very high similarity but contradicting - flag for review

# Recency decay parameters
RECENCY_DECAY_DAYS = 90     # Half-life for recency boost

# Scoring weights
WEIGHTS = {
    "cosine_sim": 0.55,     # Vector similarity (most important)
    "graph_sim": 0.15,      # Graph proximity
    "recency": 0.15,        # Time-based boost
    "trust": 0.10,          # Trust score
    "importance": 0.05,     # Importance score
}


class UpdateAction(str, Enum):
    """Possible update actions."""
    MERGE = "merge"         # Combine with existing memory
    LINK = "link"           # Create relationship to existing memory
    CREATE = "create"       # Create as new memory
    FLAG = "flag"           # Flag for review (contradiction)


class SmartUpdater:
    """Smart updater for memory deduplication and linking."""
    
    def __init__(self):
        """Initialize smart updater."""
        self._vector_store: Optional[VectorStore] = None
    
    @property
    def vector_store(self) -> VectorStore:
        """Get vector store (lazy initialization)."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    def calculate_recency_score(
        self,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """Calculate recency score with exponential decay.
        
        Args:
            created_at: When the memory was created
            reference_time: Reference time (defaults to now)
            
        Returns:
            Recency score (0.0-1.0)
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        # Handle timezone-naive vs aware datetimes
        if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
            # Make reference_time aware if created_at is aware
            from datetime import timezone
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        
        days_since = (reference_time - created_at).days
        
        # Exponential decay: e^(-days/decay_constant)
        decay = math.exp(-days_since / RECENCY_DECAY_DAYS)
        
        return max(0.0, min(1.0, decay))
    
    def calculate_graph_similarity(
        self,
        shared_entities: int,
        total_entities: int,
    ) -> float:
        """Calculate graph-based similarity score.
        
        Args:
            shared_entities: Number of shared entities
            total_entities: Total entities in new memory
            
        Returns:
            Graph similarity score (0.0-1.0)
        """
        if total_entities == 0:
            return 0.0
        
        # Jaccard-like similarity
        return min(1.0, shared_entities / total_entities)
    
    def hybrid_score(
        self,
        cosine_sim: float,
        graph_sim: float,
        days_since: float,
        trust: float,
        importance: float,
    ) -> float:
        """Calculate hybrid similarity score.
        
        Args:
            cosine_sim: Cosine similarity from vector search
            graph_sim: Graph-based similarity
            days_since: Days since candidate was created
            trust: Trust score of candidate
            importance: Importance score of candidate
            
        Returns:
            Hybrid score (0.0-1.0)
        """
        # Recency boost with exponential decay
        recency_boost = math.exp(-days_since / RECENCY_DECAY_DAYS)
        
        score = (
            WEIGHTS["cosine_sim"] * cosine_sim +
            WEIGHTS["graph_sim"] * graph_sim +
            WEIGHTS["recency"] * recency_boost +
            WEIGHTS["trust"] * trust +
            WEIGHTS["importance"] * importance
        )
        
        return min(1.0, max(0.0, score))
    
    async def find_candidates(
        self,
        embedding: List[float],
        tenant_id: str,
        sector: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find candidate memories for potential merge/link.
        
        Args:
            embedding: Query embedding vector
            tenant_id: Tenant ID filter
            sector: Sector to search in
            limit: Maximum candidates to return
            
        Returns:
            List of candidate memories with scores
        """
        try:
            # Search in sector-specific collection
            collection_name = f"memories_{sector}"
            
            results = await self.vector_store.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                filter={
                    "must": [
                        {"key": "tenant_id", "match": {"value": tenant_id}}
                    ]
                },
            )
            
            candidates = []
            for result in results:
                payload = result.get("payload", {})
                candidates.append({
                    "memory_id": payload.get("memory_id"),
                    "cosine_score": result.get("score", 0.0),
                    "trust_score": payload.get("trust_score", 0.5),
                    "importance": payload.get("importance", 0.5),
                    "created_at": payload.get("created_at"),
                    "sector": payload.get("sector", sector),
                })
            
            return candidates
            
        except Exception as e:
            logger.error(
                "Failed to find candidates",
                tenant_id=tenant_id,
                sector=sector,
                error=str(e),
            )
            return []
    
    async def decide_action(
        self,
        new_memory: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        new_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[UpdateAction, Optional[Dict[str, Any]], float]:
        """Decide what action to take for a new memory.
        
        Args:
            new_memory: New memory data
            candidates: Candidate memories from vector search
            new_entities: Entities extracted from new memory
            
        Returns:
            Tuple of (action, best_match, score)
        """
        if not candidates:
            return UpdateAction.CREATE, None, 0.0
        
        best_score = 0.0
        best_match = None
        
        now = datetime.utcnow()
        entity_count = len(new_entities) if new_entities else 0
        
        for candidate in candidates:
            cosine_sim = candidate.get("cosine_score", 0.0)
            trust = candidate.get("trust_score", 0.5)
            importance = candidate.get("importance", 0.5)
            
            # Calculate days since creation
            created_at_str = candidate.get("created_at")
            if created_at_str:
                try:
                    if isinstance(created_at_str, str):
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    else:
                        created_at = created_at_str
                    days_since = (now - created_at.replace(tzinfo=None)).days
                except Exception:
                    days_since = 30  # Default assumption
            else:
                days_since = 30
            
            # Graph similarity (would need to query graph for shared entities)
            # For now, use a simple heuristic based on cosine similarity
            graph_sim = cosine_sim * 0.5  # Approximate
            
            # Calculate hybrid score
            score = self.hybrid_score(
                cosine_sim=cosine_sim,
                graph_sim=graph_sim,
                days_since=days_since,
                trust=trust,
                importance=importance,
            )
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Determine action based on score
        if best_score >= MERGE_THRESHOLD:
            action = UpdateAction.MERGE
        elif best_score >= LINK_THRESHOLD:
            action = UpdateAction.LINK
        else:
            action = UpdateAction.CREATE
        
        logger.info(
            "Update action decided",
            action=action.value,
            best_score=round(best_score, 3),
            candidate_count=len(candidates),
            best_match_id=best_match.get("memory_id") if best_match else None,
        )
        
        return action, best_match, best_score
    
    async def check_contradiction(
        self,
        new_content: str,
        existing_content: str,
    ) -> bool:
        """Check if two pieces of content contradict each other.
        
        Args:
            new_content: New memory content
            existing_content: Existing memory content
            
        Returns:
            True if contradiction detected
        """
        # Simple heuristic: check for negation patterns
        # In production, would use an LLM or NLI model
        
        negation_words = {"not", "no", "never", "didn't", "doesn't", "won't", "isn't", "aren't"}
        
        new_lower = new_content.lower()
        existing_lower = existing_content.lower()
        
        # Count negations
        new_negations = sum(1 for word in negation_words if word in new_lower)
        existing_negations = sum(1 for word in negation_words if word in existing_lower)
        
        # If one has significantly more negations, might be contradiction
        if abs(new_negations - existing_negations) >= 2:
            return True
        
        return False


# Global smart updater instance
_smart_updater: Optional[SmartUpdater] = None


def get_smart_updater() -> SmartUpdater:
    """Get or create the global SmartUpdater instance."""
    global _smart_updater
    if _smart_updater is None:
        _smart_updater = SmartUpdater()
    return _smart_updater


async def decide_update_action(
    content: str,
    embedding: List[float],
    tenant_id: str,
    sector: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    candidate_limit: int = 10,
) -> Dict[str, Any]:
    """Main entry point for deciding update action.
    
    Args:
        content: New memory content
        embedding: Content embedding
        tenant_id: Tenant ID
        sector: Memory sector
        entities: Extracted entities
        candidate_limit: Max candidates to consider
        
    Returns:
        Decision result with action, match, and score
    """
    updater = get_smart_updater()
    
    # Find candidates
    candidates = await updater.find_candidates(
        embedding=embedding,
        tenant_id=tenant_id,
        sector=sector,
        limit=candidate_limit,
    )
    
    # Decide action
    new_memory = {"content": content, "sector": sector}
    action, match, score = await updater.decide_action(
        new_memory=new_memory,
        candidates=candidates,
        new_entities=entities,
    )
    
    result = {
        "action": action.value,
        "score": round(score, 3),
        "match": match,
        "candidate_count": len(candidates),
    }
    
    # Check for contradiction if merging
    if action == UpdateAction.MERGE and match:
        # Would need to fetch existing content for full check
        # For now, skip contradiction check
        pass
    
    return result


async def merge_memories(
    new_memory_id: str,
    existing_memory_id: str,
    new_content: Dict[str, Any],
    existing_content: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two memories into one.
    
    Args:
        new_memory_id: New memory ID
        existing_memory_id: Existing memory ID
        new_content: New memory content
        existing_content: Existing memory content
        
    Returns:
        Merged memory data
    """
    # Create merged content
    merged = {
        "primary_id": existing_memory_id,
        "merged_ids": [new_memory_id],
        "content": {
            "original": existing_content,
            "merged": new_content,
        },
        "merged_at": datetime.utcnow().isoformat(),
    }
    
    logger.info(
        "Memories merged",
        existing_id=existing_memory_id,
        new_id=new_memory_id,
    )
    
    return merged


def calculate_hybrid_score(
    cosine_sim: float,
    graph_sim: float,
    days_since: float,
    trust: float,
    importance: float,
) -> float:
    """Calculate hybrid score (convenience function).
    
    Args:
        cosine_sim: Cosine similarity
        graph_sim: Graph similarity
        days_since: Days since creation
        trust: Trust score
        importance: Importance score
        
    Returns:
        Hybrid score (0.0-1.0)
    """
    updater = get_smart_updater()
    return updater.hybrid_score(cosine_sim, graph_sim, days_since, trust, importance)

