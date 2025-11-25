"""Decay Engine for Memory Strength.

Implements memory decay similar to human memory:
- Memories fade over time if not accessed
- Decay follows exponential curve (Ebbinghaus forgetting curve)
- Decay can be reversed by access/retrieval (strengthening)
- Different decay rates for different sectors
"""

import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


# Decay rate constants (per day) - higher = faster decay
SECTOR_DECAY_RATES = {
    "episodic": 0.015,      # Personal events decay slowly
    "semantic": 0.008,      # Facts/knowledge very stable
    "emotional": 0.012,     # Emotional memories persist
    "procedural": 0.005,    # Skills/procedures most stable
    "workspace": 0.025,     # Work context decays faster
}

# Default decay if sector unknown
DEFAULT_DECAY_RATE = 0.012

# Minimum strength (memories below this can be archived)
MIN_STRENGTH = 0.1

# Strength boost on access
ACCESS_BOOST = 0.15

# Maximum strength
MAX_STRENGTH = 1.0


@dataclass
class DecayResult:
    """Result of decay calculation."""
    memory_id: str
    original_strength: float
    new_strength: float
    decay_applied: float
    days_since_access: float
    should_archive: bool
    sector: str


class DecayEngine:
    """Engine for calculating and applying memory decay."""
    
    def __init__(
        self,
        custom_decay_rates: Optional[Dict[str, float]] = None,
        min_strength: float = MIN_STRENGTH,
        access_boost: float = ACCESS_BOOST,
    ):
        """Initialize decay engine.
        
        Args:
            custom_decay_rates: Override default sector decay rates
            min_strength: Minimum strength before archiving
            access_boost: Strength boost on memory access
        """
        self.decay_rates = {**SECTOR_DECAY_RATES}
        if custom_decay_rates:
            self.decay_rates.update(custom_decay_rates)
        
        self.min_strength = min_strength
        self.access_boost = access_boost
    
    def calculate_decay(
        self,
        current_strength: float,
        last_accessed: datetime,
        sector: str,
        importance: float = 0.5,
        access_count: int = 1,
    ) -> float:
        """Calculate decayed strength for a memory.
        
        Uses modified Ebbinghaus forgetting curve:
        S(t) = S(0) * e^(-λt/r)
        
        Where:
        - S(t) = strength at time t
        - S(0) = initial strength
        - λ = decay rate
        - t = time since last access (days)
        - r = retention factor (based on importance + access count)
        
        Args:
            current_strength: Current memory strength (0-1)
            last_accessed: When memory was last accessed
            sector: Memory sector (affects decay rate)
            importance: Memory importance (0-1), slows decay
            access_count: Number of times accessed, slows decay
            
        Returns:
            New decayed strength (0-1)
        """
        if current_strength <= 0:
            return 0.0
        
        # Calculate days since last access
        now = datetime.utcnow()
        if isinstance(last_accessed, str):
            try:
                last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                if last_accessed.tzinfo is not None:
                    last_accessed = last_accessed.replace(tzinfo=None)
            except (ValueError, AttributeError):
                last_accessed = now
        
        delta = now - last_accessed
        days_since = delta.total_seconds() / 86400  # Convert to days
        
        if days_since <= 0:
            return current_strength
        
        # Get sector decay rate
        decay_rate = self.decay_rates.get(sector, DEFAULT_DECAY_RATE)
        
        # Calculate retention factor
        # Higher importance and access count = slower decay
        importance_factor = 1.0 + importance * 0.5  # 1.0 to 1.5
        access_factor = math.log(access_count + 1) * 0.3 + 1.0  # 1.0 to ~2.0
        retention = importance_factor * access_factor
        
        # Apply exponential decay with retention factor
        effective_rate = decay_rate / retention
        new_strength = current_strength * math.exp(-effective_rate * days_since)
        
        # Clamp to valid range
        return max(0.0, min(MAX_STRENGTH, new_strength))
    
    def apply_access_boost(
        self,
        current_strength: float,
        access_type: str = "retrieve",
    ) -> float:
        """Boost memory strength on access (retrieval/update).
        
        Args:
            current_strength: Current memory strength
            access_type: Type of access (retrieve, update, review)
            
        Returns:
            New strength after boost
        """
        # Different boosts for different access types
        boost_multipliers = {
            "retrieve": 1.0,      # Normal retrieval
            "update": 1.5,        # Memory was updated
            "review": 1.2,        # Explicit review
            "link": 0.8,          # Linked to another memory
            "coactivate": 0.5,    # Co-activated with other memory
        }
        
        multiplier = boost_multipliers.get(access_type, 1.0)
        boost = self.access_boost * multiplier
        
        new_strength = current_strength + boost
        return min(MAX_STRENGTH, new_strength)
    
    def calculate_batch_decay(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[DecayResult]:
        """Calculate decay for multiple memories.
        
        Args:
            memories: List of memory dicts with required fields:
                - memory_id
                - strength or importance
                - last_accessed
                - sector
                
        Returns:
            List of decay results
        """
        results = []
        
        for memory in memories:
            memory_id = memory.get("memory_id", "")
            original_strength = memory.get("strength", memory.get("importance", 0.5))
            last_accessed = memory.get("last_accessed", memory.get("updated_at", datetime.utcnow()))
            sector = memory.get("sector", memory.get("type", "semantic"))
            importance = memory.get("importance", 0.5)
            access_count = memory.get("access_count", 1)
            
            new_strength = self.calculate_decay(
                current_strength=original_strength,
                last_accessed=last_accessed,
                sector=sector,
                importance=importance,
                access_count=access_count,
            )
            
            decay_applied = original_strength - new_strength
            
            # Calculate days since access for reporting
            now = datetime.utcnow()
            if isinstance(last_accessed, str):
                try:
                    last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                    if last_accessed.tzinfo is not None:
                        last_accessed = last_accessed.replace(tzinfo=None)
                except (ValueError, AttributeError):
                    last_accessed = now
            days_since = (now - last_accessed).total_seconds() / 86400
            
            results.append(DecayResult(
                memory_id=memory_id,
                original_strength=original_strength,
                new_strength=new_strength,
                decay_applied=decay_applied,
                days_since_access=days_since,
                should_archive=new_strength < self.min_strength,
                sector=sector,
            ))
        
        return results
    
    def get_archivable_memories(
        self,
        decay_results: List[DecayResult],
    ) -> List[str]:
        """Get memory IDs that should be archived.
        
        Args:
            decay_results: Results from calculate_batch_decay
            
        Returns:
            List of memory IDs to archive
        """
        return [r.memory_id for r in decay_results if r.should_archive]
    
    def estimate_half_life(
        self,
        sector: str,
        importance: float = 0.5,
        access_count: int = 1,
    ) -> float:
        """Estimate memory half-life (days until 50% strength).
        
        Args:
            sector: Memory sector
            importance: Memory importance
            access_count: Access count
            
        Returns:
            Estimated half-life in days
        """
        decay_rate = self.decay_rates.get(sector, DEFAULT_DECAY_RATE)
        
        importance_factor = 1.0 + importance * 0.5
        access_factor = math.log(access_count + 1) * 0.3 + 1.0
        retention = importance_factor * access_factor
        
        effective_rate = decay_rate / retention
        
        # Half-life formula: t_half = ln(2) / λ
        return math.log(2) / effective_rate if effective_rate > 0 else float("inf")


# Global engine instance
_engine: Optional[DecayEngine] = None


def get_decay_engine() -> DecayEngine:
    """Get or create global DecayEngine instance."""
    global _engine
    if _engine is None:
        _engine = DecayEngine()
    return _engine


async def calculate_memory_decay(
    memory_id: str,
    current_strength: float,
    last_accessed: datetime,
    sector: str,
    importance: float = 0.5,
    access_count: int = 1,
) -> Dict[str, Any]:
    """Calculate decay for a single memory.
    
    Args:
        memory_id: Memory ID
        current_strength: Current strength
        last_accessed: Last access time
        sector: Memory sector
        importance: Importance score
        access_count: Access count
        
    Returns:
        Decay calculation result
    """
    engine = get_decay_engine()
    
    new_strength = engine.calculate_decay(
        current_strength=current_strength,
        last_accessed=last_accessed,
        sector=sector,
        importance=importance,
        access_count=access_count,
    )
    
    return {
        "memory_id": memory_id,
        "original_strength": current_strength,
        "new_strength": new_strength,
        "decay_applied": current_strength - new_strength,
        "should_archive": new_strength < MIN_STRENGTH,
    }


async def boost_memory_strength(
    memory_id: str,
    current_strength: float,
    access_type: str = "retrieve",
) -> Dict[str, Any]:
    """Boost memory strength on access.
    
    Args:
        memory_id: Memory ID
        current_strength: Current strength
        access_type: Type of access
        
    Returns:
        New strength info
    """
    engine = get_decay_engine()
    new_strength = engine.apply_access_boost(current_strength, access_type)
    
    return {
        "memory_id": memory_id,
        "original_strength": current_strength,
        "new_strength": new_strength,
        "boost_applied": new_strength - current_strength,
    }


async def run_decay_sweep(
    memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run decay calculation on batch of memories.
    
    Args:
        memories: List of memory dicts
        
    Returns:
        Sweep results with statistics
    """
    engine = get_decay_engine()
    results = engine.calculate_batch_decay(memories)
    archivable = engine.get_archivable_memories(results)
    
    total_decay = sum(r.decay_applied for r in results)
    avg_decay = total_decay / len(results) if results else 0
    
    logger.info(
        "Decay sweep completed",
        total_memories=len(memories),
        archivable_count=len(archivable),
        avg_decay=avg_decay,
    )
    
    return {
        "total_processed": len(results),
        "archivable_count": len(archivable),
        "archivable_ids": archivable,
        "average_decay": avg_decay,
        "results": [
            {
                "memory_id": r.memory_id,
                "original_strength": r.original_strength,
                "new_strength": r.new_strength,
                "should_archive": r.should_archive,
            }
            for r in results
        ],
    }
