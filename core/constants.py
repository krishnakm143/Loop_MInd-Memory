"""Application constants."""

from enum import Enum


class MemoryType(str, Enum):
    """Memory type/sector enumeration (internal use only).
    
    These types are automatically determined by the Sector Classifier.
    Users do not need to know about these types.
    """
    
    EPISODIC = "episodic"  # Personal experiences, events, "what happened"
    SEMANTIC = "semantic"  # Facts, knowledge, "what is"
    EMOTIONAL = "emotional"  # Feelings, sentiments, emotional states
    PROCEDURAL = "procedural"  # How-to, processes, steps
    WORKSPACE = "workspace"  # Project-specific, workspace context
    
    @classmethod
    def values(cls) -> list[str]:
        """Get all valid memory type values."""
        return [item.value for item in cls]
    
    @classmethod
    def default(cls) -> str:
        """Get default memory type (used as placeholder until classification)."""
        return cls.EPISODIC.value

