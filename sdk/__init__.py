"""LoopMind Python SDK.

Easy-to-use client library for AI agents to interact with the LoopMind Memory System.

Usage:
    from loopmind import LoopMind
    
    client = LoopMind(api_key="your-api-key")
    
    # Store a memory
    client.remember("I met John at the coffee shop today")
    
    # Recall memories
    results = client.recall("Who did I meet?")
"""

from sdk.loopmind_client import (
    LoopMind,
    LoopMindClient,
    MemoryResult,
    QueryResult,
    ActionResult,
    LoopMindError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
)

__all__ = [
    "LoopMind",
    "LoopMindClient",
    "MemoryResult",
    "QueryResult",
    "ActionResult",
    "LoopMindError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
]

__version__ = "1.0.0"

