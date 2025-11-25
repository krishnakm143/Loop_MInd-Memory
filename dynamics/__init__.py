"""Memory Dynamics Module.

This module implements cognitive-inspired memory dynamics:

1. **Decay Engine** - Memories fade over time if not accessed
   - Exponential decay based on Ebbinghaus forgetting curve
   - Different decay rates per sector
   - Access boosts reverse decay

2. **Coactivation Diffusion** - Related memories strengthen each other
   - "Memories that fire together, wire together" (Hebbian learning)
   - Spreading activation through graph
   - Edge weights strengthen with co-retrieval

3. **GNN Reranker** - Graph-aware retrieval ranking
   - Message passing through memory graph
   - Attention-based neighbor aggregation
   - Optional PyTorch GNN support
"""

from dynamics.decay_engine import (
    DecayEngine,
    DecayResult,
    get_decay_engine,
    calculate_memory_decay,
    boost_memory_strength,
    run_decay_sweep,
)

from dynamics.coactivation import (
    CoactivationEngine,
    CoactivationUpdate,
    ActivationNode,
    get_coactivation_engine,
    record_coactivation,
    get_associated_memories,
    spread_activation_from_query,
)

from dynamics.gnn_reranker import (
    GNNReranker,
    MessagePassingReranker,
    MemoryNode,
    get_gnn_reranker,
    gnn_rerank,
    compute_graph_features,
)

__all__ = [
    # Decay Engine
    "DecayEngine",
    "DecayResult",
    "get_decay_engine",
    "calculate_memory_decay",
    "boost_memory_strength",
    "run_decay_sweep",
    # Coactivation
    "CoactivationEngine",
    "CoactivationUpdate",
    "ActivationNode",
    "get_coactivation_engine",
    "record_coactivation",
    "get_associated_memories",
    "spread_activation_from_query",
    # GNN Reranker
    "GNNReranker",
    "MessagePassingReranker",
    "MemoryNode",
    "get_gnn_reranker",
    "gnn_rerank",
    "compute_graph_features",
]
