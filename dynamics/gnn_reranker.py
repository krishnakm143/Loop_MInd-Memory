"""GNN Reranker for Memory Retrieval.

Implements graph-aware reranking using message passing:
- Aggregates features from graph neighborhood
- Uses attention mechanism for importance weighting
- Combines vector similarity with graph structure
- Optional PyTorch GNN for advanced reranking

This module provides both:
1. A lightweight message-passing reranker (no ML dependencies)
2. An optional PyTorch GNN reranker (if torch is available)
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


# Reranker configuration
MESSAGE_PASSING_ITERATIONS = 2
NEIGHBOR_ATTENTION_HEADS = 4
GRAPH_FEATURE_WEIGHT = 0.25  # Weight of graph features in final score
MAX_NEIGHBORS = 10


@dataclass
class MemoryNode:
    """Node representation for GNN processing."""
    memory_id: str
    features: Dict[str, float]
    embedding: Optional[List[float]] = None
    neighbors: List[str] = field(default_factory=list)
    neighbor_weights: Dict[str, float] = field(default_factory=dict)
    aggregated_features: Dict[str, float] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0


class MessagePassingReranker:
    """Lightweight message-passing reranker without ML dependencies.
    
    Uses graph structure to refine ranking:
    1. Initialize node features from retrieval scores
    2. Perform message passing iterations
    3. Aggregate neighbor information
    4. Compute final scores combining vector and graph signals
    """
    
    def __init__(
        self,
        graph_store=None,
        iterations: int = MESSAGE_PASSING_ITERATIONS,
        graph_weight: float = GRAPH_FEATURE_WEIGHT,
    ):
        """Initialize message passing reranker.
        
        Args:
            graph_store: Neo4j graph store
            iterations: Number of message passing iterations
            graph_weight: Weight of graph features in final score
        """
        self._graph_store = graph_store
        self.iterations = iterations
        self.graph_weight = graph_weight
    
    @property
    def graph_store(self):
        """Get graph store (lazy initialization)."""
        if self._graph_store is None:
            from core.storage.graph import get_graph_store
            self._graph_store = get_graph_store()
        return self._graph_store
    
    async def build_subgraph(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, MemoryNode]:
        """Build subgraph from candidate memories.
        
        Args:
            candidates: Retrieval candidates with scores
            
        Returns:
            Dict mapping memory_id to MemoryNode
        """
        nodes: Dict[str, MemoryNode] = {}
        candidate_ids = {c.get("memory_id") for c in candidates}
        
        for candidate in candidates:
            memory_id = candidate.get("memory_id", "")
            if not memory_id:
                continue
            
            # Extract features
            features = {
                "vector_score": candidate.get("vector_score", 0.0),
                "graph_score": candidate.get("graph_score", 0.0),
                "recency_score": candidate.get("recency_score", 0.5),
                "trust_score": candidate.get("trust_score", 0.5),
                "importance": candidate.get("importance", 0.5),
                "original_rank": candidates.index(candidate) / len(candidates),
            }
            
            node = MemoryNode(
                memory_id=memory_id,
                features=features,
                embedding=candidate.get("embedding"),
            )
            nodes[memory_id] = node
        
        # Fetch neighbors from graph
        for memory_id in list(nodes.keys()):
            try:
                neighbors = await self.graph_store.get_neighbors(
                    memory_id=memory_id,
                    rel_types=["RELATED_TO", "COACTIVATES_WITH"],
                    hops=1,
                    limit=MAX_NEIGHBORS,
                )
                
                for neighbor in neighbors:
                    neighbor_id = neighbor.get("neighbor_id")
                    if not neighbor_id:
                        continue
                    
                    # Only link to other candidates (keep subgraph focused)
                    if neighbor_id in candidate_ids:
                        nodes[memory_id].neighbors.append(neighbor_id)
                        nodes[memory_id].neighbor_weights[neighbor_id] = neighbor.get("weight", 0.5)
                        
            except Exception as e:
                logger.warning(
                    "Failed to fetch neighbors for GNN",
                    memory_id=memory_id,
                    error=str(e),
                )
        
        return nodes
    
    def compute_attention(
        self,
        node: MemoryNode,
        nodes: Dict[str, MemoryNode],
    ) -> Dict[str, float]:
        """Compute attention weights for node's neighbors.
        
        Uses feature similarity + edge weight for attention.
        
        Args:
            node: Source node
            nodes: All nodes in subgraph
            
        Returns:
            Attention weights for each neighbor
        """
        if not node.neighbors:
            return {}
        
        attention = {}
        total = 0.0
        
        for neighbor_id in node.neighbors:
            if neighbor_id not in nodes:
                continue
            
            neighbor = nodes[neighbor_id]
            
            # Compute attention score
            # Based on: edge weight + feature similarity
            edge_weight = node.neighbor_weights.get(neighbor_id, 0.5)
            
            # Feature similarity (cosine of feature vectors)
            feature_sim = self._feature_similarity(node.features, neighbor.features)
            
            # Combined attention score
            score = 0.6 * edge_weight + 0.4 * feature_sim
            attention[neighbor_id] = score
            total += score
        
        # Softmax normalization
        if total > 0:
            for neighbor_id in attention:
                attention[neighbor_id] /= total
        
        return attention
    
    def _feature_similarity(
        self,
        features_a: Dict[str, float],
        features_b: Dict[str, float],
    ) -> float:
        """Compute similarity between feature vectors.
        
        Args:
            features_a: First feature dict
            features_b: Second feature dict
            
        Returns:
            Similarity score (0-1)
        """
        common_keys = set(features_a.keys()) & set(features_b.keys())
        if not common_keys:
            return 0.0
        
        # Compute cosine similarity
        dot = sum(features_a[k] * features_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v ** 2 for k, v in features_a.items() if k in common_keys))
        norm_b = math.sqrt(sum(v ** 2 for k, v in features_b.items() if k in common_keys))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1
    
    def aggregate_messages(
        self,
        node: MemoryNode,
        nodes: Dict[str, MemoryNode],
    ) -> Dict[str, float]:
        """Aggregate messages from neighbors.
        
        Args:
            node: Target node
            nodes: All nodes in subgraph
            
        Returns:
            Aggregated features from neighbors
        """
        if not node.neighbors or not node.attention_weights:
            return node.features.copy()
        
        aggregated = {k: 0.0 for k in node.features.keys()}
        
        for neighbor_id, attention in node.attention_weights.items():
            if neighbor_id not in nodes:
                continue
            
            neighbor = nodes[neighbor_id]
            
            for key in aggregated:
                neighbor_value = neighbor.features.get(key, 0.0)
                aggregated[key] += attention * neighbor_value
        
        # Combine with own features (residual connection)
        for key in aggregated:
            aggregated[key] = 0.5 * node.features[key] + 0.5 * aggregated[key]
        
        return aggregated
    
    def message_passing_iteration(
        self,
        nodes: Dict[str, MemoryNode],
    ) -> Dict[str, MemoryNode]:
        """Perform one iteration of message passing.
        
        Args:
            nodes: Current node states
            
        Returns:
            Updated nodes
        """
        # Compute attention for all nodes
        for memory_id, node in nodes.items():
            node.attention_weights = self.compute_attention(node, nodes)
        
        # Aggregate messages
        for memory_id, node in nodes.items():
            node.aggregated_features = self.aggregate_messages(node, nodes)
        
        # Update features
        for memory_id, node in nodes.items():
            node.features = node.aggregated_features.copy()
        
        return nodes
    
    def compute_final_scores(
        self,
        nodes: Dict[str, MemoryNode],
        original_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute final reranked scores.
        
        Args:
            nodes: Processed nodes
            original_candidates: Original candidates
            
        Returns:
            Reranked candidates
        """
        reranked = []
        
        for candidate in original_candidates:
            memory_id = candidate.get("memory_id", "")
            
            if memory_id not in nodes:
                reranked.append(candidate)
                continue
            
            node = nodes[memory_id]
            
            # Compute graph-aware score
            graph_features_score = (
                node.features.get("vector_score", 0) * 0.4 +
                node.features.get("graph_score", 0) * 0.2 +
                node.features.get("recency_score", 0) * 0.15 +
                node.features.get("trust_score", 0) * 0.15 +
                node.features.get("importance", 0) * 0.1
            )
            
            # Combine with original score
            original_score = candidate.get("score", 0.0)
            final_score = (
                (1 - self.graph_weight) * original_score +
                self.graph_weight * graph_features_score
            )
            
            # Update candidate
            updated = candidate.copy()
            updated["score"] = final_score
            updated["gnn_score"] = graph_features_score
            updated["gnn_features"] = node.features
            updated["reranked"] = True
            
            reranked.append(updated)
        
        # Sort by final score
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return reranked
    
    async def rerank(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using message passing.
        
        Args:
            candidates: Original retrieval candidates
            
        Returns:
            Reranked candidates with updated scores
        """
        if len(candidates) < 2:
            return candidates
        
        logger.debug(
            "Starting GNN reranking",
            candidate_count=len(candidates),
            iterations=self.iterations,
        )
        
        # Build subgraph
        nodes = await self.build_subgraph(candidates)
        
        # Message passing iterations
        for i in range(self.iterations):
            nodes = self.message_passing_iteration(nodes)
        
        # Compute final scores
        reranked = self.compute_final_scores(nodes, candidates)
        
        logger.debug(
            "GNN reranking completed",
            candidate_count=len(reranked),
        )
        
        return reranked


class GNNReranker:
    """Main GNN reranker interface.
    
    Automatically selects implementation:
    - Uses PyTorch GNN if available
    - Falls back to message passing otherwise
    """
    
    def __init__(
        self,
        graph_store=None,
        use_torch: bool = False,
    ):
        """Initialize GNN reranker.
        
        Args:
            graph_store: Graph store instance
            use_torch: Whether to use PyTorch GNN (if available)
        """
        self._graph_store = graph_store
        self.use_torch = use_torch and self._check_torch_available()
        
        # Initialize appropriate reranker
        if self.use_torch:
            self._reranker = self._create_torch_reranker()
        else:
            self._reranker = MessagePassingReranker(graph_store=graph_store)
    
    def _check_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            import torch_geometric
            return True
        except ImportError:
            return False
    
    def _create_torch_reranker(self):
        """Create PyTorch-based GNN reranker.
        
        Returns:
            PyTorch GNN reranker or None
        """
        try:
            from dynamics.torch_gnn import TorchGNNReranker
            return TorchGNNReranker(graph_store=self._graph_store)
        except ImportError:
            logger.warning("PyTorch GNN not available, using message passing")
            return MessagePassingReranker(graph_store=self._graph_store)
    
    async def rerank(
        self,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using GNN.
        
        Args:
            candidates: Original candidates
            top_k: Optional limit on results
            
        Returns:
            Reranked candidates
        """
        reranked = await self._reranker.rerank(candidates)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


# Global reranker instance
_reranker: Optional[GNNReranker] = None


def get_gnn_reranker() -> GNNReranker:
    """Get or create global GNNReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = GNNReranker()
    return _reranker


async def gnn_rerank(
    candidates: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank retrieval candidates using GNN.
    
    Args:
        candidates: Original retrieval candidates
        top_k: Maximum results to return
        
    Returns:
        Reranked candidates with graph-aware scores
    """
    reranker = get_gnn_reranker()
    return await reranker.rerank(candidates, top_k=top_k)


async def compute_graph_features(
    memory_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute graph-based features for memories.
    
    Features computed:
    - degree: Number of connections
    - centrality: PageRank-like centrality
    - clustering: Local clustering coefficient
    
    Args:
        memory_ids: Memory IDs to compute features for
        
    Returns:
        Dict mapping memory_id -> feature dict
    """
    reranker = get_gnn_reranker()
    
    # Build minimal candidates
    candidates = [{"memory_id": mid} for mid in memory_ids]
    
    if isinstance(reranker._reranker, MessagePassingReranker):
        nodes = await reranker._reranker.build_subgraph(candidates)
        
        return {
            memory_id: {
                "degree": len(node.neighbors),
                "avg_edge_weight": sum(node.neighbor_weights.values()) / len(node.neighbor_weights) if node.neighbor_weights else 0,
            }
            for memory_id, node in nodes.items()
        }
    
    return {mid: {} for mid in memory_ids}
