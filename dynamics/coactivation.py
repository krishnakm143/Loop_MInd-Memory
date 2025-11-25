"""Coactivation Diffusion Engine.

Implements spreading activation through the memory graph:
- When memories are retrieved together, their connections strengthen
- Activation spreads to related memories (like neural spreading activation)
- Builds associative memory chains over time
- Implements Hebbian learning: "memories that fire together, wire together"
"""

import math
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


# Coactivation configuration
COACTIVATION_BOOST = 0.1          # Strength boost when co-retrieved
DIFFUSION_DECAY = 0.5             # Decay factor per hop (0.5 = halves each hop)
MAX_DIFFUSION_HOPS = 3            # Maximum hops for spreading activation
MIN_ACTIVATION_THRESHOLD = 0.05   # Minimum activation to propagate
MAX_EDGE_WEIGHT = 1.0             # Maximum edge weight
EDGE_WEIGHT_GROWTH_RATE = 0.08    # How much edges strengthen per coactivation


@dataclass
class ActivationNode:
    """Node with activation level for spreading activation."""
    node_id: str
    activation: float
    depth: int
    source_path: List[str] = field(default_factory=list)


@dataclass
class CoactivationUpdate:
    """Record of a coactivation strength update."""
    from_id: str
    to_id: str
    old_weight: float
    new_weight: float
    relationship: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CoactivationEngine:
    """Engine for coactivation diffusion through memory graph."""
    
    def __init__(
        self,
        graph_store=None,
        boost_amount: float = COACTIVATION_BOOST,
        diffusion_decay: float = DIFFUSION_DECAY,
        max_hops: int = MAX_DIFFUSION_HOPS,
    ):
        """Initialize coactivation engine.
        
        Args:
            graph_store: Neo4j graph store instance
            boost_amount: Strength boost per coactivation
            diffusion_decay: Decay factor per hop
            max_hops: Maximum diffusion hops
        """
        self._graph_store = graph_store
        self.boost_amount = boost_amount
        self.diffusion_decay = diffusion_decay
        self.max_hops = max_hops
    
    @property
    def graph_store(self):
        """Get graph store (lazy initialization)."""
        if self._graph_store is None:
            from core.storage.graph import get_graph_store
            self._graph_store = get_graph_store()
        return self._graph_store
    
    async def record_coactivation(
        self,
        memory_ids: List[str],
        context: str = "retrieval",
    ) -> List[CoactivationUpdate]:
        """Record that memories were co-activated (retrieved together).
        
        Creates/strengthens COACTIVATES_WITH edges between all pairs.
        
        Args:
            memory_ids: List of memory IDs that were co-retrieved
            context: Context of coactivation (retrieval, review, etc.)
            
        Returns:
            List of edge updates made
        """
        if len(memory_ids) < 2:
            return []
        
        updates = []
        
        # Create coactivation edges between all pairs
        for i, from_id in enumerate(memory_ids):
            for to_id in memory_ids[i + 1:]:
                try:
                    update = await self._strengthen_edge(
                        from_id=from_id,
                        to_id=to_id,
                        rel_type="COACTIVATES_WITH",
                        context=context,
                    )
                    if update:
                        updates.append(update)
                except Exception as e:
                    logger.warning(
                        "Failed to record coactivation",
                        from_id=from_id,
                        to_id=to_id,
                        error=str(e),
                    )
        
        logger.info(
            "Coactivation recorded",
            memory_count=len(memory_ids),
            edges_updated=len(updates),
            context=context,
        )
        
        return updates
    
    async def _strengthen_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        context: str = "retrieval",
    ) -> Optional[CoactivationUpdate]:
        """Strengthen or create an edge between memories.
        
        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            rel_type: Relationship type
            context: Coactivation context
            
        Returns:
            Update record or None
        """
        # Get current edge weight (if exists)
        current_weight = await self._get_edge_weight(from_id, to_id, rel_type)
        old_weight = current_weight if current_weight else 0.3  # Default starting weight
        
        # Calculate new weight with diminishing returns
        # Uses logistic growth: weight approaches MAX_EDGE_WEIGHT asymptotically
        growth_remaining = MAX_EDGE_WEIGHT - old_weight
        boost = EDGE_WEIGHT_GROWTH_RATE * growth_remaining
        new_weight = min(MAX_EDGE_WEIGHT, old_weight + boost)
        
        # Update edge in graph
        success = await self._update_edge(
            from_id=from_id,
            to_id=to_id,
            rel_type=rel_type,
            new_weight=new_weight,
        )
        
        if success:
            return CoactivationUpdate(
                from_id=from_id,
                to_id=to_id,
                old_weight=old_weight,
                new_weight=new_weight,
                relationship=rel_type,
            )
        
        return None
    
    async def _get_edge_weight(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
    ) -> Optional[float]:
        """Get current edge weight between memories.
        
        Args:
            from_id: Source ID
            to_id: Target ID
            rel_type: Relationship type
            
        Returns:
            Edge weight or None if edge doesn't exist
        """
        try:
            query = f"""
            MATCH (a:Memory {{memory_id: $from_id}})-[r:{rel_type}]-(b:Memory {{memory_id: $to_id}})
            RETURN r.weight as weight
            """
            result = await self.graph_store.execute_query(
                query,
                {"from_id": from_id, "to_id": to_id}
            )
            if result and len(result) > 0:
                return result[0].get("weight")
        except Exception:
            pass
        return None
    
    async def _update_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        new_weight: float,
    ) -> bool:
        """Update or create edge with new weight.
        
        Args:
            from_id: Source ID
            to_id: Target ID
            rel_type: Relationship type
            new_weight: New weight value
            
        Returns:
            Success flag
        """
        try:
            await self.graph_store.create_hsg_edge(
                from_memory_id=from_id,
                to_node_id=to_id,
                to_node_type="Memory",  # Memory-to-Memory coactivation
                rel_type=rel_type,
                weight=new_weight,
                properties={
                    "last_coactivation": datetime.utcnow().isoformat(),
                }
            )
            return True
        except Exception as e:
            logger.warning("Failed to update edge", error=str(e))
            return False
    
    async def spread_activation(
        self,
        seed_memories: List[str],
        initial_activation: float = 1.0,
    ) -> Dict[str, float]:
        """Spread activation from seed memories through the graph.
        
        Implements spreading activation where:
        - Seed memories start with full activation
        - Activation propagates to neighbors
        - Activation decays with distance
        - Nodes can accumulate activation from multiple sources
        
        Args:
            seed_memories: Starting memory IDs
            initial_activation: Starting activation level
            
        Returns:
            Dict mapping memory_id -> final activation level
        """
        activations: Dict[str, float] = {}
        visited: Set[str] = set()
        queue: List[ActivationNode] = []
        
        # Initialize seeds
        for memory_id in seed_memories:
            activations[memory_id] = initial_activation
            queue.append(ActivationNode(
                node_id=memory_id,
                activation=initial_activation,
                depth=0,
                source_path=[memory_id],
            ))
        
        # BFS spreading activation
        while queue:
            current = queue.pop(0)
            
            if current.node_id in visited:
                continue
            if current.depth >= self.max_hops:
                continue
            if current.activation < MIN_ACTIVATION_THRESHOLD:
                continue
            
            visited.add(current.node_id)
            
            # Get neighbors
            try:
                neighbors = await self.graph_store.get_neighbors(
                    memory_id=current.node_id,
                    rel_types=["RELATED_TO", "COACTIVATES_WITH", "SEQUENCE_OF"],
                    hops=1,
                    limit=20,
                )
                
                for neighbor in neighbors:
                    neighbor_id = neighbor.get("neighbor_id")
                    if not neighbor_id or neighbor.get("neighbor_type") != "Memory":
                        continue
                    
                    # Calculate propagated activation
                    edge_weight = neighbor.get("weight", 0.5)
                    propagated = current.activation * self.diffusion_decay * edge_weight
                    
                    # Accumulate activation
                    current_activation = activations.get(neighbor_id, 0.0)
                    new_activation = current_activation + propagated
                    activations[neighbor_id] = min(1.0, new_activation)
                    
                    if neighbor_id not in visited:
                        queue.append(ActivationNode(
                            node_id=neighbor_id,
                            activation=propagated,
                            depth=current.depth + 1,
                            source_path=current.source_path + [neighbor_id],
                        ))
                        
            except Exception as e:
                logger.warning(
                    "Failed to get neighbors for spreading activation",
                    node_id=current.node_id,
                    error=str(e),
                )
        
        logger.debug(
            "Spreading activation completed",
            seed_count=len(seed_memories),
            activated_count=len(activations),
            max_activation=max(activations.values()) if activations else 0,
        )
        
        return activations
    
    async def get_activated_memories(
        self,
        seed_memories: List[str],
        min_activation: float = 0.1,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """Get memories activated by spreading from seeds.
        
        Args:
            seed_memories: Starting memory IDs
            min_activation: Minimum activation to include
            limit: Maximum results
            
        Returns:
            List of (memory_id, activation) tuples, sorted by activation
        """
        activations = await self.spread_activation(seed_memories)
        
        # Filter and sort
        results = [
            (mem_id, act) 
            for mem_id, act in activations.items()
            if act >= min_activation
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def calculate_association_strength(
        self,
        coactivation_count: int,
        total_retrievals_a: int,
        total_retrievals_b: int,
    ) -> float:
        """Calculate association strength between two memories.
        
        Uses PMI-like formula:
        strength = log(P(A,B) / (P(A) * P(B)))
        
        Args:
            coactivation_count: Times retrieved together
            total_retrievals_a: Total retrievals of memory A
            total_retrievals_b: Total retrievals of memory B
            
        Returns:
            Association strength (higher = stronger association)
        """
        if coactivation_count == 0 or total_retrievals_a == 0 or total_retrievals_b == 0:
            return 0.0
        
        # Avoid division by zero with smoothing
        total = max(total_retrievals_a, total_retrievals_b) + 1
        
        p_joint = coactivation_count / total
        p_a = total_retrievals_a / total
        p_b = total_retrievals_b / total
        
        if p_a * p_b <= 0:
            return 0.0
        
        # PMI with capping
        pmi = math.log(p_joint / (p_a * p_b))
        
        # Normalize to 0-1 range (using sigmoid)
        return 1.0 / (1.0 + math.exp(-pmi))


# Global engine instance
_engine: Optional[CoactivationEngine] = None


def get_coactivation_engine() -> CoactivationEngine:
    """Get or create global CoactivationEngine instance."""
    global _engine
    if _engine is None:
        _engine = CoactivationEngine()
    return _engine


async def record_coactivation(
    memory_ids: List[str],
    context: str = "retrieval",
) -> Dict[str, Any]:
    """Record coactivation between memories.
    
    Args:
        memory_ids: Co-activated memory IDs
        context: Coactivation context
        
    Returns:
        Update summary
    """
    engine = get_coactivation_engine()
    updates = await engine.record_coactivation(memory_ids, context)
    
    return {
        "memories_linked": len(memory_ids),
        "edges_updated": len(updates),
        "updates": [
            {
                "from": u.from_id,
                "to": u.to_id,
                "old_weight": u.old_weight,
                "new_weight": u.new_weight,
            }
            for u in updates
        ],
    }


async def get_associated_memories(
    seed_memory_ids: List[str],
    min_activation: float = 0.1,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get memories associated with seeds via spreading activation.
    
    Args:
        seed_memory_ids: Starting memory IDs
        min_activation: Minimum activation threshold
        limit: Maximum results
        
    Returns:
        Associated memories with activation levels
    """
    engine = get_coactivation_engine()
    results = await engine.get_activated_memories(
        seed_memories=seed_memory_ids,
        min_activation=min_activation,
        limit=limit,
    )
    
    return [
        {"memory_id": mem_id, "activation": activation}
        for mem_id, activation in results
    ]


async def spread_activation_from_query(
    retrieved_ids: List[str],
    boost_factor: float = 0.2,
) -> Dict[str, float]:
    """Spread activation from query results.
    
    Used to boost retrieval by finding associatively related memories.
    
    Args:
        retrieved_ids: Initially retrieved memory IDs
        boost_factor: How much to boost newly activated memories
        
    Returns:
        Dict of memory_id -> activation boost
    """
    engine = get_coactivation_engine()
    activations = await engine.spread_activation(retrieved_ids)
    
    # Calculate boost for non-seed memories
    boosts = {}
    for mem_id, activation in activations.items():
        if mem_id not in retrieved_ids:
            boosts[mem_id] = activation * boost_factor
    
    return boosts
