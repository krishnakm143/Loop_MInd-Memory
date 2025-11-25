"""Hybrid retrieval service for semantic memory search.

Implements the 10-step retrieval pipeline:
1. Embed query using embedding router
2. Route to sector(s) based on query
3. Pre-filter by tenant/tags/date in metadata DB
4. ANN search on Qdrant (top-128)
5. Expand via graph (1-2 hops) to get coactivation neighbors
6. Merge candidate set (deduplicate)
7. Score with hybrid formula
8. Optional: GNN/cross-encoder rerank on top-64
9. Assemble context with provenance trace
10. Return top-K results
"""

import math
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import structlog

from core.storage.vector import VectorStore
from core.storage.graph import get_graph_store, GraphStore
from services.embedding_router import get_embeddings_batch
from services.sector_classifier import classify_sector
from services.hsg_builder import update_coactivation
from api.config import settings

logger = structlog.get_logger()


# Retrieval configuration
DEFAULT_ANN_LIMIT = 128          # Initial ANN search limit
DEFAULT_GRAPH_HOPS = 1           # Graph expansion hops
DEFAULT_FINAL_LIMIT = 10         # Final results to return
RECENCY_DECAY_DAYS = 90          # Recency decay constant

# Hybrid scoring weights
WEIGHTS = {
    "vector": 0.55,              # Vector similarity
    "graph": 0.15,               # Graph proximity
    "recency": 0.15,             # Time-based boost
    "trust": 0.10,               # Trust score
    "importance": 0.05,          # Importance score
}


@dataclass
class RetrievalCandidate:
    """Candidate memory for retrieval."""
    memory_id: str
    vector_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    trust_score: float = 0.5
    importance: float = 0.5
    hybrid_score: float = 0.0
    sector: str = ""
    content: Optional[Dict[str, Any]] = None
    provenance: Optional[List[Dict[str, Any]]] = None
    source: str = "vector"  # vector, graph, or both


class HybridRetriever:
    """Hybrid retriever combining vector search and graph expansion."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_store: Optional VectorStore instance
            graph_store: Optional GraphStore instance
        """
        self._vector_store = vector_store
        self._graph_store = graph_store
    
    @property
    def vector_store(self) -> VectorStore:
        """Get vector store (lazy initialization)."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    @property
    def graph_store(self) -> GraphStore:
        """Get graph store (lazy initialization)."""
        if self._graph_store is None:
            self._graph_store = get_graph_store()
        return self._graph_store
    
    async def embed_query(self, query: str) -> List[float]:
        """Step 1: Embed query text.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        embeddings = await get_embeddings_batch([query])
        return embeddings[0] if embeddings else []
    
    async def route_to_sectors(
        self,
        query: str,
        specified_sectors: Optional[List[str]] = None,
    ) -> List[str]:
        """Step 2: Route query to relevant sectors.
        
        Args:
            query: Query text
            specified_sectors: User-specified sectors (overrides auto-detection)
            
        Returns:
            List of sectors to search
        """
        if specified_sectors:
            return specified_sectors
        
        # Auto-detect sector from query
        classification = await classify_sector(query)
        primary_sector = classification.get("sector", "semantic")
        
        # For queries, often search multiple sectors
        # Return primary + semantic as fallback
        sectors = [primary_sector]
        if primary_sector != "semantic":
            sectors.append("semantic")
        
        return sectors
    
    async def vector_search(
        self,
        query_embedding: List[float],
        tenant_id: str,
        sectors: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = DEFAULT_ANN_LIMIT,
    ) -> List[RetrievalCandidate]:
        """Steps 3-4: Pre-filter and ANN search.
        
        Args:
            query_embedding: Query embedding
            tenant_id: Tenant ID filter
            sectors: Sectors to search
            filters: Additional metadata filters
            limit: Maximum results per sector
            
        Returns:
            List of candidates from vector search
        """
        candidates = []
        seen_ids: Set[str] = set()
        
        for sector in sectors:
            collection_name = f"memories_{sector}"
            
            # Build filter
            search_filter = {
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}}
                ]
            }
            
            # Add custom filters
            if filters:
                for key, value in filters.items():
                    if key in ["tags", "user_id", "status"]:
                        search_filter["must"].append({
                            "key": key,
                            "match": {"value": value}
                        })
            
            try:
                results = await self.vector_store.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit // len(sectors),  # Distribute limit across sectors
                    filter=search_filter if search_filter["must"] else None,
                )
                
                for result in results:
                    payload = result.get("payload", {})
                    memory_id = payload.get("memory_id")
                    
                    if not memory_id or memory_id in seen_ids:
                        continue
                    
                    seen_ids.add(memory_id)
                    
                    # Parse created_at for recency calculation
                    created_at_str = payload.get("created_at")
                    recency_score = self._calculate_recency(created_at_str)
                    
                    candidate = RetrievalCandidate(
                        memory_id=memory_id,
                        vector_score=result.get("score", 0.0),
                        recency_score=recency_score,
                        trust_score=payload.get("trust_score", 0.5),
                        importance=payload.get("importance", 0.5),
                        sector=sector,
                        source="vector",
                    )
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.warning(
                    "Vector search failed for sector",
                    sector=sector,
                    error=str(e),
                )
        
        logger.debug(
            "Vector search completed",
            sectors=sectors,
            candidate_count=len(candidates),
        )
        
        return candidates
    
    async def expand_via_graph(
        self,
        candidates: List[RetrievalCandidate],
        hops: int = DEFAULT_GRAPH_HOPS,
        max_expansion: int = 50,
    ) -> List[RetrievalCandidate]:
        """Step 5: Expand candidates via graph neighbors.
        
        Args:
            candidates: Initial candidates from vector search
            hops: Number of hops for expansion
            max_expansion: Maximum new candidates to add
            
        Returns:
            Expanded candidate list
        """
        if not candidates:
            return candidates
        
        # Get memory IDs from top candidates
        top_ids = [c.memory_id for c in candidates[:20]]  # Expand from top 20
        seen_ids = {c.memory_id for c in candidates}
        
        expanded_candidates = []
        
        for memory_id in top_ids:
            try:
                neighbors = await self.graph_store.get_neighbors(
                    memory_id=memory_id,
                    rel_types=["RELATED_TO", "COACTIVATES_WITH"],
                    hops=hops,
                    limit=10,
                )
                
                for neighbor in neighbors:
                    neighbor_id = neighbor.get("neighbor_id")
                    
                    if not neighbor_id or neighbor_id in seen_ids:
                        continue
                    
                    # Only consider Memory nodes
                    if neighbor.get("neighbor_type") != "Memory":
                        continue
                    
                    seen_ids.add(neighbor_id)
                    
                    # Graph score based on edge weight and relationship type
                    weight = neighbor.get("weight", 0.5)
                    rel_type = neighbor.get("rel_type", "")
                    
                    # Boost coactivation relationships
                    if rel_type == "COACTIVATES_WITH":
                        weight *= 1.2
                    
                    candidate = RetrievalCandidate(
                        memory_id=str(neighbor_id),
                        graph_score=weight,
                        sector=neighbor.get("sector", "semantic"),
                        source="graph",
                    )
                    expanded_candidates.append(candidate)
                    
                    if len(expanded_candidates) >= max_expansion:
                        break
                        
            except Exception as e:
                logger.warning(
                    "Graph expansion failed",
                    memory_id=memory_id,
                    error=str(e),
                )
            
            if len(expanded_candidates) >= max_expansion:
                break
        
        logger.debug(
            "Graph expansion completed",
            original_count=len(candidates),
            expanded_count=len(expanded_candidates),
        )
        
        return candidates + expanded_candidates
    
    def merge_and_deduplicate(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Step 6: Merge and deduplicate candidates.
        
        Args:
            candidates: All candidates (vector + graph)
            
        Returns:
            Deduplicated candidates with merged scores
        """
        merged: Dict[str, RetrievalCandidate] = {}
        
        for candidate in candidates:
            if candidate.memory_id in merged:
                # Merge scores
                existing = merged[candidate.memory_id]
                existing.vector_score = max(existing.vector_score, candidate.vector_score)
                existing.graph_score = max(existing.graph_score, candidate.graph_score)
                existing.recency_score = max(existing.recency_score, candidate.recency_score)
                existing.trust_score = max(existing.trust_score, candidate.trust_score)
                existing.importance = max(existing.importance, candidate.importance)
                if candidate.source == "vector" or existing.source == "vector":
                    existing.source = "both" if candidate.source != existing.source else existing.source
            else:
                merged[candidate.memory_id] = candidate
        
        return list(merged.values())
    
    def hybrid_rerank(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Steps 7-8: Apply hybrid scoring and rerank.
        
        Args:
            candidates: Merged candidates
            
        Returns:
            Reranked candidates with hybrid scores
        """
        for candidate in candidates:
            # Calculate hybrid score
            candidate.hybrid_score = (
                WEIGHTS["vector"] * candidate.vector_score +
                WEIGHTS["graph"] * candidate.graph_score +
                WEIGHTS["recency"] * candidate.recency_score +
                WEIGHTS["trust"] * candidate.trust_score +
                WEIGHTS["importance"] * candidate.importance
            )
        
        # Sort by hybrid score (descending)
        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        
        return candidates
    
    async def assemble_results(
        self,
        candidates: List[RetrievalCandidate],
        limit: int = DEFAULT_FINAL_LIMIT,
        include_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """Steps 9-10: Assemble context and return results.
        
        Args:
            candidates: Reranked candidates
            limit: Maximum results to return
            include_context: Whether to include graph context
            
        Returns:
            Final retrieval results
        """
        results = []
        
        for candidate in candidates[:limit]:
            result = {
                "memory_id": candidate.memory_id,
                "score": round(candidate.hybrid_score, 4),
                "vector_score": round(candidate.vector_score, 4),
                "graph_score": round(candidate.graph_score, 4),
                "recency_score": round(candidate.recency_score, 4),
                "sector": candidate.sector,
                "source": candidate.source,
            }
            
            # Include graph context for explainability
            if include_context:
                try:
                    context = await self.graph_store.get_memory_graph_context(
                        memory_id=candidate.memory_id,
                        include_entities=True,
                        include_related=False,  # Skip to reduce query time
                        limit=5,
                    )
                    result["entities"] = context.get("entities", [])
                except Exception:
                    result["entities"] = []
            
            results.append(result)
        
        return results
    
    def _calculate_recency(self, created_at_str: Optional[str]) -> float:
        """Calculate recency score from timestamp string.
        
        Args:
            created_at_str: ISO format timestamp string
            
        Returns:
            Recency score (0.0-1.0)
        """
        if not created_at_str:
            return 0.5  # Default for unknown
        
        try:
            if isinstance(created_at_str, str):
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            else:
                created_at = created_at_str
            
            now = datetime.utcnow()
            
            # Handle timezone
            if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)
            
            days_since = (now - created_at).days
            
            # Exponential decay
            return math.exp(-days_since / RECENCY_DECAY_DAYS)
            
        except Exception:
            return 0.5


# Global retriever instance
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the global HybridRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


async def retrieve(
    query_embedding: List[float],
    filters: Dict[str, Any],
    limit: int = DEFAULT_FINAL_LIMIT,
) -> List[Dict[str, Any]]:
    """Legacy retrieve function for compatibility.
    
    Args:
        query_embedding: Query embedding vector
        filters: Metadata filters (must include tenant_id)
        limit: Maximum results
        
    Returns:
        Retrieved memories with scores
    """
    retriever = get_retriever()
    
    tenant_id = filters.get("tenant_id", "default")
    sectors = filters.get("sectors", ["semantic", "episodic"])
    
    # Vector search
    candidates = await retriever.vector_search(
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        sectors=sectors,
        filters=filters,
        limit=DEFAULT_ANN_LIMIT,
    )
    
    # Graph expansion
    candidates = await retriever.expand_via_graph(candidates)
    
    # Merge and deduplicate
    candidates = retriever.merge_and_deduplicate(candidates)
    
    # Hybrid rerank
    candidates = retriever.hybrid_rerank(candidates)
    
    # Assemble results
    return await retriever.assemble_results(candidates, limit=limit)


async def hybrid_retrieve(
    query: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    sectors: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = DEFAULT_FINAL_LIMIT,
    graph_hops: int = DEFAULT_GRAPH_HOPS,
    include_context: bool = True,
) -> Dict[str, Any]:
    """Full hybrid retrieval pipeline.
    
    Args:
        query: Query text
        tenant_id: Tenant ID
        user_id: Optional user ID filter
        sectors: Sectors to search (auto-detected if not specified)
        filters: Additional metadata filters
        limit: Maximum results
        graph_hops: Graph expansion hops (1 or 2)
        include_context: Include graph context in results
        
    Returns:
        Retrieval results with metadata
    """
    retriever = get_retriever()
    
    # Step 1: Embed query
    query_embedding = await retriever.embed_query(query)
    
    if not query_embedding:
        logger.error("Failed to embed query")
        return {
            "results": [],
            "total": 0,
            "query": query,
            "sectors_searched": [],
        }
    
    # Step 2: Route to sectors
    target_sectors = await retriever.route_to_sectors(query, sectors)
    
    # Build filters
    search_filters = filters or {}
    if user_id:
        search_filters["user_id"] = user_id
    
    # Steps 3-4: Vector search
    candidates = await retriever.vector_search(
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        sectors=target_sectors,
        filters=search_filters,
        limit=DEFAULT_ANN_LIMIT,
    )
    
    # Step 5: Graph expansion
    candidates = await retriever.expand_via_graph(
        candidates,
        hops=graph_hops,
    )
    
    # Step 6: Merge and deduplicate
    candidates = retriever.merge_and_deduplicate(candidates)
    
    # Steps 7-8: Hybrid rerank
    candidates = retriever.hybrid_rerank(candidates)
    
    # Steps 9-10: Assemble results
    results = await retriever.assemble_results(
        candidates,
        limit=limit,
        include_context=include_context,
    )
    
    # Update coactivation for top results
    if len(results) >= 2:
        result_ids = [r["memory_id"] for r in results[:5]]
        try:
            await update_coactivation(result_ids)
        except Exception as e:
            logger.warning("Failed to update coactivation", error=str(e))
    
    logger.info(
        "Hybrid retrieval completed",
        query_length=len(query),
        sectors=target_sectors,
        total_candidates=len(candidates),
        results_returned=len(results),
    )
    
    return {
        "results": results,
        "total": len(candidates),
        "query": query,
        "sectors_searched": target_sectors,
    }


async def search_by_embedding(
    embedding: List[float],
    tenant_id: str,
    sectors: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Simple vector-only search (no graph expansion).
    
    Args:
        embedding: Query embedding
        tenant_id: Tenant ID
        sectors: Sectors to search
        limit: Maximum results
        
    Returns:
        Search results
    """
    retriever = get_retriever()
    
    candidates = await retriever.vector_search(
        query_embedding=embedding,
        tenant_id=tenant_id,
        sectors=sectors,
        limit=limit,
    )
    
    candidates = retriever.hybrid_rerank(candidates)
    
    return await retriever.assemble_results(candidates, limit=limit, include_context=False)
