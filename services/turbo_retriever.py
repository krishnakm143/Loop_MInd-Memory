"""
TurboRetriever - Ultra-fast memory retrieval system.

Target performance:
- <10ms: Cache hit (synthetic embeddings)
- <50ms: Cache miss with fast mode
- <100ms: Full semantic mode with caching

Optimization techniques:
1. Synthetic embeddings (no API calls)
2. Multi-tier caching (L1 memory + L2 Redis)
3. Parallel query execution
4. Pre-computed indices
5. Early termination
6. SimHash for fast deduplication
"""

import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import structlog

from services.fast_embeddings import FastEmbeddingGenerator, fast_embed, cosine_similarity
from services.speed_cache import get_speed_cache, SpeedCache
# Simple sector classification (no async dependency)

logger = structlog.get_logger()

# Scoring weights
SCORING_WEIGHTS = {
    "similarity": 0.45,
    "overlap": 0.20,
    "recency": 0.15,
    "salience": 0.10,
    "tag_match": 0.10,
}

# Sector relationship matrix for cross-sector scoring
SECTOR_RELATIONSHIPS = {
    "semantic": {"procedural": 0.8, "episodic": 0.6, "reflective": 0.7, "emotional": 0.4},
    "procedural": {"semantic": 0.8, "episodic": 0.6, "reflective": 0.6, "emotional": 0.3},
    "episodic": {"reflective": 0.8, "semantic": 0.6, "procedural": 0.6, "emotional": 0.7},
    "reflective": {"episodic": 0.8, "semantic": 0.7, "procedural": 0.6, "emotional": 0.6},
    "emotional": {"episodic": 0.7, "reflective": 0.6, "semantic": 0.4, "procedural": 0.3},
}


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory_id: str
    content: str
    score: float
    sector: str
    similarity: float
    recency_score: float
    salience: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0


@dataclass
class RetrievalStats:
    """Statistics from a retrieval operation."""
    total_time_ms: float
    cache_hit: bool
    embedding_time_ms: float
    search_time_ms: float
    scoring_time_ms: float
    results_count: int
    mode: str


class TurboRetriever:
    """
    Ultra-fast memory retriever with multi-tier caching.
    
    Modes:
    - turbo: Synthetic embeddings only, maximum speed (<10ms)
    - fast: Synthetic + cache lookup (<50ms)
    - balanced: Hybrid embeddings with caching (<100ms)
    - full: Full semantic embeddings (500ms+)
    """
    
    def __init__(
        self,
        mode: str = "turbo",
        dim: int = 768,
        cache: Optional[SpeedCache] = None,
    ):
        self.mode = mode
        self.dim = dim
        self.cache = cache or get_speed_cache()
        self.embedding_gen = FastEmbeddingGenerator(dim)
        
        # In-memory vector index for turbo mode
        self._memory_vectors: Dict[str, np.ndarray] = {}
        self._memory_metadata: Dict[str, Dict] = {}
        self._sector_index: Dict[str, Set[str]] = {
            "episodic": set(),
            "semantic": set(),
            "procedural": set(),
            "emotional": set(),
            "reflective": set(),
        }
        
        # Statistics
        self._total_queries = 0
        self._cache_hits = 0
        self._total_time_ms = 0.0
    
    def _compute_simhash(self, text: str) -> str:
        """Compute SimHash for fast deduplication."""
        tokens = set(text.lower().split())
        hashes = [hash(t) for t in tokens]
        
        vec = [0] * 64
        for h in hashes:
            for i in range(64):
                if h & (1 << i):
                    vec[i] += 1
                else:
                    vec[i] -= 1
        
        result = ""
        for i in range(0, 64, 4):
            nibble = (
                (8 if vec[i] > 0 else 0) +
                (4 if vec[i + 1] > 0 else 0) +
                (2 if vec[i + 2] > 0 else 0) +
                (1 if vec[i + 3] > 0 else 0)
            )
            result += hex(nibble)[2:]
        
        return result
    
    def _hamming_distance(self, h1: str, h2: str) -> int:
        """Calculate Hamming distance between two SimHashes."""
        dist = 0
        for c1, c2 in zip(h1, h2):
            xor = int(c1, 16) ^ int(c2, 16)
            dist += bin(xor).count('1')
        return dist
    
    def _classify_query_sector(self, text: str) -> str:
        """Simple rule-based sector classification."""
        text_lower = text.lower()
        
        # Episodic patterns
        if any(p in text_lower for p in ["yesterday", "last week", "met", "meeting", "attended", "visited"]):
            return "episodic"
        
        # Procedural patterns
        if any(p in text_lower for p in ["how to", "how do", "steps", "process", "install", "deploy"]):
            return "procedural"
        
        # Emotional patterns
        if any(p in text_lower for p in ["feel", "felt", "happy", "excited", "worried", "sad"]):
            return "emotional"
        
        # Reflective patterns
        if any(p in text_lower for p in ["learned", "realized", "insight", "think", "thought"]):
            return "reflective"
        
        # Default to semantic
        return "semantic"
    
    def _tokenize(self, text: str) -> Set[str]:
        """Fast tokenization."""
        import re
        words = re.findall(r'[a-z0-9]+', text.lower())
        return set(words)
    
    def _compute_token_overlap(self, query_tokens: Set[str], doc_tokens: Set[str]) -> float:
        """Compute token overlap score."""
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        return overlap / len(query_tokens)
    
    def _compute_recency_score(self, last_seen_at: float, t_days: float = 7.0) -> float:
        """Compute recency score with exponential decay."""
        now = time.time() * 1000  # Convert to ms
        days_since = (now - last_seen_at) / (1000 * 60 * 60 * 24)
        return max(0, np.exp(-days_since / t_days))
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-x))
    
    def _boosted_similarity(self, sim: float, tau: float = 3.0) -> float:
        """Boost similarity score with exponential transform."""
        return 1 - np.exp(-tau * sim)
    
    async def index_memory(
        self,
        memory_id: str,
        content: str,
        sector: str,
        salience: float = 0.5,
        metadata: Optional[Dict] = None
    ):
        """
        Index a memory for fast retrieval.
        
        This creates synthetic embeddings and stores them in memory.
        """
        # Generate synthetic embedding
        vector = self.embedding_gen.generate(content, sector)
        
        # Store in memory index
        self._memory_vectors[memory_id] = vector
        self._memory_metadata[memory_id] = {
            "content": content,
            "sector": sector,
            "salience": salience,
            "simhash": self._compute_simhash(content),
            "tokens": self._tokenize(content),
            "last_seen_at": time.time() * 1000,
            **(metadata or {})
        }
        
        # Update sector index
        self._sector_index[sector].add(memory_id)
        
        # Cache the vector
        await self.cache.set_vector(content, sector, vector)
    
    async def query(
        self,
        query_text: str,
        k: int = 10,
        tenant_id: str = "default",
        sectors: Optional[List[str]] = None,
        min_salience: float = 0.0,
    ) -> Tuple[List[RetrievalResult], RetrievalStats]:
        """
        Retrieve memories matching the query.
        
        Returns:
            Tuple of (results, stats)
        """
        start_time = time.time()
        self._total_queries += 1
        
        # Check query cache first
        cached_results = await self.cache.get_query_results(
            query_text, tenant_id, k, sectors
        )
        if cached_results is not None:
            self._cache_hits += 1
            elapsed = (time.time() - start_time) * 1000
            self._total_time_ms += elapsed
            
            return (
                [RetrievalResult(**r) for r in cached_results],
                RetrievalStats(
                    total_time_ms=elapsed,
                    cache_hit=True,
                    embedding_time_ms=0,
                    search_time_ms=0,
                    scoring_time_ms=0,
                    results_count=len(cached_results),
                    mode=self.mode
                )
            )
        
        # Classify query to determine primary sector (simple rule-based)
        emb_start = time.time()
        query_sector = self._classify_query_sector(query_text)
        
        # Generate query embedding
        query_vector = self.embedding_gen.generate(query_text, query_sector)
        query_tokens = self._tokenize(query_text)
        emb_time = (time.time() - emb_start) * 1000
        
        # Determine which sectors to search
        search_sectors = sectors or list(self._sector_index.keys())
        
        # Search phase
        search_start = time.time()
        candidates: List[Tuple[str, float]] = []
        
        for sector in search_sectors:
            memory_ids = self._sector_index.get(sector, set())
            
            for mem_id in memory_ids:
                if mem_id not in self._memory_vectors:
                    continue
                
                mem_vec = self._memory_vectors[mem_id]
                mem_meta = self._memory_metadata[mem_id]
                
                # Skip low salience memories
                if mem_meta.get("salience", 0) < min_salience:
                    continue
                
                # Compute cosine similarity
                similarity = cosine_similarity(query_vector, mem_vec)
                candidates.append((mem_id, similarity))
        
        # Sort by similarity and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:k * 3]  # Get extra for scoring
        search_time = (time.time() - search_start) * 1000
        
        # Scoring phase
        score_start = time.time()
        results: List[RetrievalResult] = []
        
        for mem_id, raw_similarity in top_candidates:
            mem_meta = self._memory_metadata[mem_id]
            mem_sector = mem_meta.get("sector", "semantic")
            
            # Apply sector relationship penalty
            sector_penalty = 1.0
            if mem_sector != query_sector:
                sector_penalty = SECTOR_RELATIONSHIPS.get(
                    query_sector, {}
                ).get(mem_sector, 0.3)
            
            # Compute component scores
            boosted_sim = self._boosted_similarity(raw_similarity * sector_penalty)
            token_overlap = self._compute_token_overlap(
                query_tokens,
                mem_meta.get("tokens", set())
            )
            recency_score = self._compute_recency_score(
                mem_meta.get("last_seen_at", 0)
            )
            salience = mem_meta.get("salience", 0.5)
            
            # Compute hybrid score
            raw_score = (
                SCORING_WEIGHTS["similarity"] * boosted_sim +
                SCORING_WEIGHTS["overlap"] * token_overlap +
                SCORING_WEIGHTS["recency"] * recency_score +
                SCORING_WEIGHTS["salience"] * salience
            )
            final_score = self._sigmoid(raw_score)
            
            results.append(RetrievalResult(
                memory_id=mem_id,
                content=mem_meta.get("content", ""),
                score=final_score,
                sector=mem_sector,
                similarity=raw_similarity,
                recency_score=recency_score,
                salience=salience,
                metadata={
                    "sector_penalty": sector_penalty,
                    "token_overlap": token_overlap,
                },
            ))
        
        # Sort by final score and take top k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:k]
        
        # Update retrieval times
        score_time = (time.time() - score_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        for r in results:
            r.retrieval_time_ms = total_time
        
        # Cache results
        await self.cache.set_query_results(
            query_text, tenant_id, k,
            [self._result_to_dict(r) for r in results],
            sectors
        )
        
        self._total_time_ms += total_time
        
        stats = RetrievalStats(
            total_time_ms=total_time,
            cache_hit=False,
            embedding_time_ms=emb_time,
            search_time_ms=search_time,
            scoring_time_ms=score_time,
            results_count=len(results),
            mode=self.mode
        )
        
        return results, stats
    
    def _result_to_dict(self, result: RetrievalResult) -> Dict:
        """Convert result to dict for caching."""
        return {
            "memory_id": result.memory_id,
            "content": result.content,
            "score": result.score,
            "sector": result.sector,
            "similarity": result.similarity,
            "recency_score": result.recency_score,
            "salience": result.salience,
            "metadata": result.metadata,
            "retrieval_time_ms": result.retrieval_time_ms,
        }
    
    async def batch_query(
        self,
        queries: List[str],
        k: int = 10,
        tenant_id: str = "default",
    ) -> List[Tuple[List[RetrievalResult], RetrievalStats]]:
        """Execute multiple queries in parallel."""
        tasks = [
            self.query(q, k, tenant_id)
            for q in queries
        ]
        return await asyncio.gather(*tasks)
    
    def stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        avg_time = self._total_time_ms / max(1, self._total_queries)
        cache_rate = self._cache_hits / max(1, self._total_queries)
        
        return {
            "mode": self.mode,
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_rate,
            "average_time_ms": avg_time,
            "indexed_memories": len(self._memory_vectors),
            "sector_distribution": {
                k: len(v) for k, v in self._sector_index.items()
            },
            "cache_stats": self.cache.stats(),
        }
    
    async def reinforce_memory(self, memory_id: str, boost: float = 0.1):
        """Boost a memory's salience on successful retrieval."""
        if memory_id in self._memory_metadata:
            current = self._memory_metadata[memory_id].get("salience", 0.5)
            self._memory_metadata[memory_id]["salience"] = min(1.0, current + boost)
            self._memory_metadata[memory_id]["last_seen_at"] = time.time() * 1000


# Singleton instance
_turbo_retriever: Optional[TurboRetriever] = None


def get_turbo_retriever(mode: str = "turbo") -> TurboRetriever:
    """Get or create the turbo retriever singleton."""
    global _turbo_retriever
    if _turbo_retriever is None or _turbo_retriever.mode != mode:
        _turbo_retriever = TurboRetriever(mode=mode)
    return _turbo_retriever

