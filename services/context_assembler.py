"""Context Assembler Service.

Assembles retrieved memories into coherent context with:
- Provenance chain for transparency
- Waypoint traces explaining WHY each memory was retrieved
- Structured context for LLM consumption
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import structlog

logger = structlog.get_logger()


@dataclass
class WaypointStep:
    """A single step in the retrieval waypoint trace."""
    step_number: int
    step_type: str  # vector_match, graph_expansion, recency_boost, trust_factor, etc.
    description: str
    contribution: float  # How much this step contributed to the score (0-1)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryWaypointTrace:
    """Complete waypoint trace for a retrieved memory."""
    memory_id: str
    final_score: float
    retrieval_path: List[WaypointStep]
    explanation: str  # Human-readable summary
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProvenanceEntry:
    """A single provenance entry tracking memory origin."""
    source: str  # api, browser_extension, agent_connector, etc.
    timestamp: str
    action: str  # created, updated, merged, linked
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledMemory:
    """A memory assembled with full context and trace."""
    memory_id: str
    content: str
    sector: str
    score: float
    waypoint_trace: MemoryWaypointTrace
    provenance: List[ProvenanceEntry]
    entities: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    """Assembles retrieved memories into coherent context."""
    
    def __init__(self):
        """Initialize the context assembler."""
        self.max_context_length = 8000  # Max chars for assembled context
        self.include_provenance = True
        self.include_waypoints = True
    
    def generate_waypoint_trace(
        self,
        memory: Dict[str, Any],
    ) -> MemoryWaypointTrace:
        """Generate waypoint trace explaining why this memory was retrieved.
        
        Args:
            memory: Retrieved memory with scores
            
        Returns:
            Complete waypoint trace
        """
        steps = []
        step_num = 1
        
        # Step 1: Vector similarity (always first)
        vector_score = memory.get("vector_score", 0.0)
        if vector_score > 0:
            steps.append(WaypointStep(
                step_number=step_num,
                step_type="vector_match",
                description=f"Semantic similarity matched with score {vector_score:.3f}",
                contribution=vector_score * 0.55,  # Weight from hybrid formula
                details={
                    "raw_score": vector_score,
                    "weight": 0.55,
                    "method": "cosine_similarity",
                    "index": f"memories_{memory.get('sector', 'semantic')}",
                }
            ))
            step_num += 1
        
        # Step 2: Graph expansion (if applicable)
        graph_score = memory.get("graph_score", 0.0)
        source = memory.get("source", "vector")
        
        if graph_score > 0 or source in ["graph", "both"]:
            graph_desc = "Discovered via graph neighborhood"
            if source == "both":
                graph_desc = "Reinforced by graph connections"
            
            steps.append(WaypointStep(
                step_number=step_num,
                step_type="graph_expansion",
                description=f"{graph_desc} (weight: {graph_score:.3f})",
                contribution=graph_score * 0.15,
                details={
                    "raw_score": graph_score,
                    "weight": 0.15,
                    "relationships": memory.get("relationships", []),
                    "hops": memory.get("graph_hops", 1),
                }
            ))
            step_num += 1
        
        # Step 3: Recency factor
        recency_score = memory.get("recency_score", 0.5)
        if recency_score > 0:
            recency_desc = "recent" if recency_score > 0.7 else "moderately recent" if recency_score > 0.4 else "older"
            steps.append(WaypointStep(
                step_number=step_num,
                step_type="recency_boost",
                description=f"Memory is {recency_desc} (decay: {recency_score:.3f})",
                contribution=recency_score * 0.15,
                details={
                    "raw_score": recency_score,
                    "weight": 0.15,
                    "decay_days": 90,  # From RECENCY_DECAY_DAYS constant
                }
            ))
            step_num += 1
        
        # Step 4: Trust factor
        trust_score = memory.get("trust_score", 0.5)
        trust_level = "high" if trust_score > 0.8 else "medium" if trust_score > 0.5 else "low"
        steps.append(WaypointStep(
            step_number=step_num,
            step_type="trust_factor",
            description=f"Trust level: {trust_level} ({trust_score:.2f})",
            contribution=trust_score * 0.10,
            details={
                "raw_score": trust_score,
                "weight": 0.10,
                "source_verified": trust_score > 0.7,
            }
        ))
        step_num += 1
        
        # Step 5: Importance factor
        importance = memory.get("importance", 0.5)
        importance_level = "high" if importance > 0.7 else "medium" if importance > 0.4 else "standard"
        steps.append(WaypointStep(
            step_number=step_num,
            step_type="importance_factor",
            description=f"Importance: {importance_level} ({importance:.2f})",
            contribution=importance * 0.05,
            details={
                "raw_score": importance,
                "weight": 0.05,
            }
        ))
        
        # Generate human-readable explanation
        final_score = memory.get("score", 0.0)
        explanation = self._generate_explanation(
            memory_id=memory.get("memory_id", ""),
            vector_score=vector_score,
            graph_score=graph_score,
            recency_score=recency_score,
            trust_score=trust_score,
            importance=importance,
            final_score=final_score,
            source=source,
        )
        
        return MemoryWaypointTrace(
            memory_id=memory.get("memory_id", ""),
            final_score=final_score,
            retrieval_path=steps,
            explanation=explanation,
            factors={
                "vector_similarity": vector_score,
                "graph_proximity": graph_score,
                "recency": recency_score,
                "trust": trust_score,
                "importance": importance,
            }
        )
    
    def _generate_explanation(
        self,
        memory_id: str,
        vector_score: float,
        graph_score: float,
        recency_score: float,
        trust_score: float,
        importance: float,
        final_score: float,
        source: str,
    ) -> str:
        """Generate human-readable explanation for retrieval.
        
        Args:
            Various scores and factors
            
        Returns:
            Human-readable explanation string
        """
        parts = []
        
        # Primary reason
        if vector_score > 0.7:
            parts.append(f"This memory is highly semantically similar to your query (similarity: {vector_score:.0%})")
        elif vector_score > 0.4:
            parts.append(f"This memory has moderate semantic similarity ({vector_score:.0%})")
        
        # Graph contribution
        if source == "graph":
            parts.append("It was discovered through graph connections to other relevant memories")
        elif source == "both":
            parts.append("It was also reinforced by graph relationships")
        
        # Recency
        if recency_score > 0.8:
            parts.append("This is a recent memory")
        elif recency_score < 0.3:
            parts.append("Although older, it remains relevant")
        
        # Trust
        if trust_score > 0.8:
            parts.append("It comes from a highly trusted source")
        elif trust_score < 0.4:
            parts.append("Note: This source has lower trust score")
        
        # Importance
        if importance > 0.7:
            parts.append("This memory has been marked as high importance")
        
        if not parts:
            return f"Retrieved with combined score of {final_score:.3f}"
        
        return ". ".join(parts) + f". Final score: {final_score:.3f}"
    
    def extract_provenance(
        self,
        memory: Dict[str, Any],
    ) -> List[ProvenanceEntry]:
        """Extract provenance chain from memory data.
        
        Args:
            memory: Memory data dict
            
        Returns:
            List of provenance entries
        """
        provenance_data = memory.get("provenance", [])
        
        if not provenance_data:
            # Create default provenance entry
            return [ProvenanceEntry(
                source="unknown",
                timestamp=datetime.utcnow().isoformat(),
                action="retrieved",
            )]
        
        entries = []
        for entry in provenance_data:
            if isinstance(entry, dict):
                entries.append(ProvenanceEntry(
                    source=entry.get("source", "unknown"),
                    timestamp=entry.get("timestamp", datetime.utcnow().isoformat()),
                    action=entry.get("action", "unknown"),
                    user_id=entry.get("user_id"),
                    agent_id=entry.get("agent_id"),
                    details=entry.get("details", {}),
                ))
            elif isinstance(entry, ProvenanceEntry):
                entries.append(entry)
        
        return entries
    
    async def assemble_memory(
        self,
        memory: Dict[str, Any],
        include_trace: bool = True,
        include_provenance: bool = True,
    ) -> AssembledMemory:
        """Assemble a single memory with full context.
        
        Args:
            memory: Raw memory data from retriever
            include_trace: Include waypoint trace
            include_provenance: Include provenance chain
            
        Returns:
            Assembled memory with context
        """
        # Generate waypoint trace
        waypoint_trace = self.generate_waypoint_trace(memory) if include_trace else MemoryWaypointTrace(
            memory_id=memory.get("memory_id", ""),
            final_score=memory.get("score", 0.0),
            retrieval_path=[],
            explanation="",
        )
        
        # Extract provenance
        provenance = self.extract_provenance(memory) if include_provenance else []
        
        # Extract content
        content = memory.get("content", "")
        if isinstance(content, dict):
            content = content.get("text", content.get("content", str(content)))
        
        return AssembledMemory(
            memory_id=memory.get("memory_id", ""),
            content=str(content),
            sector=memory.get("sector", "semantic"),
            score=memory.get("score", 0.0),
            waypoint_trace=waypoint_trace,
            provenance=provenance,
            entities=memory.get("entities", []),
            related_memories=memory.get("related_memories", []),
            metadata={
                "trust_score": memory.get("trust_score", 0.5),
                "importance": memory.get("importance", 0.5),
                "source": memory.get("source", "vector"),
            }
        )
    
    async def assemble_context(
        self,
        memories: List[Dict[str, Any]],
        query: str = "",
        max_memories: int = 10,
        include_trace: bool = True,
        include_provenance: bool = True,
        format_for_llm: bool = False,
    ) -> Dict[str, Any]:
        """Assemble multiple memories into coherent context.
        
        Args:
            memories: List of retrieved memories
            query: Original query (for context)
            max_memories: Maximum memories to include
            include_trace: Include waypoint traces
            include_provenance: Include provenance chains
            format_for_llm: Format output for LLM consumption
            
        Returns:
            Assembled context with all metadata
        """
        if not memories:
            return {
                "context": "",
                "memories": [],
                "provenance_summary": [],
                "waypoint_summary": [],
                "total_memories": 0,
            }
        
        # Limit memories
        memories = memories[:max_memories]
        
        # Assemble each memory
        assembled_memories = []
        for memory in memories:
            assembled = await self.assemble_memory(
                memory,
                include_trace=include_trace,
                include_provenance=include_provenance,
            )
            assembled_memories.append(assembled)
        
        # Build context string
        if format_for_llm:
            context = self._format_for_llm(assembled_memories, query)
        else:
            context = self._format_context(assembled_memories)
        
        # Truncate if needed
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...[truncated]"
        
        # Build provenance summary
        provenance_summary = []
        for mem in assembled_memories:
            for prov in mem.provenance:
                provenance_summary.append({
                    "memory_id": mem.memory_id,
                    "source": prov.source,
                    "action": prov.action,
                    "timestamp": prov.timestamp,
                })
        
        # Build waypoint summary
        waypoint_summary = []
        for mem in assembled_memories:
            waypoint_summary.append({
                "memory_id": mem.memory_id,
                "explanation": mem.waypoint_trace.explanation,
                "final_score": mem.waypoint_trace.final_score,
                "factors": mem.waypoint_trace.factors,
                "step_count": len(mem.waypoint_trace.retrieval_path),
            })
        
        logger.info(
            "Context assembled",
            memory_count=len(assembled_memories),
            context_length=len(context),
            has_provenance=include_provenance,
            has_waypoints=include_trace,
        )
        
        return {
            "context": context,
            "memories": [self._serialize_memory(m) for m in assembled_memories],
            "provenance_summary": provenance_summary,
            "waypoint_summary": waypoint_summary,
            "total_memories": len(assembled_memories),
            "query": query,
        }
    
    def _format_context(self, memories: List[AssembledMemory]) -> str:
        """Format memories into plain text context.
        
        Args:
            memories: Assembled memories
            
        Returns:
            Formatted context string
        """
        parts = []
        
        for i, mem in enumerate(memories, 1):
            part = f"[Memory {i}] (Score: {mem.score:.3f}, Sector: {mem.sector})\n"
            part += f"{mem.content}\n"
            
            if mem.entities:
                part += f"Entities: {', '.join(mem.entities[:5])}\n"
            
            parts.append(part)
        
        return "\n---\n".join(parts)
    
    def _format_for_llm(self, memories: List[AssembledMemory], query: str) -> str:
        """Format memories for LLM consumption with clear structure.
        
        Args:
            memories: Assembled memories
            query: Original query
            
        Returns:
            LLM-optimized context string
        """
        parts = [
            "# Retrieved Memory Context",
            "",
            f"Query: {query}" if query else "",
            f"Memories Retrieved: {len(memories)}",
            "",
            "## Memories (sorted by relevance)",
            "",
        ]
        
        for i, mem in enumerate(memories, 1):
            parts.append(f"### Memory {i}")
            parts.append(f"- **Relevance Score**: {mem.score:.3f}")
            parts.append(f"- **Category**: {mem.sector}")
            parts.append(f"- **Why Retrieved**: {mem.waypoint_trace.explanation}")
            parts.append("")
            parts.append(f"**Content**:")
            parts.append(mem.content)
            parts.append("")
            
            if mem.entities:
                parts.append(f"**Key Entities**: {', '.join(mem.entities[:5])}")
                parts.append("")
        
        return "\n".join(parts)
    
    def _serialize_memory(self, memory: AssembledMemory) -> Dict[str, Any]:
        """Serialize assembled memory to dict.
        
        Args:
            memory: Assembled memory
            
        Returns:
            Serializable dict
        """
        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "sector": memory.sector,
            "score": memory.score,
            "entities": memory.entities,
            "related_memories": memory.related_memories,
            "metadata": memory.metadata,
            "waypoint_trace": {
                "memory_id": memory.waypoint_trace.memory_id,
                "final_score": memory.waypoint_trace.final_score,
                "explanation": memory.waypoint_trace.explanation,
                "factors": memory.waypoint_trace.factors,
                "path": [asdict(step) for step in memory.waypoint_trace.retrieval_path],
            },
            "provenance": [asdict(p) for p in memory.provenance],
        }


# Global instance
_assembler: Optional[ContextAssembler] = None


def get_context_assembler() -> ContextAssembler:
    """Get or create global ContextAssembler instance."""
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler()
    return _assembler


async def assemble_context(
    memories: List[Dict[str, Any]],
    query: str = "",
    include_trace: bool = True,
    include_provenance: bool = True,
    format_for_llm: bool = False,
) -> Dict[str, Any]:
    """Assemble context from retrieved memories.

    Args:
        memories: Retrieved memories
        query: Original query
        include_trace: Include waypoint traces
        include_provenance: Include provenance chains
        format_for_llm: Format for LLM consumption

    Returns:
        Assembled context with provenance and waypoint traces
    """
    assembler = get_context_assembler()
    return await assembler.assemble_context(
        memories=memories,
        query=query,
        include_trace=include_trace,
        include_provenance=include_provenance,
        format_for_llm=format_for_llm,
    )


async def generate_waypoint_explanation(
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate waypoint trace explanation for a single memory.
    
    Args:
        memory: Memory data with scores
        
    Returns:
        Waypoint trace as dict
    """
    assembler = get_context_assembler()
    trace = assembler.generate_waypoint_trace(memory)
    return {
        "memory_id": trace.memory_id,
        "final_score": trace.final_score,
        "explanation": trace.explanation,
        "factors": trace.factors,
        "path": [asdict(step) for step in trace.retrieval_path],
    }
