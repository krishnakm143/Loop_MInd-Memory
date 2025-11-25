"""HSG (Hierarchical Semantic Graph) builder service.

Builds and maintains the graph structure for memories:
- Creates Memory nodes in Neo4j with sector classification
- Links entities to memories via MENTIONS edges
- Creates RELATED_TO edges between memories sharing entities
- Manages temporal SEQUENCE_OF edges
- Updates coactivation weights
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from core.storage.graph import get_graph_store, GraphStore
from services.extractor import extract_entities_and_triples

logger = structlog.get_logger()


class HSGBuilder:
    """Builder for Hierarchical Semantic Graph nodes and edges."""
    
    def __init__(self, graph_store: Optional[GraphStore] = None):
        """Initialize HSG builder.
        
        Args:
            graph_store: Optional GraphStore instance (uses global if not provided)
        """
        self._graph_store = graph_store
    
    @property
    def graph(self) -> GraphStore:
        """Get graph store (lazy initialization)."""
        if self._graph_store is None:
            self._graph_store = get_graph_store()
        return self._graph_store
    
    async def build_memory_node(
        self,
        memory_id: str,
        tenant_id: str,
        user_id: str,
        sector: str,
        importance: float = 0.5,
        trust_score: float = 0.5,
        provenance: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Build a Memory node in the graph.
        
        Args:
            memory_id: Unique memory identifier (UUID)
            tenant_id: Tenant identifier
            user_id: User identifier
            sector: Memory sector (episodic, semantic, emotional, procedural, workspace)
            importance: Importance score (0.0-1.0)
            trust_score: Trust score (0.0-1.0)
            provenance: Provenance chain
            metadata: Additional metadata
            
        Returns:
            Neo4j node ID if created, None on error
        """
        try:
            node_id = await self.graph.create_memory_node(
                memory_id=memory_id,
                tenant_id=tenant_id,
                user_id=user_id,
                sector=sector,
                importance=importance,
                trust_score=trust_score,
                metadata=metadata,
            )
            
            if node_id:
                logger.info(
                    "Memory node created in HSG",
                    memory_id=memory_id,
                    sector=sector,
                    node_id=node_id,
                )
            return node_id
            
        except Exception as e:
            logger.error(
                "Failed to build memory node",
                memory_id=memory_id,
                error=str(e),
            )
            return None
    
    async def extract_and_link_entities(
        self,
        memory_id: str,
        entities: List[Dict[str, Any]],
    ) -> int:
        """Create Entity nodes and MENTIONS edges from memory.
        
        Args:
            memory_id: Memory ID to link entities from
            entities: List of extracted entities
            
        Returns:
            Number of entities linked
        """
        linked_count = 0
        
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "concept")
            confidence = entity.get("confidence", 0.7)
            
            if not name:
                continue
            
            try:
                # Create or merge entity node
                entity_node_id = await self.graph.create_entity_node(
                    name=name,
                    entity_type=entity_type,
                    properties={"confidence": confidence},
                )
                
                if entity_node_id:
                    # Create MENTIONS edge from memory to entity
                    success = await self.graph.create_hsg_edge(
                        from_memory_id=memory_id,
                        to_node_id=entity_node_id,
                        to_node_type="Entity",
                        rel_type="MENTIONS",
                        weight=confidence,
                        properties={"entity_type": entity_type},
                    )
                    
                    if success:
                        linked_count += 1
                        
            except Exception as e:
                logger.warning(
                    "Failed to link entity",
                    memory_id=memory_id,
                    entity=name,
                    error=str(e),
                )
        
        logger.debug(
            "Entities linked to memory",
            memory_id=memory_id,
            total_entities=len(entities),
            linked_count=linked_count,
        )
        
        return linked_count
    
    async def link_facts(
        self,
        memory_id: str,
        triples: List[Dict[str, str]],
    ) -> int:
        """Create Fact nodes from extracted triples and link to memory.
        
        Args:
            memory_id: Memory ID
            triples: List of extracted triples
            
        Returns:
            Number of facts linked
        """
        linked_count = 0
        
        for triple in triples:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            
            if not (subject and predicate and obj):
                continue
            
            try:
                # Create fact node
                fact_node_id = await self.graph.create_fact_node(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=0.8,
                    source_memory_id=memory_id,
                )
                
                if fact_node_id:
                    # Create edge from memory to fact
                    success = await self.graph.create_hsg_edge(
                        from_memory_id=memory_id,
                        to_node_id=fact_node_id,
                        to_node_type="Fact",
                        rel_type="CONTAINS_FACT",
                        weight=0.8,
                    )
                    
                    if success:
                        linked_count += 1
                        
            except Exception as e:
                logger.warning(
                    "Failed to link fact",
                    memory_id=memory_id,
                    error=str(e),
                )
        
        return linked_count
    
    async def find_and_link_related_memories(
        self,
        memory_id: str,
        tenant_id: str,
        entities: List[Dict[str, Any]],
        min_shared_entities: int = 1,
        max_related: int = 10,
    ) -> int:
        """Find memories sharing entities and create RELATED_TO edges.
        
        Args:
            memory_id: Current memory ID
            tenant_id: Tenant ID for filtering
            entities: Entities from current memory
            min_shared_entities: Minimum shared entities to create link
            max_related: Maximum related memories to link
            
        Returns:
            Number of related memories linked
        """
        if not entities:
            return 0
        
        # Get entity names
        entity_names = [e.get("name", "") for e in entities if e.get("name")]
        
        if not entity_names:
            return 0
        
        try:
            # Find memories with shared entities
            related_memories = await self.graph.find_memories_by_entities(
                entity_names=entity_names,
                tenant_id=tenant_id,
                limit=max_related + 1,  # +1 to exclude self
            )
            
            linked_count = 0
            for related in related_memories:
                related_id = related.get("memory_id")
                shared_count = related.get("shared_entities", 0)
                
                # Skip self and memories with insufficient overlap
                if related_id == memory_id or shared_count < min_shared_entities:
                    continue
                
                # Calculate weight based on shared entity ratio
                weight = min(1.0, shared_count / len(entity_names))
                
                # Create RELATED_TO edge
                success = await self.graph.create_hsg_edge(
                    from_memory_id=memory_id,
                    to_node_id=related_id,
                    to_node_type="Memory",
                    rel_type="RELATED_TO",
                    weight=weight,
                    properties={
                        "shared_entities": shared_count,
                        "relationship_type": "entity_overlap",
                    },
                )
                
                if success:
                    linked_count += 1
                    
                if linked_count >= max_related:
                    break
            
            logger.debug(
                "Related memories linked",
                memory_id=memory_id,
                linked_count=linked_count,
            )
            
            return linked_count
            
        except Exception as e:
            logger.error(
                "Failed to find/link related memories",
                memory_id=memory_id,
                error=str(e),
            )
            return 0
    
    async def create_temporal_edge(
        self,
        from_memory_id: str,
        to_memory_id: str,
        sequence_type: str = "follows",
    ) -> bool:
        """Create a temporal SEQUENCE_OF edge between memories.
        
        Args:
            from_memory_id: Earlier memory ID
            to_memory_id: Later memory ID
            sequence_type: Type of sequence (follows, part_of, etc.)
            
        Returns:
            True if edge created
        """
        try:
            return await self.graph.create_hsg_edge(
                from_memory_id=from_memory_id,
                to_node_id=to_memory_id,
                to_node_type="Memory",
                rel_type="SEQUENCE_OF",
                weight=0.8,
                properties={"sequence_type": sequence_type},
            )
        except Exception as e:
            logger.error(
                "Failed to create temporal edge",
                from_id=from_memory_id,
                to_id=to_memory_id,
                error=str(e),
            )
            return False
    
    async def update_coactivation(
        self,
        memory_ids: List[str],
        weight_delta: float = 0.05,
    ) -> int:
        """Strengthen coactivation edges between co-retrieved memories.
        
        Args:
            memory_ids: List of memory IDs retrieved together
            weight_delta: Weight increase amount
            
        Returns:
            Number of edges updated
        """
        try:
            return await self.graph.update_coactivation(
                memory_ids=memory_ids,
                weight_delta=weight_delta,
            )
        except Exception as e:
            logger.error(
                "Failed to update coactivation",
                memory_count=len(memory_ids),
                error=str(e),
            )
            return 0
    
    async def get_memory_context(
        self,
        memory_id: str,
        include_entities: bool = True,
        include_related: bool = True,
    ) -> Dict[str, Any]:
        """Get full graph context for a memory (for explainability).
        
        Args:
            memory_id: Memory ID
            include_entities: Include mentioned entities
            include_related: Include related memories
            
        Returns:
            Graph context dictionary
        """
        try:
            return await self.graph.get_memory_graph_context(
                memory_id=memory_id,
                include_entities=include_entities,
                include_related=include_related,
            )
        except Exception as e:
            logger.error(
                "Failed to get memory context",
                memory_id=memory_id,
                error=str(e),
            )
            return {
                "memory_id": memory_id,
                "entities": [],
                "related_memories": [],
                "coactivated_memories": [],
            }


# Global HSG builder instance
_hsg_builder: Optional[HSGBuilder] = None


def get_hsg_builder() -> HSGBuilder:
    """Get or create the global HSGBuilder instance."""
    global _hsg_builder
    if _hsg_builder is None:
        _hsg_builder = HSGBuilder()
    return _hsg_builder


async def build_hsg_node(
    memory_id: str,
    content: Dict[str, Any],
    sector: str,
    tenant_id: str,
    user_id: str,
    importance: float = 0.5,
    trust_score: float = 0.5,
    provenance: Optional[List[Dict[str, Any]]] = None,
    extract_entities: bool = True,
    link_related: bool = True,
) -> Dict[str, Any]:
    """Build a complete HSG node with entities and relationships.
    
    This is the main entry point for building HSG nodes. It:
    1. Creates the Memory node in Neo4j
    2. Extracts entities from content (if enabled)
    3. Creates Entity nodes and MENTIONS edges
    4. Finds and links related memories
    
    Args:
        memory_id: Memory UUID
        content: Memory content dictionary
        sector: Sector classification
        tenant_id: Tenant ID
        user_id: User ID
        importance: Importance score
        trust_score: Trust score
        provenance: Provenance chain
        extract_entities: Whether to extract and link entities
        link_related: Whether to find and link related memories
        
    Returns:
        HSG node structure with stats
    """
    builder = get_hsg_builder()
    
    result = {
        "memory_id": memory_id,
        "sector": sector,
        "node_id": None,
        "entities_linked": 0,
        "facts_linked": 0,
        "related_linked": 0,
        "success": False,
    }
    
    try:
        # Step 1: Create Memory node
        node_id = await builder.build_memory_node(
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            sector=sector,
            importance=importance,
            trust_score=trust_score,
            provenance=provenance,
        )
        
        result["node_id"] = node_id
        
        if not node_id:
            logger.warning("Failed to create memory node", memory_id=memory_id)
            return result
        
        # Step 2: Extract entities and triples from content
        entities = []
        triples = []
        
        if extract_entities:
            # Get text content
            text = ""
            if isinstance(content, dict):
                text = content.get("text", "") or content.get("raw", "") or str(content)
            elif isinstance(content, str):
                text = content
            
            if text:
                extraction = await extract_entities_and_triples(text)
                entities = extraction.get("entities", [])
                triples = extraction.get("triples", [])
        
        # Step 3: Link entities
        if entities:
            result["entities_linked"] = await builder.extract_and_link_entities(
                memory_id=memory_id,
                entities=entities,
            )
        
        # Step 4: Link facts
        if triples:
            result["facts_linked"] = await builder.link_facts(
                memory_id=memory_id,
                triples=triples,
            )
        
        # Step 5: Find and link related memories
        if link_related and entities:
            result["related_linked"] = await builder.find_and_link_related_memories(
                memory_id=memory_id,
                tenant_id=tenant_id,
                entities=entities,
            )
        
        result["success"] = True
        
        logger.info(
            "HSG node built successfully",
            memory_id=memory_id,
            sector=sector,
            entities_linked=result["entities_linked"],
            facts_linked=result["facts_linked"],
            related_linked=result["related_linked"],
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Failed to build HSG node",
            memory_id=memory_id,
            error=str(e),
        )
        return result


async def update_coactivation(memory_ids: List[str], weight_delta: float = 0.05) -> int:
    """Strengthen coactivation between co-retrieved memories.
    
    Args:
        memory_ids: List of memory IDs
        weight_delta: Weight increase
        
    Returns:
        Number of edges updated
    """
    builder = get_hsg_builder()
    return await builder.update_coactivation(memory_ids, weight_delta)


async def get_memory_context(memory_id: str) -> Dict[str, Any]:
    """Get graph context for a memory.
    
    Args:
        memory_id: Memory ID
        
    Returns:
        Graph context dictionary
    """
    builder = get_hsg_builder()
    return await builder.get_memory_context(memory_id)


async def create_temporal_link(
    from_memory_id: str,
    to_memory_id: str,
    sequence_type: str = "follows",
) -> bool:
    """Create temporal link between memories.
    
    Args:
        from_memory_id: Earlier memory
        to_memory_id: Later memory
        sequence_type: Type of sequence
        
    Returns:
        Success boolean
    """
    builder = get_hsg_builder()
    return await builder.create_temporal_edge(from_memory_id, to_memory_id, sequence_type)
