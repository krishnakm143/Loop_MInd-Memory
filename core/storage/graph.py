"""Graph database client wrapper for Neo4j with HSG support.

Provides graph operations for the Hierarchical Semantic Graph (HSG):
- Memory nodes with sector classification
- Entity nodes (person, location, date, concept)
- Fact nodes (subject-predicate-object triples)
- Weighted edges (RELATED_TO, COACTIVATES_WITH, MENTIONS, SEQUENCE_OF)
"""

import sys
import ssl
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from neo4j import AsyncGraphDatabase, GraphDatabase
import structlog
from api.config import settings

logger = structlog.get_logger()

# Ensure certifi certificates are available for SSL on Windows
try:
    import certifi
    # Set SSL certificate file to use certifi's bundle
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    # certifi not installed, use system certificates
    pass


class GraphStore:
    """Wrapper for Neo4j graph database operations."""

    def __init__(self):
        """Initialize Neo4j driver."""
        # For Neo4j Aura, convert neo4j+s:// to bolt+s:// for direct connection
        # This avoids routing issues with Aura
        uri = settings.neo4j_uri
        original_uri = uri
        
        # Ensure port is specified for Aura (default is 7687)
        if uri.startswith("neo4j+s://"):
            # Convert to bolt+s:// for direct connection (Aura compatible)
            uri = uri.replace("neo4j+s://", "bolt+s://", 1)
            # Add port if not present (Aura uses 7687)
            host_part = uri.split("://", 1)[1]
            if ":" not in host_part:
                uri = uri + ":7687"
            print(f"Converted Neo4j URI from {original_uri[:30]}... to {uri[:30]}...")
        elif uri.startswith("neo4j+ssc://"):
            # Convert to bolt+ssc:// for self-signed certs
            uri = uri.replace("neo4j+ssc://", "bolt+ssc://", 1)
            host_part = uri.split("://", 1)[1]
            if ":" not in host_part:
                uri = uri + ":7687"
            print(f"Converted Neo4j URI from {original_uri[:30]}... to {uri[:30]}...")
        elif uri.startswith("bolt+s://"):
            # Ensure port is present for bolt+s://
            host_part = uri.split("://", 1)[1]
            if ":" not in host_part:
                uri = uri + ":7687"
        
        self.uri = uri
        self.use_sync_driver = sys.platform == "win32"  # Use sync driver on Windows as workaround
        
        # For Aura on Windows, use sync driver wrapped in thread pool to avoid socket issues
        # The async driver on Windows has known issues with WinError 10038
        try:
            driver_config = {
                "auth": (settings.neo4j_user, settings.neo4j_password),
                "max_connection_lifetime": 30 * 60,  # 30 minutes
                "connection_acquisition_timeout": 60,  # 60 seconds
            }
            
            if self.use_sync_driver:
                # Use sync driver on Windows (wrapped in executor for async compatibility)
                # For bolt+s://, encryption is automatic - don't set encrypted/trust config
                # The driver automatically handles SSL for bolt+s:// URI schemes
                # Only set encrypted/trust for plain bolt:// or neo4j:// schemes
                
                self.driver = GraphDatabase.driver(uri, **driver_config)
                self.executor = ThreadPoolExecutor(max_workers=4)
                print(f"Neo4j sync driver initialized (Windows workaround) with URI: {uri[:40]}...")
            else:
                # Use async driver on non-Windows platforms
                self.driver = AsyncGraphDatabase.driver(uri, **driver_config)
                self.executor = None
                print(f"Neo4j async driver initialized with URI: {uri[:40]}...")
        except Exception as e:
            print(f"Failed to initialize Neo4j driver with URI {uri[:40]}...: {e}")
            raise

    async def close(self):
        """Close the Neo4j driver connection."""
        if self.use_sync_driver:
            # Sync driver close is blocking, run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.driver.close)
            self.executor.shutdown(wait=True)
        else:
            await self.driver.close()

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        if self.use_sync_driver:
            # Use sync driver wrapped in executor for Windows
            def _execute():
                with self.driver.session(database="neo4j") as session:
                    result = session.run(query, parameters or {})
                    return [dict(record) for record in result]
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _execute)
        else:
            # Use async driver on non-Windows
            # Neo4j 5.x async driver: iterate using async for to consume records
            async with self.driver.session(database="neo4j") as session:
                result = await session.run(query, parameters or {})
                records = []
                async for record in result:
                    records.append(dict(record))
                return records

    async def create_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Optional[str]:
        """Create a node in the graph."""
        query = f"CREATE (n:{label} $properties) RETURN id(n) as node_id"
        result = await self.execute_query(query, {"properties": properties})
        if result:
            return str(result[0].get("node_id"))
        return None

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between nodes."""
        props = properties or {}
        query = (
            f"MATCH (a), (b) WHERE id(a) = $from_id AND id(b) = $to_id "
            f"CREATE (a)-[r:{rel_type} $props]->(b) RETURN r"
        )
        result = await self.execute_query(
            query, {"from_id": int(from_id), "to_id": int(to_id), "props": props}
        )
        return len(result) > 0

    async def health_check(self) -> tuple[bool, str | None]:
        """Check if Neo4j is healthy.
        
        Returns:
            Tuple of (is_healthy, error_message)
        """
        try:
            if self.use_sync_driver:
                # Use sync driver wrapped in executor for Windows
                def _health_check():
                    with self.driver.session(database="neo4j") as session:
                        result = session.run("RETURN 1 as health")
                        record = result.single()
                        if record and record["health"] == 1:
                            return (True, None)
                        else:
                            return (False, "Query returned unexpected result")
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.executor, _health_check)
            else:
                # Use async driver on non-Windows
                async with self.driver.session(database="neo4j") as session:
                    result = await session.run("RETURN 1 as health")
                    record = await result.single()
                    if record and record["health"] == 1:
                        return (True, None)
                    else:
                        return (False, "Query returned unexpected result")
        except Exception as e:
            # Return the error message for debugging
            error_msg = f"Neo4j health check failed: {str(e)}"
            print(error_msg)
            return (False, error_msg)

    # ========================================================================
    # HSG-Specific Methods
    # ========================================================================

    async def create_memory_node(
        self,
        memory_id: str,
        tenant_id: str,
        user_id: str,
        sector: str,
        importance: float = 0.5,
        trust_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a Memory node in the graph.
        
        Args:
            memory_id: Unique memory identifier (UUID)
            tenant_id: Tenant identifier
            user_id: User identifier
            sector: Memory sector (episodic, semantic, emotional, procedural, workspace)
            importance: Importance score (0.0-1.0)
            trust_score: Trust score (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            Neo4j node ID if created, None on error
        """
        properties = {
            "memory_id": memory_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "sector": sector,
            "importance": importance,
            "trust_score": trust_score,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,
        }
        if metadata:
            properties["metadata"] = str(metadata)  # Neo4j doesn't support nested maps well
        
        query = """
        MERGE (m:Memory {memory_id: $memory_id})
        ON CREATE SET m += $properties
        ON MATCH SET m.last_accessed = $now
        RETURN id(m) as node_id, m.memory_id as memory_id
        """
        try:
            result = await self.execute_query(query, {
                "memory_id": memory_id,
                "properties": properties,
                "now": datetime.utcnow().isoformat(),
            })
            if result:
                logger.debug("Memory node created/updated", memory_id=memory_id, sector=sector)
                return str(result[0].get("node_id"))
            return None
        except Exception as e:
            logger.error("Failed to create memory node", memory_id=memory_id, error=str(e))
            return None

    async def create_entity_node(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create or merge an Entity node in the graph.
        
        Args:
            name: Entity name (e.g., "John", "London", "2024-01-15")
            entity_type: Entity type (person, location, date, concept, organization)
            properties: Additional properties
            
        Returns:
            Neo4j node ID if created, None on error
        """
        # Normalize entity name for consistent matching
        normalized_name = name.strip().lower()
        
        node_properties = {
            "name": name,
            "normalized_name": normalized_name,
            "type": entity_type,
            "created_at": datetime.utcnow().isoformat(),
            "mention_count": 1,
        }
        if properties:
            node_properties.update(properties)
        
        query = """
        MERGE (e:Entity {normalized_name: $normalized_name, type: $type})
        ON CREATE SET e += $properties
        ON MATCH SET e.mention_count = e.mention_count + 1
        RETURN id(e) as node_id, e.name as name
        """
        try:
            result = await self.execute_query(query, {
                "normalized_name": normalized_name,
                "type": entity_type,
                "properties": node_properties,
            })
            if result:
                logger.debug("Entity node created/updated", name=name, type=entity_type)
                return str(result[0].get("node_id"))
            return None
        except Exception as e:
            logger.error("Failed to create entity node", name=name, error=str(e))
            return None

    async def create_fact_node(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source_memory_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a Fact node (subject-predicate-object triple).
        
        Args:
            subject: Subject of the triple
            predicate: Predicate/relationship
            obj: Object of the triple
            confidence: Confidence score (0.0-1.0)
            source_memory_id: Memory ID that this fact came from
            
        Returns:
            Neo4j node ID if created, None on error
        """
        # Create a unique key for the fact
        fact_key = f"{subject.lower()}:{predicate.lower()}:{obj.lower()}"
        
        properties = {
            "fact_key": fact_key,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source_memory_id": source_memory_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        query = """
        MERGE (f:Fact {fact_key: $fact_key})
        ON CREATE SET f += $properties
        ON MATCH SET f.confidence = CASE WHEN $confidence > f.confidence THEN $confidence ELSE f.confidence END
        RETURN id(f) as node_id
        """
        try:
            result = await self.execute_query(query, {
                "fact_key": fact_key,
                "properties": properties,
                "confidence": confidence,
            })
            if result:
                logger.debug("Fact node created/updated", subject=subject, predicate=predicate)
                return str(result[0].get("node_id"))
            return None
        except Exception as e:
            logger.error("Failed to create fact node", error=str(e))
            return None

    async def create_hsg_edge(
        self,
        from_memory_id: str,
        to_node_id: str,
        to_node_type: str,
        rel_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a weighted edge in the HSG.
        
        Args:
            from_memory_id: Source memory ID (UUID string)
            to_node_id: Target node ID (Neo4j internal ID or memory_id)
            to_node_type: Type of target node (Memory, Entity, Fact)
            rel_type: Relationship type (RELATED_TO, COACTIVATES_WITH, MENTIONS, SEQUENCE_OF)
            weight: Edge weight (0.0-1.0)
            properties: Additional edge properties
            
        Returns:
            True if edge created, False on error
        """
        edge_props = {
            "weight": weight,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat(),
            "use_count": 1,
        }
        if properties:
            edge_props.update(properties)
        
        # Build query based on target node type
        if to_node_type == "Memory":
            query = f"""
            MATCH (a:Memory {{memory_id: $from_id}})
            MATCH (b:Memory {{memory_id: $to_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += $props
            ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
                         r.last_used = $now,
                         r.use_count = r.use_count + 1
            RETURN r
            """
            params = {
                "from_id": from_memory_id,
                "to_id": to_node_id,
                "props": edge_props,
                "weight": weight,
                "now": datetime.utcnow().isoformat(),
            }
        elif to_node_type == "Entity":
            query = f"""
            MATCH (a:Memory {{memory_id: $from_id}})
            MATCH (b:Entity) WHERE id(b) = $to_id
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += $props
            ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
                         r.last_used = $now,
                         r.use_count = r.use_count + 1
            RETURN r
            """
            params = {
                "from_id": from_memory_id,
                "to_id": int(to_node_id),
                "props": edge_props,
                "weight": weight,
                "now": datetime.utcnow().isoformat(),
            }
        else:
            query = f"""
            MATCH (a:Memory {{memory_id: $from_id}})
            MATCH (b) WHERE id(b) = $to_id
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r += $props
            ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
                         r.last_used = $now,
                         r.use_count = r.use_count + 1
            RETURN r
            """
            params = {
                "from_id": from_memory_id,
                "to_id": int(to_node_id),
                "props": edge_props,
                "weight": weight,
                "now": datetime.utcnow().isoformat(),
            }
        
        try:
            result = await self.execute_query(query, params)
            if result:
                logger.debug(
                    "HSG edge created",
                    from_id=from_memory_id,
                    to_id=to_node_id,
                    rel_type=rel_type,
                    weight=weight,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to create HSG edge",
                from_id=from_memory_id,
                to_id=to_node_id,
                error=str(e),
            )
            return False

    async def get_neighbors(
        self,
        memory_id: str,
        rel_types: Optional[List[str]] = None,
        hops: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get N-hop neighbors of a memory node.
        
        Args:
            memory_id: Source memory ID
            rel_types: Filter by relationship types (None = all types)
            hops: Number of hops (1 or 2)
            limit: Maximum neighbors to return
            
        Returns:
            List of neighbor nodes with relationship info
        """
        # Build relationship filter
        if rel_types:
            rel_filter = ":" + "|".join(rel_types)
        else:
            rel_filter = ""
        
        if hops == 1:
            query = f"""
            MATCH (m:Memory {{memory_id: $memory_id}})-[r{rel_filter}]-(neighbor)
            RETURN 
                CASE WHEN neighbor:Memory THEN neighbor.memory_id ELSE id(neighbor) END as neighbor_id,
                labels(neighbor)[0] as neighbor_type,
                neighbor.name as name,
                neighbor.sector as sector,
                type(r) as rel_type,
                r.weight as weight,
                r.use_count as use_count
            ORDER BY r.weight DESC, r.use_count DESC
            LIMIT $limit
            """
        else:
            # 2-hop query
            query = f"""
            MATCH path = (m:Memory {{memory_id: $memory_id}})-[r1{rel_filter}]-(n1)-[r2{rel_filter}]-(n2)
            WHERE n2 <> m
            WITH n2, 
                 CASE WHEN n2:Memory THEN n2.memory_id ELSE id(n2) END as neighbor_id,
                 labels(n2)[0] as neighbor_type,
                 n2.name as name,
                 n2.sector as sector,
                 type(r2) as rel_type,
                 r2.weight as weight,
                 r2.use_count as use_count,
                 length(path) as path_length
            RETURN DISTINCT neighbor_id, neighbor_type, name, sector, rel_type, weight, use_count, path_length
            ORDER BY weight DESC, use_count DESC
            LIMIT $limit
            """
        
        try:
            result = await self.execute_query(query, {
                "memory_id": memory_id,
                "limit": limit,
            })
            logger.debug(
                "Retrieved neighbors",
                memory_id=memory_id,
                hops=hops,
                count=len(result),
            )
            return result
        except Exception as e:
            logger.error("Failed to get neighbors", memory_id=memory_id, error=str(e))
            return []

    async def find_memories_by_entities(
        self,
        entity_names: List[str],
        tenant_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find memories that mention the same entities.
        
        Args:
            entity_names: List of entity names to search for
            tenant_id: Tenant filter
            limit: Maximum results
            
        Returns:
            List of memories with shared entity count
        """
        if not entity_names:
            return []
        
        # Normalize entity names
        normalized_names = [name.strip().lower() for name in entity_names]
        
        query = """
        MATCH (m:Memory {tenant_id: $tenant_id})-[:MENTIONS]->(e:Entity)
        WHERE e.normalized_name IN $entity_names
        WITH m, count(DISTINCT e) as shared_entities, collect(e.name) as entities
        RETURN 
            m.memory_id as memory_id,
            m.sector as sector,
            m.importance as importance,
            m.trust_score as trust_score,
            shared_entities,
            entities
        ORDER BY shared_entities DESC, m.importance DESC
        LIMIT $limit
        """
        try:
            result = await self.execute_query(query, {
                "tenant_id": tenant_id,
                "entity_names": normalized_names,
                "limit": limit,
            })
            logger.debug(
                "Found memories by entities",
                entity_count=len(entity_names),
                result_count=len(result),
            )
            return result
        except Exception as e:
            logger.error("Failed to find memories by entities", error=str(e))
            return []

    async def update_edge_weight(
        self,
        from_memory_id: str,
        to_memory_id: str,
        rel_type: str,
        weight_delta: float = 0.1,
    ) -> bool:
        """Update edge weight (for coactivation strengthening).
        
        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            rel_type: Relationship type
            weight_delta: Amount to increase weight by
            
        Returns:
            True if updated, False on error
        """
        query = f"""
        MATCH (a:Memory {{memory_id: $from_id}})-[r:{rel_type}]-(b:Memory {{memory_id: $to_id}})
        SET r.weight = CASE WHEN r.weight + $delta > 1.0 THEN 1.0 ELSE r.weight + $delta END,
            r.last_used = $now,
            r.use_count = r.use_count + 1
        RETURN r.weight as new_weight
        """
        try:
            result = await self.execute_query(query, {
                "from_id": from_memory_id,
                "to_id": to_memory_id,
                "delta": weight_delta,
                "now": datetime.utcnow().isoformat(),
            })
            if result:
                logger.debug(
                    "Edge weight updated",
                    from_id=from_memory_id,
                    to_id=to_memory_id,
                    new_weight=result[0].get("new_weight"),
                )
                return True
            return False
        except Exception as e:
            logger.error("Failed to update edge weight", error=str(e))
            return False

    async def update_coactivation(self, memory_ids: List[str], weight_delta: float = 0.05) -> int:
        """Strengthen coactivation edges between memories retrieved together.
        
        Args:
            memory_ids: List of memory IDs that were co-retrieved
            weight_delta: Amount to increase weight by
            
        Returns:
            Number of edges updated
        """
        if len(memory_ids) < 2:
            return 0
        
        # Create or update COACTIVATES_WITH edges between all pairs
        query = """
        UNWIND $pairs as pair
        MATCH (a:Memory {memory_id: pair[0]})
        MATCH (b:Memory {memory_id: pair[1]})
        MERGE (a)-[r:COACTIVATES_WITH]-(b)
        ON CREATE SET r.weight = $delta, r.created_at = $now, r.use_count = 1
        ON MATCH SET r.weight = CASE WHEN r.weight + $delta > 1.0 THEN 1.0 ELSE r.weight + $delta END,
                     r.last_used = $now,
                     r.use_count = r.use_count + 1
        RETURN count(r) as updated
        """
        
        # Generate all pairs
        pairs = []
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                pairs.append([memory_ids[i], memory_ids[j]])
        
        try:
            result = await self.execute_query(query, {
                "pairs": pairs,
                "delta": weight_delta,
                "now": datetime.utcnow().isoformat(),
            })
            updated = result[0].get("updated", 0) if result else 0
            logger.debug(
                "Coactivation updated",
                memory_count=len(memory_ids),
                edges_updated=updated,
            )
            return updated
        except Exception as e:
            logger.error("Failed to update coactivation", error=str(e))
            return 0

    async def get_memory_graph_context(
        self,
        memory_id: str,
        include_entities: bool = True,
        include_related: bool = True,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get full graph context for a memory (for explainability).
        
        Args:
            memory_id: Memory ID to get context for
            include_entities: Include mentioned entities
            include_related: Include related memories
            limit: Maximum items per category
            
        Returns:
            Graph context with entities, related memories, and paths
        """
        context = {
            "memory_id": memory_id,
            "entities": [],
            "related_memories": [],
            "coactivated_memories": [],
        }
        
        if include_entities:
            entity_query = """
            MATCH (m:Memory {memory_id: $memory_id})-[:MENTIONS]->(e:Entity)
            RETURN e.name as name, e.type as type, e.mention_count as mention_count
            LIMIT $limit
            """
            try:
                entities = await self.execute_query(entity_query, {
                    "memory_id": memory_id,
                    "limit": limit,
                })
                context["entities"] = entities
            except Exception as e:
                logger.error("Failed to get entities", error=str(e))
        
        if include_related:
            related_query = """
            MATCH (m:Memory {memory_id: $memory_id})-[r:RELATED_TO]-(other:Memory)
            RETURN 
                other.memory_id as memory_id,
                other.sector as sector,
                r.weight as weight
            ORDER BY r.weight DESC
            LIMIT $limit
            """
            try:
                related = await self.execute_query(related_query, {
                    "memory_id": memory_id,
                    "limit": limit,
                })
                context["related_memories"] = related
            except Exception as e:
                logger.error("Failed to get related memories", error=str(e))
            
            coact_query = """
            MATCH (m:Memory {memory_id: $memory_id})-[r:COACTIVATES_WITH]-(other:Memory)
            RETURN 
                other.memory_id as memory_id,
                other.sector as sector,
                r.weight as weight,
                r.use_count as use_count
            ORDER BY r.weight DESC, r.use_count DESC
            LIMIT $limit
            """
            try:
                coactivated = await self.execute_query(coact_query, {
                    "memory_id": memory_id,
                    "limit": limit,
                })
                context["coactivated_memories"] = coactivated
            except Exception as e:
                logger.error("Failed to get coactivated memories", error=str(e))
        
        return context

    async def delete_memory_node(self, memory_id: str) -> bool:
        """Delete a memory node and all its relationships.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if deleted, False on error
        """
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        DETACH DELETE m
        RETURN count(m) as deleted
        """
        try:
            result = await self.execute_query(query, {"memory_id": memory_id})
            deleted = result[0].get("deleted", 0) if result else 0
            if deleted > 0:
                logger.info("Memory node deleted", memory_id=memory_id)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete memory node", memory_id=memory_id, error=str(e))
            return False


# Global graph store instance (lazy initialization)
_graph_store: Optional[GraphStore] = None


def get_graph_store() -> GraphStore:
    """Get or create the global GraphStore instance."""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store
