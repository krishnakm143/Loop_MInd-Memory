"""Graph database client wrapper for Neo4j."""

import sys
import ssl
import asyncio
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from neo4j import AsyncGraphDatabase, GraphDatabase
from api.config import settings

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
            async with self.driver.session(database="neo4j") as session:
                result = await session.run(query, parameters or {})
                records = await result.values()
                return [dict(record) for record in records]

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
