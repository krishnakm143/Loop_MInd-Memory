"""Vector database client wrapper for Qdrant."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from api.config import settings


class VectorStore:
    """Wrapper for Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client."""
        if settings.qdrant_api_key:
            # Cloud Qdrant with API key
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            # Local Qdrant without authentication
            self.client = QdrantClient(url=settings.qdrant_url)

    async def create_collection(
        self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE
    ) -> bool:
        """Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors (e.g., 1536 for text-embedding-3-small)
            distance: Distance metric (COSINE, EUCLIDEAN, DOT)
            
        Returns:
            True if collection was created or already exists, False on error
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name in existing_names:
                return True  # Collection already exists
            
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return True
        except Exception as e:
            # Collection might already exist (race condition)
            # Or other error - log and return False
            import structlog
            logger = structlog.get_logger()
            logger.warning(
                "Failed to create Qdrant collection",
                collection_name=collection_name,
                error=str(e),
            )
            return False

    async def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> bool:
        """Upsert points to a collection."""
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            print(f"Error upserting points: {e}")
            return False

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter,
            )
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            # Simple health check - try to list collections
            self.client.get_collections()
            return True
        except Exception:
            return False

