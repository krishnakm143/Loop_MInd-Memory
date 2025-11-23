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
        """Create a new collection in Qdrant."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return True
        except Exception:
            # Collection might already exist
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

