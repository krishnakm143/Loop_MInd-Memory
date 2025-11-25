"""LoopMind Python SDK Client.

A production-ready Python client for the LoopMind Memory System API.
"""

import httpx
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json


class LoopMindError(Exception):
    """Base exception for LoopMind SDK."""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class AuthenticationError(LoopMindError):
    """Authentication failed."""
    pass


class RateLimitError(LoopMindError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class NotFoundError(LoopMindError):
    """Resource not found."""
    pass


class ValidationError(LoopMindError):
    """Request validation failed."""
    pass


@dataclass
class MemoryResult:
    """Result of a memory operation."""
    memory_id: str
    status: str
    content: Optional[Any] = None
    type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    trust_score: float = 1.0
    created_at: Optional[datetime] = None
    message: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryResult":
        """Create from API response."""
        return cls(
            memory_id=data.get("memory_id", ""),
            status=data.get("status", ""),
            content=data.get("content"),
            type=data.get("type"),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            trust_score=data.get("trust_score", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            message=data.get("message", ""),
        )


@dataclass
class QueryResult:
    """Result of a query operation."""
    query_id: str
    results: List[Dict[str, Any]]
    total: int
    waypoints: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
    latency_ms: int = 0
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        """Create from API response."""
        return cls(
            query_id=data.get("query_id", ""),
            results=data.get("results", []),
            total=data.get("total", 0),
            waypoints=data.get("waypoints"),
            context=data.get("context"),
            latency_ms=data.get("latency_ms", 0),
        )


@dataclass
class ActionResult:
    """Result of an action submission."""
    action_id: str
    status: str
    risk_score: float
    requires_approval: bool
    message: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActionResult":
        """Create from API response."""
        return cls(
            action_id=data.get("action_id", ""),
            status=data.get("status", ""),
            risk_score=data.get("risk_score", 0.0),
            requires_approval=data.get("requires_approval", False),
            message=data.get("message", ""),
        )


class LoopMindClient:
    """Low-level client for LoopMind API.
    
    Provides direct access to all API endpoints with proper error handling.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize the client.
        
        Args:
            api_key: API key or JWT token for authentication
            base_url: Base URL of the LoopMind API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LoopMind-SDK/1.0.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses."""
        try:
            data = response.json()
            detail = data.get("detail", response.text)
        except Exception:
            detail = response.text
            
        if response.status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {detail}",
                status_code=response.status_code,
            )
        elif response.status_code == 403:
            raise AuthenticationError(
                f"Access denied: {detail}",
                status_code=response.status_code,
            )
        elif response.status_code == 404:
            raise NotFoundError(
                f"Not found: {detail}",
                status_code=response.status_code,
            )
        elif response.status_code == 422:
            raise ValidationError(
                f"Validation error: {detail}",
                status_code=response.status_code,
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", 60)
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s",
                retry_after=int(retry_after),
                status_code=response.status_code,
            )
        else:
            raise LoopMindError(
                f"Request failed: {detail}",
                status_code=response.status_code,
            )
    
    def _request_sync(
        self,
        method: str,
        path: str,
        data: dict = None,
        params: dict = None,
    ) -> dict:
        """Make a synchronous request."""
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
            )
            
        if not response.is_success:
            self._handle_error(response)
            
        return response.json()
    
    async def _request_async(
        self,
        method: str,
        path: str,
        data: dict = None,
        params: dict = None,
    ) -> dict:
        """Make an asynchronous request."""
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
            )
            
        if not response.is_success:
            self._handle_error(response)
            
        return response.json()
    
    # Memory operations
    def create_memory(
        self,
        content: Union[str, dict],
        tags: List[str] = None,
        metadata: dict = None,
    ) -> MemoryResult:
        """Create a new memory synchronously."""
        data = {
            "content": content,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        response = self._request_sync("POST", "/v1/memories", data=data)
        return MemoryResult.from_dict(response)
    
    async def create_memory_async(
        self,
        content: Union[str, dict],
        tags: List[str] = None,
        metadata: dict = None,
    ) -> MemoryResult:
        """Create a new memory asynchronously."""
        data = {
            "content": content,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        response = await self._request_async("POST", "/v1/memories", data=data)
        return MemoryResult.from_dict(response)
    
    def get_memory(self, memory_id: str) -> MemoryResult:
        """Get a memory by ID."""
        response = self._request_sync("GET", f"/v1/memories/{memory_id}")
        return MemoryResult.from_dict(response)
    
    async def get_memory_async(self, memory_id: str) -> MemoryResult:
        """Get a memory by ID asynchronously."""
        response = await self._request_async("GET", f"/v1/memories/{memory_id}")
        return MemoryResult.from_dict(response)
    
    def delete_memory(self, memory_id: str) -> dict:
        """Delete a memory."""
        return self._request_sync("DELETE", f"/v1/memories/{memory_id}")
    
    # Query operations
    def query(
        self,
        query_text: str,
        limit: int = 10,
        sectors: List[str] = None,
        filters: dict = None,
        include_waypoints: bool = False,
    ) -> QueryResult:
        """Query memories synchronously."""
        data = {
            "query_text": query_text,
            "limit": limit,
            "sectors": sectors or [],
            "filters": filters or {},
            "include_waypoints": include_waypoints,
        }
        response = self._request_sync("POST", "/v1/query", data=data)
        return QueryResult.from_dict(response)
    
    async def query_async(
        self,
        query_text: str,
        limit: int = 10,
        sectors: List[str] = None,
        filters: dict = None,
        include_waypoints: bool = False,
    ) -> QueryResult:
        """Query memories asynchronously."""
        data = {
            "query_text": query_text,
            "limit": limit,
            "sectors": sectors or [],
            "filters": filters or {},
            "include_waypoints": include_waypoints,
        }
        response = await self._request_async("POST", "/v1/query", data=data)
        return QueryResult.from_dict(response)
    
    # Action operations
    def submit_action(
        self,
        action_type: str,
        target: str,
        parameters: dict = None,
    ) -> ActionResult:
        """Submit an action for NeuralGuard approval."""
        data = {
            "action_type": action_type,
            "target": target,
            "parameters": parameters or {},
        }
        response = self._request_sync("POST", "/v1/actions/submit", data=data)
        return ActionResult.from_dict(response)
    
    # Batch operations
    def create_memories_batch(
        self,
        memories: List[Dict[str, Any]],
    ) -> dict:
        """Create multiple memories in batch."""
        data = {"memories": memories}
        return self._request_sync("POST", "/v1/batch/memories", data=data)
    
    def query_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> dict:
        """Execute multiple queries in batch."""
        data = {"queries": queries}
        return self._request_sync("POST", "/v1/batch/queries", data=data)
    
    # Analytics
    def get_analytics_overview(self) -> dict:
        """Get analytics overview."""
        return self._request_sync("GET", "/v1/analytics/overview")
    
    # Health check
    def health_check(self) -> dict:
        """Check API health."""
        return self._request_sync("GET", "/health/all")


class LoopMind:
    """High-level client for LoopMind Memory System.
    
    Provides a simple, intuitive interface for common memory operations.
    
    Usage:
        client = LoopMind(api_key="your-api-key")
        
        # Store memories
        client.remember("I met John at the coffee shop")
        client.remember("John likes espresso", tags=["preferences", "people"])
        
        # Recall memories
        results = client.recall("What does John like?")
        for memory in results:
            print(memory.content)
        
        # Forget memories
        client.forget(memory_id)
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "http://localhost:8000",
        **kwargs,
    ):
        """Initialize LoopMind client.
        
        Args:
            api_key: API key or JWT token
            base_url: LoopMind API base URL
            **kwargs: Additional arguments passed to LoopMindClient
        """
        self._client = LoopMindClient(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    def remember(
        self,
        content: Union[str, dict],
        tags: List[str] = None,
        metadata: dict = None,
    ) -> MemoryResult:
        """Store a memory.
        
        Args:
            content: Memory content (text or structured data)
            tags: Optional tags for categorization
            metadata: Optional metadata
            
        Returns:
            MemoryResult with the created memory details
            
        Example:
            result = client.remember("Meeting with John tomorrow at 3pm")
            print(f"Stored memory: {result.memory_id}")
        """
        return self._client.create_memory(content, tags, metadata)
    
    def recall(
        self,
        query: str,
        limit: int = 10,
        sectors: List[str] = None,
        with_explanations: bool = False,
    ) -> List[Dict[str, Any]]:
        """Recall memories matching a query.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            sectors: Optional filter by memory sectors
            with_explanations: Include waypoint explanations
            
        Returns:
            List of matching memories
            
        Example:
            memories = client.recall("What meetings do I have?")
            for m in memories:
                print(m["content"])
        """
        result = self._client.query(
            query_text=query,
            limit=limit,
            sectors=sectors,
            include_waypoints=with_explanations,
        )
        return result.results
    
    def get(self, memory_id: str) -> MemoryResult:
        """Get a specific memory by ID.
        
        Args:
            memory_id: Memory UUID
            
        Returns:
            MemoryResult with full memory details
        """
        return self._client.get_memory(memory_id)
    
    def forget(self, memory_id: str) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: Memory UUID to delete
            
        Returns:
            True if deleted successfully
        """
        self._client.delete_memory(memory_id)
        return True
    
    def remember_many(
        self,
        memories: List[Union[str, Dict[str, Any]]],
    ) -> dict:
        """Store multiple memories at once.
        
        Args:
            memories: List of memory contents (strings or dicts with content/tags)
            
        Returns:
            Batch result with success/failure counts
            
        Example:
            client.remember_many([
                "First memory",
                {"content": "Second memory", "tags": ["important"]},
            ])
        """
        formatted = []
        for m in memories:
            if isinstance(m, str):
                formatted.append({"content": m})
            else:
                formatted.append(m)
        return self._client.create_memories_batch(formatted)
    
    def recall_many(
        self,
        queries: List[str],
        limit_per_query: int = 10,
    ) -> List[QueryResult]:
        """Execute multiple queries at once.
        
        Args:
            queries: List of query strings
            limit_per_query: Max results per query
            
        Returns:
            List of QueryResults
        """
        formatted = [
            {"query_text": q, "limit": limit_per_query}
            for q in queries
        ]
        response = self._client.query_batch(formatted)
        return response.get("results", [])
    
    def stats(self) -> dict:
        """Get memory statistics.
        
        Returns:
            Analytics overview including counts, sectors, usage
        """
        return self._client.get_analytics_overview()
    
    def health(self) -> bool:
        """Check if the API is healthy.
        
        Returns:
            True if API is responding
        """
        try:
            result = self._client.health_check()
            return result.get("status") == "healthy"
        except Exception:
            return False
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

