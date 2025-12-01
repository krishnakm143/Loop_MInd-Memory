"""Embedding router service with OpenAI text-embedding-3-small support.

Includes performance caching to avoid redundant API calls.
"""

from typing import List, Optional, Dict
import httpx
from api.config import settings
from core.storage.perf_cache import get_perf_cache
import structlog

logger = structlog.get_logger()


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI text-embedding-3-small.

    Args:
        text: Text to embed

    Returns:
        Embedding vector (1536 dimensions for text-embedding-3-small)
    """
    if settings.embedding_model == "openai":
        if not settings.openai_api_key:
            logger.warning("OpenAI API key not set, falling back to placeholder")
            return [0.0] * settings.embedding_dimension

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_embedding_model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
        except Exception as e:
            logger.error("Failed to get OpenAI embedding", error=str(e))
            raise

    elif settings.embedding_model == "azure":
        # TODO: Implement Azure OpenAI embeddings
        logger.warning("Azure embeddings not yet implemented")
        return [0.0] * settings.embedding_dimension

    elif settings.embedding_model == "gemini":
        # TODO: Implement Gemini embeddings
        logger.warning("Gemini embeddings not yet implemented")
        return [0.0] * settings.embedding_dimension

    else:  # local
        # TODO: Implement local ONNX embeddings (E5/BGE)
        logger.warning("Local embeddings not yet implemented")
        return [0.0] * settings.embedding_dimension


async def get_embeddings_batch(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI.
    
    Uses performance cache to avoid redundant API calls.

    Args:
        texts: List of texts to embed
        use_cache: Whether to use embedding cache (default: True)

    Returns:
        List of embedding vectors
    """
    if settings.embedding_model == "openai":
        if not settings.openai_api_key:
            logger.warning("OpenAI API key not set, falling back to placeholder")
            return [[0.0] * settings.embedding_dimension] * len(texts)

        # Check cache for existing embeddings
        cache = get_perf_cache()
        cached_embeddings: Dict[int, List[float]] = {}
        
        if use_cache:
            cached_embeddings = await cache.get_embeddings_batch(texts)
            
            if len(cached_embeddings) == len(texts):
                # All embeddings cached - return immediately
                logger.debug("All embeddings from cache", count=len(texts))
                return [cached_embeddings[i] for i in range(len(texts))]

        # Filter out empty strings - OpenAI API doesn't accept them
        # Also skip texts that are already cached
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if i in cached_embeddings:
                continue  # Already cached
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            logger.debug("All valid texts from cache or empty")
            # Return cached + placeholders for empty
            return [
                cached_embeddings.get(i, [0.0] * settings.embedding_dimension)
                for i in range(len(texts))
            ]
        
        # Track which indices had valid text (for mapping back to original positions)
        valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        
        # OpenAI supports up to 2048 inputs per request
        batch_size = min(settings.embedding_batch_size, 2048)
        all_embeddings = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {settings.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": settings.openai_embedding_model,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Validate response structure
                    if not data or "data" not in data:
                        logger.error(
                            "Invalid OpenAI API response structure",
                            response_keys=list(data.keys()) if isinstance(data, dict) else "not a dict"
                        )
                        raise ValueError("Invalid API response: missing 'data' field")
                    
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    
            except httpx.HTTPStatusError as e:
                # Log detailed error information
                error_detail = str(e)
                if e.response is not None:
                    try:
                        error_body = e.response.json()
                        error_detail = f"{error_detail}. Response: {error_body}"
                    except:
                        error_detail = f"{error_detail}. Response text: {e.response.text[:200]}"
                
                logger.error(
                    "Failed to get OpenAI embeddings batch - HTTP error",
                    error=error_detail,
                    status_code=e.response.status_code if e.response else None,
                    batch_start=i,
                    batch_size=len(batch),
                )
                # Fallback to placeholder embeddings for failed batch
                all_embeddings.extend(
                    [[0.0] * settings.embedding_dimension] * len(batch)
                )
            except Exception as e:
                logger.error(
                    "Failed to get OpenAI embeddings batch - unexpected error",
                    error=str(e),
                    error_type=type(e).__name__,
                    batch_start=i,
                    batch_size=len(batch),
                )
                # Fallback to placeholder embeddings for failed batch
                all_embeddings.extend(
                    [[0.0] * settings.embedding_dimension] * len(batch)
                )

        # Map embeddings back to original positions
        result_embeddings = []
        new_embeddings_map: Dict[int, List[float]] = {}  # For caching
        
        valid_idx = 0
        for orig_idx, text in enumerate(texts):
            if orig_idx in cached_embeddings:
                # Use cached embedding
                result_embeddings.append(cached_embeddings[orig_idx])
            elif text and text.strip():
                if valid_idx < len(all_embeddings):
                    embedding = all_embeddings[valid_idx]
                    result_embeddings.append(embedding)
                    new_embeddings_map[orig_idx] = embedding
                    valid_idx += 1
                else:
                    # Fallback if mapping fails
                    result_embeddings.append([0.0] * settings.embedding_dimension)
            else:
                # Empty string gets placeholder embedding
                result_embeddings.append([0.0] * settings.embedding_dimension)
        
        # Cache new embeddings for future use
        if use_cache and new_embeddings_map:
            texts_to_cache = [texts[i] for i in new_embeddings_map.keys()]
            embeddings_to_cache = [new_embeddings_map[i] for i in new_embeddings_map.keys()]
            await cache.set_embeddings_batch(texts_to_cache, embeddings_to_cache)
            logger.debug(
                "Cached new embeddings",
                cached_count=len(new_embeddings_map),
                from_cache=len(cached_embeddings),
            )

        return result_embeddings

    else:
        # For other providers, call get_embedding in a loop (can be optimized later)
        embeddings = []
        for text in texts:
            embedding = await get_embedding(text)
            embeddings.append(embedding)
        return embeddings
