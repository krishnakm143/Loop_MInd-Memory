"""Embedding router service with OpenAI text-embedding-3-small support."""

from typing import List, Optional
import httpx
from api.config import settings
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


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if settings.embedding_model == "openai":
        if not settings.openai_api_key:
            logger.warning("OpenAI API key not set, falling back to placeholder")
            return [[0.0] * settings.embedding_dimension] * len(texts)

        # Filter out empty strings - OpenAI API doesn't accept them
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts to embed (all empty or whitespace)")
            return [[0.0] * settings.embedding_dimension] * len(texts)
        
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

        # Map embeddings back to original positions (empty strings get placeholder embeddings)
        result_embeddings = []
        valid_idx = 0
        for orig_idx, text in enumerate(texts):
            if text and text.strip():
                if valid_idx < len(all_embeddings):
                    result_embeddings.append(all_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # Fallback if mapping fails
                    result_embeddings.append([0.0] * settings.embedding_dimension)
            else:
                # Empty string gets placeholder embedding
                result_embeddings.append([0.0] * settings.embedding_dimension)

        return result_embeddings

    else:
        # For other providers, call get_embedding in a loop (can be optimized later)
        embeddings = []
        for text in texts:
            embedding = await get_embedding(text)
            embeddings.append(embedding)
        return embeddings
