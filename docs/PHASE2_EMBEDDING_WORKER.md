# Phase 2: Embedding Worker Implementation

## Overview

Phase 2 implements the embedding generation pipeline that:
1. Enqueues embedding jobs when memories are created
2. Processes jobs in the background using a worker
3. Generates embeddings using OpenAI API
4. Stores vectors in Qdrant per sector (episodic, semantic, emotional, procedural, workspace)

## Components

### 1. Job Queue (`core/queue/job_queue.py`)
- Redis-based FIFO queue for embedding jobs
- Methods:
  - `enqueue_embedding_job()` - Add job to queue
  - `dequeue_embedding_job()` - Get job from queue
  - `get_queue_length()` - Check queue status
  - `health_check()` - Verify Redis connection

### 2. Embedding Worker (`workers/embedding_worker.py`)
- Background worker that processes embedding jobs
- Flow:
  1. Dequeue job from Redis
  2. Fetch memory from PostgreSQL
  3. Generate embeddings for all chunks using OpenAI
  4. Create/ensure Qdrant collection exists for sector
  5. Store embeddings in Qdrant with metadata
  6. Update memory status to "processed"

### 3. Integration (`api/routes/v1/memories.py`)
- Memory creation endpoint now enqueues embedding jobs
- Jobs are processed asynchronously by workers

## Usage

### Start the Worker

```bash
# Basic usage
python workers/embedding_worker.py

# With custom worker ID
python workers/embedding_worker.py --worker-id my-worker

# With custom poll interval
python workers/embedding_worker.py --poll-interval 2
```

### Create a Memory

When you create a memory via `POST /v1/memories`, it will:
1. Store the memory in PostgreSQL
2. Automatically enqueue an embedding job
3. Worker will process it in the background

### Check Queue Status

```python
from core.queue.job_queue import JobQueue

queue = JobQueue()
print(f"Queue length: {queue.get_queue_length()}")
```

### Verify Embeddings in Qdrant

Collections are created per sector:
- `memories_episodic` - Episodic memories
- `memories_semantic` - Semantic memories
- `memories_emotional` - Emotional memories
- `memories_procedural` - Procedural memories
- `memories_workspace` - Workspace memories

Each vector point contains:
- `id`: `{memory_id}_{chunk_index}`
- `vector`: 1536-dimensional embedding
- `payload`:
  - `memory_id`: UUID of the memory
  - `tenant_id`: Tenant identifier
  - `user_id`: User identifier
  - `chunk_index`: Index of the chunk
  - `chunk_text`: First 500 chars of chunk text
  - `memory_type`: Sector type
  - `trust_score`: Trust score from memory
  - `importance`: Importance score from memory
  - `created_at`: Timestamp

## Testing

### End-to-End Test

```bash
python scripts/test_embedding_worker.py
```

This script:
1. Creates a memory
2. Checks the queue
3. Processes the embedding job
4. Verifies embeddings in Qdrant

### Manual Testing

1. **Create a memory:**
   ```bash
   curl -X POST http://localhost:8000/v1/memories \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "I learned Python in 2023",
       "tags": ["learning", "python"]
     }'
   ```

2. **Start the worker** (in another terminal):
   ```bash
   python workers/embedding_worker.py
   ```

3. **Check Qdrant dashboard:**
   - Open your Qdrant cloud dashboard
   - You should see collections like `memories_episodic`
   - Vectors should be stored with metadata

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for embedding generation
- `EMBEDDING_MODEL` - Default: "openai"
- `EMBEDDING_DIMENSION` - Default: 1536 (for text-embedding-3-small)
- `EMBEDDING_BATCH_SIZE` - Default: 32
- `REDIS_URL` - Redis connection URL
- `REDIS_HOST`, `REDIS_PORT` - Redis connection details
- `REDIS_USERNAME`, `REDIS_PASSWORD` - For cloud Redis

### Qdrant Configuration

- `QDRANT_URL` - Qdrant server URL
- `QDRANT_API_KEY` - For cloud Qdrant

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /v1/memories
       ▼
┌─────────────────┐
│  Memory API     │
│  (FastAPI)      │
└──────┬──────────┘
       │
       ├─► PostgreSQL (Store memory)
       │
       └─► Redis Queue (Enqueue job)
              │
              ▼
       ┌──────────────┐
       │   Worker     │
       │  (Background)│
       └──────┬───────┘
              │
              ├─► OpenAI API (Generate embeddings)
              │
              └─► Qdrant (Store vectors)
```

## Status Flow

1. **Memory Created** → Status: `ingested`
2. **Job Enqueued** → Job added to Redis queue
3. **Worker Processing** → Worker dequeues and processes
4. **Embeddings Generated** → OpenAI API called
5. **Vectors Stored** → Qdrant updated
6. **Memory Updated** → Status: `processed`

## Error Handling

- If Redis is unavailable, memory creation still succeeds (job not enqueued)
- If OpenAI API fails, job is marked as failed (can be retried)
- If Qdrant is unavailable, job fails (can be retried)
- Worker logs all errors for debugging

## Next Steps

After Phase 2, Phase 3 will implement:
- HSG Builder (extract relationships)
- Neo4j graph storage
- Graph traversal for retrieval

