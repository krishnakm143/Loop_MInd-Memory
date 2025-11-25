# Storage Architecture - Where Data is Stored

## Current Status (Phase 0-1 Implementation)

### ✅ PostgreSQL - **PRIMARY STORAGE (ACTIVE)**
**What it stores:**
- **Canonical memories** - All memory records with full content, metadata, tags, provenance
- **Audit ledger** - Append-only event log with signatures

**Status:** ✅ **FULLY IMPLEMENTED**
- Memories are stored here when you call `POST /v1/memories`
- This is where your memories are RIGHT NOW
- Check with: `SELECT * FROM memories;` in PostgreSQL

**Why it's the primary store:**
- Reliable, ACID-compliant storage
- JSONB for flexible schema
- Full-text search capabilities
- Versioning and audit trail

---

### ⏳ Qdrant - **VECTOR EMBEDDINGS (NOT YET IMPLEMENTED)**
**What it will store:**
- **Vector embeddings** - 1536-dimensional vectors for semantic search
- **Per-sector collections** - Separate HNSW indices for episodic/semantic/emotional/procedural/workspace
- **Metadata** - memory_id, tenant_id, trust_score, importance

**Status:** ⏳ **TODO - Not yet implemented**
- Code exists in `core/storage/vector.py` but not called yet
- See `api/routes/v1/memories.py` line 202-204: `# TODO: Generate embeddings`

**When it will be used:**
1. After memory is created in PostgreSQL
2. Background worker generates embeddings (OpenAI API)
3. Embeddings stored in Qdrant for fast semantic search
4. Used by query endpoint for vector similarity search

**Why it's empty:**
- Embedding generation is not yet implemented
- No vectors have been created yet
- Will be populated when Phase 2 (Embedding Worker) is implemented

---

### ⏳ Neo4j - **GRAPH RELATIONSHIPS (NOT YET IMPLEMENTED)**
**What it will store:**
- **Nodes** - Memory nodes, Entity nodes, Fact nodes
- **Edges** - Relationships like:
  - `RELATED_TO` - Two memories are related
  - `COACTIVATES_WITH` - Memories that activate together
  - `MENTIONS` - Memory mentions an entity
  - `SEQUENCE_OF` - Temporal sequences

**Status:** ⏳ **TODO - Not yet implemented**
- Code exists in `core/storage/graph.py` but not called yet
- See `api/routes/v1/memories.py` line 205: `# TODO: Build HSG nodes/edges`

**When it will be used:**
1. After memory is created and embedded
2. HSG Builder extracts relationships
3. Creates nodes and edges in Neo4j
4. Used by query endpoint for graph traversal

**Why it's empty:**
- Graph building is not yet implemented
- No nodes/edges have been created yet
- Will be populated when Phase 3 (HSG Builder) is implemented

---

### ⏳ Redis - **CACHE & TASK QUEUE (PARTIALLY IMPLEMENTED)**
**What it stores:**
- **Cache** - Frequently accessed data for fast retrieval
- **Task queue** - Background jobs (embedding generation, graph building, etc.)

**Status:** ⏳ **PARTIALLY IMPLEMENTED**
- Cache wrapper exists in `core/storage/cache.py`
- Task queue not yet implemented
- See `api/routes/v1/memories.py` line 202: `# TODO: Enqueue embedding job to Redis queue`

**When it will be used:**
- Caching: Query results, frequently accessed memories
- Task queue: Background embedding jobs, graph building jobs

**Why it might be empty:**
- No caching implemented yet
- No task queue jobs running yet
- Will be used when background workers are implemented

---

## Data Flow (Current vs. Future)

### Current Flow (Phase 0-1):
```
1. Client → POST /v1/memories
2. Preprocessor → Normalize, PII redaction, chunking
3. ANTIVIRAL → Security scan
4. Sector Classifier → Auto-detect type
5. PostgreSQL → Store memory ✅ (THIS IS WHERE IT IS NOW)
6. [TODO] Redis → Enqueue embedding job
7. [TODO] Embedding Worker → Generate embeddings
8. [TODO] Qdrant → Store vectors
9. [TODO] HSG Builder → Extract relationships
10. [TODO] Neo4j → Store graph nodes/edges
```

### Future Flow (Phase 2-3):
```
1. Client → POST /v1/memories
2. Preprocessor → Normalize, PII redaction, chunking
3. ANTIVIRAL → Security scan
4. Sector Classifier → Auto-detect type
5. PostgreSQL → Store memory ✅
6. Redis → Enqueue embedding job ✅
7. Embedding Worker → Generate embeddings ✅
8. Qdrant → Store vectors ✅
9. HSG Builder → Extract relationships ✅
10. Neo4j → Store graph nodes/edges ✅
```

---

## How to Check Where Your Memories Are

### Check PostgreSQL (Where memories ARE stored):
```bash
# Using psql
psql -U postgres -d LoopMind_Memory
SELECT memory_id, type, status, created_at FROM memories;

# Or use the script
python scripts/check_memories_in_db.py
```

### Check Qdrant (Will be populated later):
```bash
# Collections will be created when embeddings are generated
# Collections: episodic, semantic, emotional, procedural, workspace
```

### Check Neo4j (Will be populated later):
```bash
# Nodes and edges will be created when HSG Builder runs
MATCH (n) RETURN n LIMIT 10;
```

---

## Summary

| Service | Purpose | Status | Why Empty? |
|---------|---------|--------|------------|
| **PostgreSQL** | Primary memory storage | ✅ Active | **NOT empty - your memories are here!** |
| **Qdrant** | Vector embeddings for search | ⏳ TODO | Embedding generation not implemented yet |
| **Neo4j** | Graph relationships | ⏳ TODO | Graph building not implemented yet |
| **Redis** | Cache & task queue | ⏳ Partial | Background workers not implemented yet |

**Your memories ARE being saved - they're in PostgreSQL!** The other services will be populated as we implement the remaining phases.

