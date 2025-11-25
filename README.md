# LoopMind_Memory HSG v4 Backend

Production-grade memory platform providing long-term, explainable, and safe memory for agents and applications.

## Overview

LoopMind_Memory HSG v4 combines:
- Hierarchical Semantic Graph (HSG) with sector shards (episodic, semantic, emotional, procedural, workspace)
- Hot vector ANN index (HNSW) for sub-100ms retrieval
- Safe training pipeline (ANTIVIRAL + sandbox + canary)
- Adapter-based personalization (LoRA / PEFT)
- Append-only sector files for cold storage and tamper-evidence
- Provenance-first approach and signed audit ledger

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- 32 GB RAM recommended (16 GB minimum)
- Multi-core CPU

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neuralstack-Backend
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - PostgreSQL (port 5432)
   - Redis (port 6379)
   - Qdrant (port 6333)
   - Neo4j (port 7474, 7687)
   - API service (port 8000)

4. **Initialize the database**
   ```bash
   python scripts/init_db.py
   ```

5. **Run Alembic migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the API server**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Verify health**
   ```bash
   curl http://localhost:8000/health
   ```

## Project Structure

```
neuralstack-Backend/
├── api/              # FastAPI application
├── core/             # Core models and database
├── services/         # Business logic services
├── workers/          # Background workers
├── dynamics/         # Memory dynamics (decay, coactivation)
├── training/         # Adapter training pipeline
├── tests/            # Test suite
└── scripts/          # Utility scripts
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. All endpoints except `/health` require a valid JWT token.

### Generating a Token

#### Option 1: Using the API Endpoint (Recommended for Swagger UI)

1. **Quick Token Generation** (default user):
   ```
   GET /v1/auth/token/quick
   ```
   Returns a token with `user_id="test_user"` and `tenant_id="default_tenant"`.

2. **Custom Token Generation**:
   ```
   POST /v1/auth/token
   Body: {
     "user_id": "your_user_id",
     "tenant_id": "your_tenant_id"
   }
   ```

#### Option 2: Using the Script

```bash
python scripts/generate_jwt_token.py --user-id my_user --tenant-id my_tenant
```

### Using the Token in Swagger UI

1. Start the API server: `uvicorn api.main:app --reload`
2. Open Swagger UI: `http://localhost:8000/docs`
3. Click the **"Authorize"** button (lock icon) at the top right
4. In the "Value" field, enter your token (without "Bearer " prefix)
5. Click **"Authorize"**
6. Click **"Close"**
7. Now you can test all protected endpoints

### Using the Token with curl

```bash
# Get a token first
TOKEN=$(curl -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "tenant_id": "default_tenant"}' \
  | jq -r '.access_token')

# Use the token
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/memories
```

### Token Configuration

- **Secret Key**: Set via `JWT_SECRET_KEY` environment variable (default: `your-secret-key-change-in-production`)
- **Algorithm**: HS256 (configurable via `JWT_ALGORITHM`)
- **Expiration**: 30 minutes (configurable via `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`)

**⚠️ Important**: In production, use a strong secret key and implement proper authentication flow. The token generation endpoints are for development/testing only.

## API Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /health/db` - Database connectivity
- `GET /health/redis` - Redis connectivity
- `GET /health/qdrant` - Qdrant connectivity
- `GET /health/neo4j` - Neo4j connectivity

### Authentication
- `POST /v1/auth/token` - Generate JWT token (with custom user_id/tenant_id)
- `GET /v1/auth/token/quick` - Generate quick test token (default values)

### Memory Management
- `POST /v1/memories` - Ingest new memory
- `GET /v1/memories/{id}` - Get memory by ID
- `POST /v1/query` - Query memories

### Actions
- `POST /v1/actions/submit` - Submit action for NeuralGuard approval

## Development Workflow

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Start services** (using Docker Compose)
   ```bash
   docker-compose up -d
   ```
   This starts PostgreSQL, Redis, Qdrant, and Neo4j.

3. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

4. **Start the API server**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Start the embedding worker** (in a separate terminal)
   ```bash
   python workers/embedding_worker.py
   ```
   The worker processes embedding jobs from the Redis queue and stores vectors in Qdrant.

6. **Run tests**
   ```bash
   pytest
   ```

3. **Format code**
   ```bash
   black .
   ruff check --fix .
   ```

4. **Type checking**
   ```bash
   mypy .
   ```

## Environment Variables

See `.env.example` for all required environment variables.

Key variables:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `QDRANT_URL` - Qdrant server URL
- `NEO4J_URI` - Neo4j connection URI
- `JWT_SECRET_KEY` - JWT signing secret
- `ED25519_PRIVATE_KEY_PATH` - Path to Ed25519 private key

## Architecture

See `LoopMind_Memory_HSG_v4_Build_Guide.md` for detailed architecture documentation.

### Data Flow

Client → API Gateway → Preprocessor → ANTIVIRAL → Embedding Router → Sector Classifier → HSG Builder → Storage (Vector/Graph/Metadata) → Hybrid Retriever → Context Assembler → LLM Reasoner → NeuralGuard → Action Broker

## Roadmap

- **Phase 0** (Day 0-7): ✅ Repository setup, infrastructure skeleton, docker-compose, health checks
- **Phase 1** (Day 8-21): Ingest service, PII redaction, ANTIVIRAL pre-scan, embedding worker
- **Phase 2** (Day 22-40): HSG core, sector files, graph DB integration, vector DB integration
- **Phase 3** (Day 41-60): Retrieval engine, caching, explainability
- **Phase 4** (Day 61-80): Adapter training pipeline, sandbox validation, canary orchestrator
- **Phase 5** (Day 81-90): Monitoring, RBAC & policies, enterprise readiness

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]

