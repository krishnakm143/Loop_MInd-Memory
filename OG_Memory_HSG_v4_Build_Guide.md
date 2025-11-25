LoopMind_Memory HSG v4 — Detailed Backend Build Guide
Generated: 2025-11-15 UTC

--------------------------------------------------------------------------------
CONTENTS
1. Project overview
2. Backend goals
3. Prerequisites & environment
4. High-level architecture (components)
5. Component responsibilities (detailed)
6. Data models and schemas
7. Smart Updater: logic & pseudocode
8. ANTIVIRAL: ingestion defense & rules
9. Embedding worker & Vector DB pattern
10. Retrieval pipeline (step-by-step)
11. Adapter training & safe continuous learning
12. CI/CD, monitoring, and ops
13. Local dev (docker-compose)
14. Security & compliance
15. Test plan & benchmarks
16. 90-day roadmap & sprint plan
17. Appendix: endpoints, sample code, dev checklist
--------------------------------------------------------------------------------

1. Project overview
-------------------
LoopMind_Memory HSG v4 is a production-grade memory platform that provides long-term,
explainable, and safe memory for agents and applications. It combines:
- a Hierarchical Semantic Graph (HSG) with sector shards (episodic, semantic, emotional, procedural, workspace),
- a hot vector ANN index (HNSW) for sub-100 ms retrieval,
- a safe training pipeline (ANTIVIRAL + sandbox + canary),
- adapter-based personalization (LoRA / PEFT),
- append-only sector files for cold storage and tamper-evidence,
- a provenance-first approach and signed audit ledger.

Key non-functional goals:
- Privacy & data ownership,
- Explainability (waypoint trace),
- Attack-resilience (ANTIVIRAL + quarantine),
- Scalability to 10M+ memories (hot/cold tiering),
- Retrieval p95 < 100ms on hot set.

2. Backend goals
----------------
- Reliable ingest, safe persistence, and immutable provenance.
- Deterministic MERGE/LINK/CREATE decisions (Smart Updater).
- Rapid hybrid retrieval combining vector + graph + temporal signals.
- Safe continual learning via adapter bank and sandboxed training.
- Strong observability + governance and operator control.

3. Prerequisites & environment
------------------------------
Dev machine minimum (for engineering & integration):
- Python 3.10+, Docker, Docker Compose, Node 18+, Git.
Recommended dev: 32 GB RAM, multi-core CPU. For local training: NVIDIA GPU (>=12GB VRAM).

Production:
- Kubernetes (EKS/GKE/AKS), ArgoCD, Managed Postgres, Qdrant/Chroma cluster or self-hosted Qdrant,
- Neo4j (or DuckDB Graph) with HA, Redis cluster, S3-compatible object store,
- GPU pool for training (optional cloud GPUs).

Libraries / tech stack:
- Backend: FastAPI (Python), asyncio, SQLAlchemy/Alembic
- Vector store: Qdrant / Chroma / custom HNSW
- Graph store: Neo4j (Bolt) or DuckDB Graph
- Queues: Redis (RQ / Celery) for tasks; Kafka optional for event stream
- Model infra: ONNX for local small embedders; PyTorch for GNN & training jobs
- CI/CD: GitHub Actions + ArgoCD, Docker registry
- Observability: Prometheus, Grafana, Loki, Sentry

4. High-level architecture (components)
---------------------------------------
(Components & how they connect; keep this mirrored with your FigJam)
- Clients: Web dashboard, Browser extension (IndexedDB), Agent SDKs, Connectors
- API Gateway: Auth, rate limit, tenancy, request validation (HTTPS, JWT)
- Ingest Service: Sanitizer, PII redaction, ANTIVIRAL pre-scan, parser
- Extractor: LLM or rule-based entity/fact extractor
- Embedding Router: local ONNX / cloud embed    ding, batching
- Sector Classifier: route memory to episodic/semantic/emotional/procedural/workspace
- HSG Core: Node builder, Edge constructor, coactivation engine, decay scheduler
- Vector DB (hot): HNSW index for sector-specific hot sets
- Graph DB: Neo4j / DuckDB Graph for relations & traversal
- Metadata DB: Postgres (JSONB), canonical memory with versions
- Sector files: append-only binary/mmapped files for cold store
- Queues & Workers: Redis tasks; workers handle embedding, reflection, compression, training
- Retriever Service: hybrid retrieval (filter → ANN → graph → rerank)
- Trainer: adapter trainer jobs (K8s GPU)
- Sandbox + Canary: validator & rollout controller
- Ops: Logging (Loki/ELK), Metrics (Prometheus/Grafana), Audit ledger & SIEM

5. Component responsibilities (detailed)
----------------------------------------
API Gateway (FastAPI)
- Authenticate (JWT or agent Ed25519 signatures).
- Rate-limit and multi-tenant routing.
- Routing to ingest & retrieval endpoints.
- Basic request validation and schema enforcement.

Ingest Service
- Receive POST /v1/memories and other sources.
- Normalize text, language detection, segmentation.
- PII redaction (pattern match + ML detectors).
- Call ANTIVIRAL pre-scan: rule-based / classifier ensemble.
- On pass: push to parser/extractor → produce structured facts.
- Write minimal canonical record to Postgres (status = ingested) + enqueue embedding job.

Extractor
- Use LLM or deterministic rules to extract entities, triples, candidate facts, importance heuristics.
- Produce structured triples and candidate training examples.

Embedding Worker & Router
- Batch embedding requests (32–128).
- Local ONNX (E5/BGE) if available; fallback to cloud (OpenAI/Azure/Gemini).
- Persist vectors to vector DB with rich metadata: memory_id, tenant_id, trust_score, importance, sector tags.

Sector Classifier
- Rule-first + small classifier that assigns memories to a sector (episodic/semantic/emotional/etc.).
- Sector determines which HNSW index and traversal rules apply.

HSG Core (Node & Edge Store)
- Build canonical memory node protobuf with versioning & provenance list.
- Add edges with weights and proofs (coactivation, mentions, temporal).
- Maintain an append-only sector file (mmapped) for cold data with manifest; index points to offsets.
- Emit events to job queue for reflections / adapter candidate generation.

Graph DB
- Nodes: Memory, Fact, Entity, User, Project.
- Edges: RELATED_TO, COACTIVATES_WITH, MENTIONS, SEQUENCE_OF. Keep last_used and weight.
- Provide path extraction and neighborhood queries for retrieval & explainability.

Vector DB (HNSW)
- Sector-local indices. Each vector stored with metadata (trust_score, importance).
- Maintain hotset: a small fast index that fits in memory for sub-100ms lookups.
- Store older vectors in cold store; cold fallback path exists.

Metadata DB (Postgres)
- Canonical memory JSONB, versions, audit pointer.
- Append-only audit_ledger table with event JSON and signatures.
- Jobs, quotas, tenant rate limits.

Queues & Workers
- Redis (tasks) for quick jobs, Kafka optional for event streaming.
- Worker types: embedding, re-embed, reflection (condense), index maintenance, training jobs.

Retriever
- Pre-filter → ANN (top-128) → Graph neighbor expansion (1–2 hops) → merge candidates → reranker (dot-product + recency + trust) → GNN cross-encoder optional → top-K.
- Return context snippet(s) with trace path and provenance.

Trainer (Adapter)
- Candidate store (trust filtered).
- Training jobs run on k8s GPU nodes (LoRA/PEFT).
- Validation sandbox and ANTIVIRAL applied to outputs.
- Canary orchestrator promotes adapter artifacts.

6. Data models and schemas
--------------------------
Canonical memory (Postgres JSONB example)
- memory_id, tenant_id, user_id, type, content, structured JSON, tags, importance, trust_score, version, status, provenance[], created_at, updated_at.

Vector DB document metadata
- id (memory_id), embedding[], tenant_id, user_id, importance, trust_score, sector, created_at.

Graph nodes & edges
- Node: (id, type, properties JSON)
- Edge: (from, to, rel_type, weight, created_at, last_used, provenance)

Audit ledger (append-only)
- event_id, timestamp, tenant_id, entity_type, entity_id, event JSON, signature (ed25519)

Append-only sector file (Protobuf)
- For each record: memory_proto { header (id, len, crc), payload (protobuf bytes) }

7. Smart Updater: logic & pseudocode
-----------------------------------
Purpose: Decide whether to MERGE / LINK / CREATE / FLAG for new incoming memories.

Hybrid scoring components:
- cosine_sim (vector)
- graph_sim (shared entities / path distance)
- recency_boost (exponential decay)
- trust_score (derived from provenance & source trust)
- importance (user actions, frequency)

Pseudocode:
MERGE_THRESHOLD = 0.86
UPDATE_THRESHOLD = 0.72
RECENT_DECAY_DAYS = 90

def hybrid_score(cosine_sim, graph_sim, days_since, trust, importance):
return 0.55cosine_sim + 0.15graph_sim + 0.15exp(-days_since/RECENT_DECAY_DAYS) + 0.10trust + 0.05*importance

def decide_update(new_vec, top_k_results):
best = argmax(hybrid_score(...) over top_k_results)
if best.score >= MERGE_THRESHOLD:
return "merge", best
elif best.score >= UPDATE_THRESHOLD:
return "link", best
else:
return "create", None


MERGE process:
- Reconcile via LLM (rephrase + resolve duplicates) or deterministic triple union.
- Append provenance, increment version, recompute embedding, upsert vector, emit audit event.

Conflict handling:
- If contradiction (high-confidence conflicting facts) → create contradiction object, send to human review queue and set status flagged/quarantined.

8. ANTIVIRAL: ingestion defense & rules
---------------------------------------
Multi-stage approach:
- Deterministic rules: regex for “ignore previous instructions”, injection patterns, suspicious file types, exfil commands.
- Heuristic detectors: unusual token distributions, repetition, code-like sequences, unseen domain patterns.
- Model-based classifier: small transformer-based detector returns threat_score in [0,1].
- Provenance analysis: low provenance + high influence → suspicious.

Actions:
- threat_score > 0.85 → quarantine (immutable store, admin alert)
- 0.5 < threat_score <= 0.85 → mark low-trust & exclude from train pool
- threat_score <= 0.5 → pass on

9. Embedding worker & Vector DB pattern
---------------------------------------
Worker behavior:
- Pull jobs in batches (32–128).
- Compute embedding(s): semantic + episodic + emotional (optional).
- Upsert to vector DB with metadata.
- Update canonical memory record with embedding pointer.
- If new or merged memory, enqueue retrain candidate or adapter candidate generator if conditions met.

Best-practices:
- Batch embeddings to reduce API calls & cost.
- Cache embeddings for repeated canonical texts.
- Maintain hotset selector using access frequency, importance, trust, and recency.

10. Retrieval pipeline (step-by-step)
-------------------------------------
1. Client provides query (user_id, tenant_id, query_text, optional filters).
2. Embedding the query (same embedder as memory embedding).
3. Sector router picks one or more sectors to search.
4. Pre-filter by tenant/tags/date using metadata DB.
5. ANN search on sector's HNSW index (top N = 128 recommended).
6. Expand via graph traversal (1–2 hops) to include coactivation neighbors.
7. Merge the candidate set (de-dup).
8. Score candidates with hybrid formula (vector + graph + recency + trust + importance).
9. Optional GNN or cross-encoder rerank on top-64 (if latency budget allows).
10. Assemble context block and explanation path (waypoint trace).
11. Return top-K results with provenance & explanation.

Performance tuning knobs:
- ef_search and M for HNSW
- Candidate N size (trade-off: recall vs latency)
- Hot/cold index size (keep hot index small enough to fit in RAM)
- Caching common queries (TTL configurable)

11. Adapter training & safe continuous learning
----------------------------------------------
Candidate selection:
- Only memories with trust_score >= threshold and sufficient provenance.
- User corrections and repeated merges are high-priority sources.

Training config:
- LoRA / PEFT adapters
- Replay buffer mixing (20–30% golden examples)
- Conservative LR, early stopping (small steps, e.g., up to 400 steps)
- Monitor for regression on validation set

Validation & sandbox:
- Golden test suite for regressions
- ANTIVIRAL applied to adapter outputs (detect malicious outputs)
- Behavioral checks for hallucination increase

Canary & promotion:
- Canary 1–5% traffic with monitoring (latency, success rate, user feedback)
- Automatic rollback if metrics drop beyond thresholds
- Signed adapter artifact saved to Adapter Registry with provenance

12. CI/CD, monitoring, and ops
------------------------------
- CI: GitHub Actions (lint → unit tests → image build → push)
- CD: ArgoCD or Flux (GitOps)
- Container registry: private registry (ECR/GCR/GHCR)
- Monitoring: Prometheus exporters for services; Grafana dashboards for retrieval latency, embedding queue size, canary health
- Logs: Loki / ELK; Sentry for errors
- Alerts: embedding backlog > X, retrieval p95 > 100ms, quarantine surge, canary regression
- Audit & compliance: periodic snapshot hashing of audit ledger and write to S3; rotate keys via KMS

13. Local dev (docker-compose)
------------------------------
Dev compose (recommended minimal stack):
- Postgres (canonical DB)
- Redis (task queue)
- Qdrant (vector DB dev)
- Neo4j (graph dev)
- API (FastAPI stub)
- Worker (embedding + queue consumer)

14. Security & compliance
-------------------------
- All endpoints TLS; internal mTLS for microservices (gRPC).
- Agent actions signed with Ed25519; audit entries include signatures.
- PII redaction with salted hash tombstones; deletion requests create redaction entries in ledger.
- RBAC + OPA policies for action approval (NeuralGuard).
- SOC-ready: logging, access control, periodic pen tests.

15. Test plan & benchmarks
--------------------------
- Unit tests for ingestion pipeline, Smart Updater rules, ANTIVIRAL detectors.
- Integration tests for ingest→index→retrieve→action path.
- Load & latency tests: synthetic dataset with realistic query distribution.
- Benchmarks to record: recall@5, recall@10, p95 latency, embedding throughput, QPS.
- Security fuzz tests: adversarial prompt injections to test ANTIVIRAL and quarantining.

16. 90-day roadmap & sprint plan
--------------------------------
Phase 0 (Day 0–7): repo, infra skeleton, docker-compose, health checks.
Phase 1 (Day 8–21): ingestion service, PII redaction, ANTIVIRAL pre-scan, embedding worker.
Phase 2 (Day 22–40): HSG core (node build, sector files, graph DB integration, vector DB integration).
Phase 3 (Day 41–60): retrieval engine (hybrid), caching, explainability / waypoint traces.
Phase 4 (Day 61–80): adapter training pipeline, sandbox validation, canary orchestrator.
Phase 5 (Day 81–90): monitoring, RBAC & policies, enterprise readiness.

17. Appendix: endpoints, sample code & checklist
------------------------------------------------
Essential endpoints (developer primer):
- POST /v1/memories   : ingestion (signed or JWT)
- POST /v1/query      : retrieval
- GET  /v1/memories/{id} : full memory record with versions & provenance
- POST /v1/actions/submit : signed action for NeuralGuard

FastAPI skeleton example (paste into api/main.py):
```python
from fastapi import FastAPI, Request
app = FastAPI()

@app.post('/v1/memories')
async def create_memory(payload: dict):
    # validate, sanitize, push to queue
    job_id = enqueue_job('process_memory', payload)
    return {'memory_id': job_id, 'status': 'ingested'}