"""
Speed Benchmark for LoopMind Memory System.

Compares our TurboRetriever against competitor benchmarks.

Targets:
- OpenMemory: 115ms avg, 338 QPS
- Zep: 310ms avg, 180 QPS
- Supermemory: 270ms avg, 220 QPS
- Mem0: 250ms avg, 150 QPS
- Vector DBs: 160ms avg, 250 QPS
"""

import asyncio
import time
import statistics
import random
import string
import sys
from typing import List, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, ".")

from services.fast_embeddings import FastEmbeddingGenerator, fast_embed
from services.speed_cache import SpeedCache
from services.turbo_retriever import TurboRetriever


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    total_queries: int
    total_time_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    qps: float
    cache_hit_rate: float


# Competitor baselines
COMPETITORS = {
    "OpenMemory": {"latency_ms": 115, "qps": 338},
    "Zep": {"latency_ms": 310, "qps": 180},
    "Supermemory": {"latency_ms": 270, "qps": 220},
    "Mem0": {"latency_ms": 250, "qps": 150},
    "Vector DBs": {"latency_ms": 160, "qps": 250},
}


def generate_random_text(length: int = 200) -> str:
    """Generate random text for testing."""
    words = [
        "memory", "recall", "context", "semantic", "episodic",
        "knowledge", "learning", "retrieval", "embedding", "vector",
        "cache", "fast", "quick", "search", "query", "result",
        "data", "information", "store", "retrieve", "find", "match",
        "similar", "relevant", "important", "recent", "history",
    ]
    return " ".join(random.choices(words, k=length // 10))


def generate_test_queries() -> List[str]:
    """Generate diverse test queries."""
    return [
        "What did I discuss in the meeting yesterday?",
        "How do I configure the deployment pipeline?",
        "What made me happy last week?",
        "Tell me about the project architecture",
        "What are the key findings from the research?",
        "How do I fix the authentication bug?",
        "What decisions were made about the product?",
        "Summarize my interactions with the client",
        "What skills did I learn recently?",
        "How do I optimize database queries?",
        "What are my upcoming tasks?",
        "Tell me about CompanyX",
        "What meetings do I have scheduled?",
        "How do I use the API endpoints?",
        "What feedback did I receive?",
        "Tell me about the deployment process",
        "What patterns should I avoid?",
        "How do I handle errors?",
        "What best practices should I follow?",
        "Summarize the project status",
    ]


async def benchmark_embedding_generation(num_samples: int = 1000) -> BenchmarkResult:
    """Benchmark synthetic embedding generation."""
    generator = FastEmbeddingGenerator(dim=768)
    texts = [generate_random_text() for _ in range(num_samples)]
    
    latencies = []
    
    for text in texts:
        start = time.perf_counter()
        _ = generator.generate(text, "semantic")
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    
    total_time = sum(latencies)
    
    return BenchmarkResult(
        name="Embedding Generation",
        total_queries=num_samples,
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(num_samples * 0.95)],
        p99_latency_ms=sorted(latencies)[int(num_samples * 0.99)],
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        qps=num_samples / (total_time / 1000),
        cache_hit_rate=0.0,
    )


async def benchmark_retrieval(
    num_indexed: int = 10000,
    num_queries: int = 1000,
    warmup_queries: int = 100
) -> BenchmarkResult:
    """Benchmark memory retrieval with TurboRetriever."""
    cache = SpeedCache()
    retriever = TurboRetriever(mode="turbo", cache=cache)
    
    print(f"  Indexing {num_indexed} memories...")
    
    # Index memories
    sectors = ["episodic", "semantic", "procedural", "emotional", "reflective"]
    for i in range(num_indexed):
        content = generate_random_text(random.randint(50, 500))
        sector = random.choice(sectors)
        await retriever.index_memory(
            memory_id=f"mem_{i}",
            content=content,
            sector=sector,
            salience=random.random(),
        )
        
        if (i + 1) % 1000 == 0:
            print(f"    Indexed {i + 1}/{num_indexed}")
    
    print(f"  Running {warmup_queries} warmup queries...")
    
    # Warmup phase
    queries = generate_test_queries()
    for _ in range(warmup_queries):
        query = random.choice(queries)
        await retriever.query(query, k=10)
    
    print(f"  Running {num_queries} benchmark queries...")
    
    # Benchmark phase
    latencies = []
    cache_hits = 0
    
    for i in range(num_queries):
        query = random.choice(queries)
        
        start = time.perf_counter()
        results, stats = await retriever.query(query, k=10)
        elapsed = (time.perf_counter() - start) * 1000
        
        latencies.append(elapsed)
        if stats.cache_hit:
            cache_hits += 1
        
        if (i + 1) % 100 == 0:
            avg = statistics.mean(latencies[-100:])
            print(f"    Progress: {i + 1}/{num_queries}, Avg: {avg:.2f}ms")
    
    total_time = sum(latencies)
    sorted_latencies = sorted(latencies)
    
    return BenchmarkResult(
        name="TurboRetriever",
        total_queries=num_queries,
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted_latencies[int(num_queries * 0.95)],
        p99_latency_ms=sorted_latencies[int(num_queries * 0.99)],
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        qps=num_queries / (total_time / 1000),
        cache_hit_rate=cache_hits / num_queries,
    )


async def benchmark_batch_queries(
    retriever: TurboRetriever,
    batch_sizes: List[int] = [10, 50, 100]
) -> Dict[int, BenchmarkResult]:
    """Benchmark batch query performance."""
    queries = generate_test_queries()
    results = {}
    
    for batch_size in batch_sizes:
        batch = [random.choice(queries) for _ in range(batch_size)]
        
        latencies = []
        for _ in range(10):  # 10 iterations per batch size
            start = time.perf_counter()
            await retriever.batch_query(batch, k=10)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        avg_per_query = statistics.mean(latencies) / batch_size
        
        results[batch_size] = BenchmarkResult(
            name=f"Batch ({batch_size})",
            total_queries=batch_size * 10,
            total_time_ms=sum(latencies),
            avg_latency_ms=avg_per_query,
            p50_latency_ms=statistics.median(latencies) / batch_size,
            p95_latency_ms=sorted(latencies)[8] / batch_size,
            p99_latency_ms=sorted(latencies)[9] / batch_size,
            min_latency_ms=min(latencies) / batch_size,
            max_latency_ms=max(latencies) / batch_size,
            qps=(batch_size * 10) / (sum(latencies) / 1000),
            cache_hit_rate=0.0,
        )
    
    return results


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n  {result.name}:")
    print(f"    Queries: {result.total_queries}")
    print(f"    Avg Latency: {result.avg_latency_ms:.3f}ms")
    print(f"    P50 Latency: {result.p50_latency_ms:.3f}ms")
    print(f"    P95 Latency: {result.p95_latency_ms:.3f}ms")
    print(f"    P99 Latency: {result.p99_latency_ms:.3f}ms")
    print(f"    Min/Max: {result.min_latency_ms:.3f}ms / {result.max_latency_ms:.3f}ms")
    print(f"    QPS: {result.qps:.1f}")
    if result.cache_hit_rate > 0:
        print(f"    Cache Hit Rate: {result.cache_hit_rate * 100:.1f}%")


def print_competitor_comparison(our_result: BenchmarkResult):
    """Print comparison against competitors."""
    print("\n" + "=" * 70)
    print("COMPETITOR COMPARISON")
    print("=" * 70)
    
    our_latency = our_result.avg_latency_ms
    our_qps = our_result.qps
    
    print(f"\n{'System':<15} {'Latency':<12} {'QPS':<10} {'vs LoopMind':<20}")
    print("-" * 60)
    print(f"{'LoopMind':<15} {our_latency:<12.1f} {our_qps:<10.1f} {'(baseline)':<20}")
    
    for name, metrics in COMPETITORS.items():
        latency_diff = metrics["latency_ms"] / our_latency
        qps_diff = our_qps / metrics["qps"]
        
        status = ""
        if our_latency < metrics["latency_ms"]:
            status = f"✅ {latency_diff:.1f}x FASTER"
        else:
            status = f"❌ {1/latency_diff:.1f}x slower"
        
        print(f"{name:<15} {metrics['latency_ms']:<12.0f} {metrics['qps']:<10.0f} {status:<20}")
    
    print("\n" + "=" * 70)
    
    # Summary
    beaten = sum(1 for m in COMPETITORS.values() if our_latency < m["latency_ms"])
    print(f"\n📊 SUMMARY: LoopMind beats {beaten}/{len(COMPETITORS)} competitors on latency!")
    
    if our_latency < 10:
        print("🚀 TURBO MODE: Sub-10ms latency achieved!")
    elif our_latency < 50:
        print("⚡ FAST MODE: Sub-50ms latency achieved!")
    elif our_latency < 100:
        print("✅ TARGET: Sub-100ms latency achieved!")
    
    fastest_competitor = min(COMPETITORS.items(), key=lambda x: x[1]["latency_ms"])
    if our_latency < fastest_competitor[1]["latency_ms"]:
        speedup = fastest_competitor[1]["latency_ms"] / our_latency
        print(f"🏆 FASTER than {fastest_competitor[0]} by {speedup:.1f}x!")


async def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("LOOPMIND SPEED BENCHMARK")
    print("=" * 70)
    
    # 1. Embedding generation benchmark
    print("\n📊 Benchmark 1: Synthetic Embedding Generation")
    emb_result = await benchmark_embedding_generation(num_samples=1000)
    print_result(emb_result)
    
    # 2. Main retrieval benchmark
    print("\n📊 Benchmark 2: Memory Retrieval (TurboRetriever)")
    retrieval_result = await benchmark_retrieval(
        num_indexed=5000,
        num_queries=500,
        warmup_queries=50
    )
    print_result(retrieval_result)
    
    # 3. Print competitor comparison
    print_competitor_comparison(retrieval_result)
    
    # Final stats
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n  Embedding Generation:")
    print(f"    Avg: {emb_result.avg_latency_ms:.3f}ms")
    print(f"    QPS: {emb_result.qps:.0f}")
    print(f"\n  Memory Retrieval:")
    print(f"    Avg: {retrieval_result.avg_latency_ms:.3f}ms")
    print(f"    P95: {retrieval_result.p95_latency_ms:.3f}ms")
    print(f"    QPS: {retrieval_result.qps:.0f}")
    print(f"    Cache Hit Rate: {retrieval_result.cache_hit_rate * 100:.1f}%")
    
    # Targets check
    print("\n📋 TARGET COMPLIANCE:")
    targets = [
        ("Avg Latency < 10ms", retrieval_result.avg_latency_ms < 10),
        ("P95 Latency < 50ms", retrieval_result.p95_latency_ms < 50),
        ("QPS > 350", retrieval_result.qps > 350),
        ("Faster than OpenMemory", retrieval_result.avg_latency_ms < 115),
    ]
    
    for target, passed in targets:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{status}] {target}")


if __name__ == "__main__":
    asyncio.run(main())



