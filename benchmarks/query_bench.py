"""Query Benchmarks.

Benchmarks for query operations including:
- Simple queries
- Filtered queries
- Batch queries
- Query with waypoints
"""

import asyncio
import uuid
import random
from typing import Dict, Any, List, Optional
import httpx

from benchmarks.base import (
    BenchmarkRunner, ConcurrentBenchmarkRunner, Timer,
    BenchmarkResult, LatencyStats
)
from benchmarks.config import BenchmarkConfig, SAMPLE_MEMORIES, SAMPLE_QUERIES, get_default_config


class QueryBenchmark(BenchmarkRunner):
    """Benchmark query latency."""
    
    def __init__(self, config: BenchmarkConfig = None, include_waypoints: bool = False):
        name = "query_with_waypoints" if include_waypoints else "query_simple"
        super().__init__(name, config or get_default_config())
        self.include_waypoints = include_waypoints
        self.client: Optional[httpx.AsyncClient] = None
        
    async def setup(self):
        """Set up HTTP client and seed data."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Seed some memories for querying
        for memory in SAMPLE_MEMORIES:
            payload = {"content": memory, "tags": ["benchmark", "seed"]}
            await self.client.post("/v1/memories", json=payload)
        
    async def run_iteration(self, iteration: int) -> float:
        """Execute a query and measure latency."""
        query = random.choice(SAMPLE_QUERIES)
        
        payload = {
            "query_text": query,
            "limit": 10,
            "include_waypoints": self.include_waypoints,
        }
        
        with Timer() as timer:
            response = await self.client.post("/v1/query", json=payload)
            
        if not response.is_success:
            raise Exception(f"Query failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


class FilteredQueryBenchmark(BenchmarkRunner):
    """Benchmark filtered query latency."""
    
    def __init__(self, config: BenchmarkConfig = None):
        super().__init__("query_filtered", config or get_default_config())
        self.client: Optional[httpx.AsyncClient] = None
        self.sectors = ["episodic", "semantic", "procedural", "emotional", "workspace"]
        
    async def setup(self):
        """Set up HTTP client and seed data with sectors."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Seed memories in different sectors
        for i, memory in enumerate(SAMPLE_MEMORIES):
            payload = {
                "content": memory,
                "tags": ["benchmark", f"sector_{self.sectors[i % len(self.sectors)]}"],
            }
            await self.client.post("/v1/memories", json=payload)
        
    async def run_iteration(self, iteration: int) -> float:
        """Execute a filtered query."""
        query = random.choice(SAMPLE_QUERIES)
        sector = random.choice(self.sectors)
        
        payload = {
            "query_text": query,
            "limit": 10,
            "sectors": [sector],
            "filters": {
                "tags": ["benchmark"],
            },
        }
        
        with Timer() as timer:
            response = await self.client.post("/v1/query", json=payload)
            
        if not response.is_success:
            raise Exception(f"Filtered query failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


class BatchQueryBenchmark(BenchmarkRunner):
    """Benchmark batch query operations."""
    
    def __init__(self, config: BenchmarkConfig = None, batch_size: int = 5):
        super().__init__(f"query_batch_{batch_size}", config or get_default_config())
        self.batch_size = batch_size
        self.client: Optional[httpx.AsyncClient] = None
        
    async def setup(self):
        """Set up HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Seed data
        for memory in SAMPLE_MEMORIES:
            payload = {"content": memory, "tags": ["benchmark"]}
            await self.client.post("/v1/memories", json=payload)
        
    async def run_iteration(self, iteration: int) -> float:
        """Execute a batch query."""
        queries = random.sample(SAMPLE_QUERIES, min(self.batch_size, len(SAMPLE_QUERIES)))
        
        payload = {
            "queries": [
                {"query_text": q, "limit": 10}
                for q in queries
            ],
        }
        
        with Timer() as timer:
            response = await self.client.post("/v1/batch/queries", json=payload)
            
        if not response.is_success:
            raise Exception(f"Batch query failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


class ConcurrentQueryBenchmark(ConcurrentBenchmarkRunner):
    """Benchmark queries under concurrent load."""
    
    def __init__(self, config: BenchmarkConfig = None, concurrency: int = 10):
        super().__init__(
            f"query_concurrent_{concurrency}",
            config or get_default_config(),
            concurrency,
        )
        self.client: Optional[httpx.AsyncClient] = None
        
    async def setup(self):
        """Set up HTTP client and seed data."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Seed data
        for memory in SAMPLE_MEMORIES * 10:  # 100 memories
            payload = {"content": memory, "tags": ["benchmark"]}
            await self.client.post("/v1/memories", json=payload)
        
    async def run_iteration(self, iteration: int) -> float:
        """Execute a query."""
        query = random.choice(SAMPLE_QUERIES)
        
        payload = {
            "query_text": query,
            "limit": 10,
        }
        
        with Timer() as timer:
            response = await self.client.post("/v1/query", json=payload)
            
        if not response.is_success:
            raise Exception(f"Query failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


async def run_query_benchmarks(config: BenchmarkConfig = None) -> Dict[str, BenchmarkResult]:
    """Run all query benchmarks.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary of benchmark results
    """
    config = config or get_default_config()
    results = {}
    
    # Simple query
    print("Running simple query benchmark...")
    bench = QueryBenchmark(config, include_waypoints=False)
    results["query_simple"] = await bench.run(iterations=500)
    
    # Query with waypoints
    print("Running query with waypoints benchmark...")
    bench = QueryBenchmark(config, include_waypoints=True)
    results["query_waypoints"] = await bench.run(iterations=500)
    
    # Filtered query
    print("Running filtered query benchmark...")
    bench = FilteredQueryBenchmark(config)
    results["query_filtered"] = await bench.run(iterations=500)
    
    # Batch queries
    for batch_size in [5, 10, 20]:
        print(f"Running batch query benchmark (size={batch_size})...")
        bench = BatchQueryBenchmark(config, batch_size=batch_size)
        results[f"query_batch_{batch_size}"] = await bench.run(iterations=100)
    
    # Concurrent queries
    for concurrency in [10, 50, 100]:
        print(f"Running concurrent query benchmark (c={concurrency})...")
        bench = ConcurrentQueryBenchmark(config, concurrency=concurrency)
        results[f"query_concurrent_{concurrency}"] = await bench.run_concurrent(duration_seconds=30)
    
    return results


if __name__ == "__main__":
    import asyncio
    results = asyncio.run(run_query_benchmarks())
    
    print("\n=== Query Benchmark Results ===")
    for name, result in results.items():
        print(f"\n{name}:")
        if result.latency:
            print(f"  Mean: {result.latency.mean_ms:.2f}ms")
            print(f"  P95: {result.latency.p95_ms:.2f}ms")
        if result.throughput:
            print(f"  QPS: {result.throughput.ops_per_second:.2f}")



