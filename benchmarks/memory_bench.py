"""Memory Operation Benchmarks.

Benchmarks for memory creation, retrieval, update, and deletion operations.
Tests both single operations and batch operations.
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
from benchmarks.config import BenchmarkConfig, SAMPLE_MEMORIES, get_default_config


class MemoryCreationBenchmark(BenchmarkRunner):
    """Benchmark memory creation latency."""
    
    def __init__(self, config: BenchmarkConfig = None):
        super().__init__("memory_creation", config or get_default_config())
        self.client: Optional[httpx.AsyncClient] = None
        self.created_ids: List[str] = []
        
    async def setup(self):
        """Set up HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        self.created_ids = []
        
    async def run_iteration(self, iteration: int) -> float:
        """Create a single memory and measure latency."""
        content = random.choice(SAMPLE_MEMORIES)
        content = f"{content} (iteration {iteration}, uuid: {uuid.uuid4().hex[:8]})"
        
        payload = {
            "content": content,
            "tags": ["benchmark", "test"],
            "metadata": {"iteration": iteration},
        }
        
        with Timer() as timer:
            response = await self.client.post("/v1/memories", json=payload)
            
        if response.is_success:
            data = response.json()
            if "memory_id" in data:
                self.created_ids.append(data["memory_id"])
        else:
            raise Exception(f"Memory creation failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Clean up created memories and close client."""
        # Delete created memories (optional, for cleanup)
        for memory_id in self.created_ids[:100]:  # Limit cleanup
            try:
                await self.client.delete(f"/v1/memories/{memory_id}")
            except Exception:
                pass
        
        if self.client:
            await self.client.aclose()


class MemoryRetrievalBenchmark(BenchmarkRunner):
    """Benchmark memory retrieval by ID."""
    
    def __init__(self, config: BenchmarkConfig = None, memory_ids: List[str] = None):
        super().__init__("memory_retrieval", config or get_default_config())
        self.client: Optional[httpx.AsyncClient] = None
        self.memory_ids = memory_ids or []
        
    async def setup(self):
        """Set up HTTP client and create test memories if needed."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Create test memories if none provided
        if not self.memory_ids:
            for i in range(100):
                content = random.choice(SAMPLE_MEMORIES)
                payload = {"content": content, "tags": ["benchmark"]}
                response = await self.client.post("/v1/memories", json=payload)
                if response.is_success:
                    data = response.json()
                    if "memory_id" in data:
                        self.memory_ids.append(data["memory_id"])
        
    async def run_iteration(self, iteration: int) -> float:
        """Retrieve a memory by ID."""
        if not self.memory_ids:
            raise Exception("No memory IDs available")
            
        memory_id = random.choice(self.memory_ids)
        
        with Timer() as timer:
            response = await self.client.get(f"/v1/memories/{memory_id}")
            
        if not response.is_success:
            raise Exception(f"Memory retrieval failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


class BatchMemoryBenchmark(BenchmarkRunner):
    """Benchmark batch memory operations."""
    
    def __init__(self, config: BenchmarkConfig = None, batch_size: int = 10):
        super().__init__(f"batch_memory_{batch_size}", config or get_default_config())
        self.batch_size = batch_size
        self.client: Optional[httpx.AsyncClient] = None
        
    async def setup(self):
        """Set up HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=60.0,  # Longer timeout for batch
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
    async def run_iteration(self, iteration: int) -> float:
        """Create a batch of memories."""
        memories = []
        for i in range(self.batch_size):
            content = random.choice(SAMPLE_MEMORIES)
            memories.append({
                "content": f"{content} (batch {iteration}, item {i})",
                "tags": ["benchmark", "batch"],
            })
        
        payload = {"memories": memories}
        
        with Timer() as timer:
            response = await self.client.post("/v1/batch/memories", json=payload)
            
        if not response.is_success:
            raise Exception(f"Batch creation failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


class ConcurrentMemoryBenchmark(ConcurrentBenchmarkRunner):
    """Benchmark memory operations under concurrent load."""
    
    def __init__(
        self,
        config: BenchmarkConfig = None,
        concurrency: int = 10,
        operation: str = "create",
    ):
        super().__init__(
            f"concurrent_memory_{operation}_{concurrency}",
            config or get_default_config(),
            concurrency,
        )
        self.operation = operation
        self.client: Optional[httpx.AsyncClient] = None
        self.memory_ids: List[str] = []
        
    async def setup(self):
        """Set up HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Pre-create memories for read operations
        if self.operation in ["read", "mixed"]:
            for i in range(100):
                content = random.choice(SAMPLE_MEMORIES)
                payload = {"content": content, "tags": ["benchmark"]}
                response = await self.client.post("/v1/memories", json=payload)
                if response.is_success:
                    data = response.json()
                    if "memory_id" in data:
                        self.memory_ids.append(data["memory_id"])
        
    async def run_iteration(self, iteration: int) -> float:
        """Run a single operation."""
        if self.operation == "create":
            content = random.choice(SAMPLE_MEMORIES)
            payload = {"content": f"{content} ({iteration})", "tags": ["benchmark"]}
            
            with Timer() as timer:
                response = await self.client.post("/v1/memories", json=payload)
                
            if not response.is_success:
                raise Exception(f"Create failed: {response.status_code}")
                
        elif self.operation == "read":
            if not self.memory_ids:
                raise Exception("No memory IDs")
            memory_id = random.choice(self.memory_ids)
            
            with Timer() as timer:
                response = await self.client.get(f"/v1/memories/{memory_id}")
                
            if not response.is_success:
                raise Exception(f"Read failed: {response.status_code}")
                
        elif self.operation == "mixed":
            # 80% reads, 20% writes
            if random.random() < 0.8 and self.memory_ids:
                memory_id = random.choice(self.memory_ids)
                with Timer() as timer:
                    response = await self.client.get(f"/v1/memories/{memory_id}")
            else:
                content = random.choice(SAMPLE_MEMORIES)
                payload = {"content": content, "tags": ["benchmark"]}
                with Timer() as timer:
                    response = await self.client.post("/v1/memories", json=payload)
                    
            if not response.is_success:
                raise Exception(f"Operation failed: {response.status_code}")
        else:
            raise Exception(f"Unknown operation: {self.operation}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


async def run_memory_benchmarks(config: BenchmarkConfig = None) -> Dict[str, BenchmarkResult]:
    """Run all memory benchmarks.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary of benchmark results
    """
    config = config or get_default_config()
    results = {}
    
    # Single operation benchmarks
    print("Running memory creation benchmark...")
    bench = MemoryCreationBenchmark(config)
    results["memory_creation"] = await bench.run(iterations=500)
    
    print("Running memory retrieval benchmark...")
    bench = MemoryRetrievalBenchmark(config)
    results["memory_retrieval"] = await bench.run(iterations=500)
    
    # Batch benchmarks
    for batch_size in [10, 50, 100]:
        print(f"Running batch benchmark (size={batch_size})...")
        bench = BatchMemoryBenchmark(config, batch_size=batch_size)
        results[f"batch_{batch_size}"] = await bench.run(iterations=50)
    
    # Concurrent benchmarks
    for concurrency in [10, 50, 100]:
        print(f"Running concurrent benchmark (c={concurrency})...")
        bench = ConcurrentMemoryBenchmark(config, concurrency=concurrency, operation="mixed")
        results[f"concurrent_{concurrency}"] = await bench.run_concurrent(duration_seconds=30)
    
    return results


if __name__ == "__main__":
    import asyncio
    results = asyncio.run(run_memory_benchmarks())
    
    print("\n=== Memory Benchmark Results ===")
    for name, result in results.items():
        print(f"\n{name}:")
        if result.latency:
            print(f"  Mean: {result.latency.mean_ms:.2f}ms")
            print(f"  P95: {result.latency.p95_ms:.2f}ms")
        if result.throughput:
            print(f"  QPS: {result.throughput.ops_per_second:.2f}")



