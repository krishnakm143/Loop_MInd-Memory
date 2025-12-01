"""Scalability Benchmarks.

Tests system performance at different scales:
- 1K, 10K, 100K, 1M memories
- Memory usage tracking
- Index rebuild time
- Query latency vs dataset size
"""

import asyncio
import random
import time
import gc
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import httpx
import psutil

from benchmarks.base import (
    BenchmarkRunner, Timer, BenchmarkResult, LatencyStats, ThroughputStats
)
from benchmarks.config import BenchmarkConfig, SAMPLE_MEMORIES, SAMPLE_QUERIES, get_default_config


@dataclass
class ScaleMetrics:
    """Metrics for scalability testing."""
    dataset_size: int
    memory_usage_mb: float
    index_time_seconds: float
    query_latency_ms: float
    throughput_qps: float
    ms_per_item: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "index_time_seconds": round(self.index_time_seconds, 2),
            "query_latency_ms": round(self.query_latency_ms, 2),
            "throughput_qps": round(self.throughput_qps, 2),
            "ms_per_item": round(self.ms_per_item, 4),
        }


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def generate_memory_content(index: int) -> str:
    """Generate unique memory content."""
    base = random.choice(SAMPLE_MEMORIES)
    return f"{base} (Entry #{index}, ID: {random.randint(1000, 9999)})"


class ScalabilityBenchmark(BenchmarkRunner):
    """Benchmark system scalability with different dataset sizes."""
    
    def __init__(
        self,
        config: BenchmarkConfig = None,
        target_size: int = 1000,
        batch_size: int = 100,
    ):
        super().__init__(f"scale_{target_size}", config or get_default_config())
        self.target_size = target_size
        self.batch_size = batch_size
        self.client: Optional[httpx.AsyncClient] = None
        self.created_count = 0
        self.scale_metrics: Optional[ScaleMetrics] = None
        
    async def setup(self):
        """Set up HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=120.0,  # Longer timeout for large batches
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        self.created_count = 0
        
    async def populate_data(self) -> float:
        """Populate database with target number of memories.
        
        Returns:
            Time taken in seconds
        """
        start_time = time.perf_counter()
        
        while self.created_count < self.target_size:
            # Create batch of memories
            batch_size = min(self.batch_size, self.target_size - self.created_count)
            memories = []
            
            for i in range(batch_size):
                content = generate_memory_content(self.created_count + i)
                memories.append({
                    "content": content,
                    "tags": ["scale_test", f"batch_{self.created_count // self.batch_size}"],
                })
            
            payload = {"memories": memories}
            
            try:
                response = await self.client.post("/v1/batch/memories", json=payload)
                if response.is_success:
                    self.created_count += batch_size
                else:
                    # Fall back to single creation
                    for memory in memories:
                        response = await self.client.post("/v1/memories", json=memory)
                        if response.is_success:
                            self.created_count += 1
            except Exception as e:
                print(f"Batch creation error: {e}")
                await asyncio.sleep(1)
            
            # Progress
            if self.created_count % 1000 == 0:
                print(f"  Created {self.created_count}/{self.target_size} memories...")
        
        return time.perf_counter() - start_time
        
    async def run_iteration(self, iteration: int) -> float:
        """Run a query and measure latency."""
        query = random.choice(SAMPLE_QUERIES)
        
        with Timer() as timer:
            response = await self.client.post("/v1/query", json={
                "query_text": query,
                "limit": 10,
            })
            
        if not response.is_success:
            raise Exception(f"Query failed: {response.status_code}")
            
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()
    
    async def run_scale_test(self, query_iterations: int = 100) -> BenchmarkResult:
        """Run complete scalability test.
        
        Args:
            query_iterations: Number of queries to run for latency measurement
            
        Returns:
            BenchmarkResult with scale metrics
        """
        from datetime import datetime
        
        started_at = datetime.utcnow()
        initial_memory = get_memory_usage_mb()
        
        await self.setup()
        
        # Populate data
        print(f"Populating {self.target_size} memories...")
        index_time = await self.populate_data()
        
        # Measure memory after population
        gc.collect()
        await asyncio.sleep(1)
        memory_after = get_memory_usage_mb()
        memory_increase = memory_after - initial_memory
        
        # Run query benchmark
        print(f"Running {query_iterations} query iterations...")
        latencies = []
        start_time = time.perf_counter()
        
        for i in range(query_iterations):
            try:
                latency = await self.run_iteration(i)
                latencies.append(latency)
            except Exception:
                pass
        
        query_duration = time.perf_counter() - start_time
        
        await self.teardown()
        
        # Calculate metrics
        latency_stats = LatencyStats.from_samples(latencies) if latencies else None
        throughput = len(latencies) / query_duration if query_duration > 0 else 0
        ms_per_item = index_time * 1000 / self.target_size if self.target_size > 0 else 0
        
        self.scale_metrics = ScaleMetrics(
            dataset_size=self.target_size,
            memory_usage_mb=memory_increase,
            index_time_seconds=index_time,
            query_latency_ms=latency_stats.mean_ms if latency_stats else 0,
            throughput_qps=throughput,
            ms_per_item=ms_per_item,
        )
        
        result = BenchmarkResult(
            benchmark_name=self.name,
            status="completed",
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_seconds=index_time + query_duration,
            latency=latency_stats,
            throughput=ThroughputStats.calculate(
                total=query_iterations,
                successful=len(latencies),
                duration_seconds=query_duration,
            ),
            metrics={
                "scale": self.scale_metrics.to_dict(),
                "dataset_size": self.target_size,
                "index_time_seconds": index_time,
                "ms_per_item": ms_per_item,
            },
            meets_targets={
                "ms_per_item": ms_per_item < 15.0,  # Target: <15ms/item
            },
        )
        
        return result


class IncrementalScaleBenchmark:
    """Run scalability benchmark at multiple dataset sizes."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or get_default_config()
        self.sizes = [1000, 5000, 10000, 50000, 100000]
        self.results: Dict[int, ScaleMetrics] = {}
        
    async def run(self, max_size: int = None) -> Dict[str, BenchmarkResult]:
        """Run benchmarks at each scale.
        
        Args:
            max_size: Maximum dataset size to test (default: use all)
            
        Returns:
            Dictionary of results by size
        """
        results = {}
        sizes = [s for s in self.sizes if max_size is None or s <= max_size]
        
        for size in sizes:
            print(f"\n=== Testing scale: {size} ===")
            bench = ScalabilityBenchmark(
                self.config,
                target_size=size,
                batch_size=min(100, size // 10),
            )
            results[f"scale_{size}"] = await bench.run_scale_test(query_iterations=200)
            
            if bench.scale_metrics:
                self.results[size] = bench.scale_metrics
        
        return results
    
    def get_scaling_analysis(self) -> Dict[str, Any]:
        """Analyze scaling characteristics."""
        if len(self.results) < 2:
            return {"error": "Need at least 2 data points for analysis"}
        
        sizes = sorted(self.results.keys())
        
        # Calculate scaling factors
        latency_growth = []
        throughput_growth = []
        
        for i in range(1, len(sizes)):
            prev = self.results[sizes[i-1]]
            curr = self.results[sizes[i]]
            size_ratio = sizes[i] / sizes[i-1]
            
            latency_ratio = curr.query_latency_ms / prev.query_latency_ms if prev.query_latency_ms > 0 else 1
            throughput_ratio = curr.throughput_qps / prev.throughput_qps if prev.throughput_qps > 0 else 1
            
            latency_growth.append(latency_ratio / size_ratio)
            throughput_growth.append(throughput_ratio)
        
        return {
            "sizes_tested": sizes,
            "avg_ms_per_item": sum(r.ms_per_item for r in self.results.values()) / len(self.results),
            "latency_growth_factor": sum(latency_growth) / len(latency_growth) if latency_growth else 1,
            "throughput_degradation": sum(throughput_growth) / len(throughput_growth) if throughput_growth else 1,
            "scales_sublinearly": all(g < 1.5 for g in latency_growth),  # Latency doesn't grow faster than O(n)
        }


async def run_scale_benchmarks(
    config: BenchmarkConfig = None,
    max_size: int = 10000,
) -> Dict[str, BenchmarkResult]:
    """Run scalability benchmarks.
    
    Args:
        config: Benchmark configuration
        max_size: Maximum dataset size to test
        
    Returns:
        Dictionary of benchmark results
    """
    config = config or get_default_config()
    
    print("Running scalability benchmarks...")
    bench = IncrementalScaleBenchmark(config)
    results = await bench.run(max_size=max_size)
    
    # Add scaling analysis
    analysis = bench.get_scaling_analysis()
    print(f"\nScaling Analysis: {analysis}")
    
    return results


if __name__ == "__main__":
    import asyncio
    
    results = asyncio.run(run_scale_benchmarks(max_size=10000))
    
    print("\n=== Scalability Benchmark Results ===")
    for name, result in results.items():
        print(f"\n{name}:")
        if result.metrics.get("scale"):
            scale = result.metrics["scale"]
            print(f"  Dataset size: {scale['dataset_size']}")
            print(f"  Index time: {scale['index_time_seconds']:.2f}s")
            print(f"  ms/item: {scale['ms_per_item']:.4f}")
            print(f"  Query latency: {scale['query_latency_ms']:.2f}ms")
            print(f"  Throughput: {scale['throughput_qps']:.2f} QPS")



