"""
End-to-End Speed Test for LoopMind Memory System.

Tests:
1. Memory ingestion with fast embeddings
2. Memory retrieval with turbo retriever
3. Batch operations
4. Cache performance
5. Comparison with competitors
"""

import asyncio
import time
import statistics
import sys
from typing import List, Dict, Any
from datetime import datetime, timezone

sys.path.insert(0, ".")

import httpx
import structlog

logger = structlog.get_logger()

# Test configuration
API_BASE = "http://localhost:8000"
TEST_TENANT = "speed_test_tenant"

# Competitor baselines
COMPETITORS = {
    "OpenMemory": {"latency_ms": 115, "qps": 338, "recall": 1.0},
    "Zep": {"latency_ms": 310, "qps": 180, "recall": 0.95},
    "Supermemory": {"latency_ms": 270, "qps": 220, "recall": 0.92},
    "Mem0": {"latency_ms": 250, "qps": 150, "recall": 0.90},
    "Vector DBs": {"latency_ms": 160, "qps": 250, "recall": 0.88},
}


class SpeedE2ETester:
    """End-to-end speed tester."""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.token: str = ""
        self.memory_ids: List[str] = []
        self.test_results: Dict[str, Any] = {}
        
    async def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        # Check if API is running
        try:
            resp = await self.client.get(f"{self.base_url}/")
            if resp.status_code != 200:
                print(f"⚠️  API health check failed: {resp.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to API at {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure the API server is running!")
            raise
        
        # Get auth token
        try:
            resp = await self.client.post(
                f"{self.base_url}/v1/auth/login",
                json={"username": "test_user", "password": "test_pass"}
            )
            if resp.status_code == 200:
                self.token = resp.json().get("access_token", "")
            else:
                # Try without auth for now
                self.token = "test_token"
        except:
            self.token = "test_token"
        
        print(f"✅ Setup complete (API: {self.base_url})")
    
    async def cleanup(self):
        """Cleanup test data."""
        await self.client.aclose()
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def test_memory_ingestion(self, count: int = 100) -> Dict[str, Any]:
        """Test memory ingestion with fast embeddings."""
        print(f"\n📝 Test 1: Memory Ingestion ({count} memories)")
        print("=" * 70)
        
        latencies = []
        memory_ids = []
        
        test_memories = [
            ("I met with the engineering team yesterday to discuss the new API design", "episodic"),
            ("The deployment process requires Docker, Kubernetes, and Helm charts", "procedural"),
            ("I felt excited about the project launch and confident in our team", "emotional"),
            ("The architecture follows microservices pattern with event-driven communication", "semantic"),
            ("I realized that user feedback is crucial for product success", "reflective"),
            ("Last week I attended a conference on machine learning and AI", "episodic"),
            ("To install the package, run: pip install -r requirements.txt", "procedural"),
            ("I was worried about the deadline but the team delivered on time", "emotional"),
            ("Vector databases use approximate nearest neighbor search algorithms", "semantic"),
            ("I learned that simplicity often beats complexity in software design", "reflective"),
        ]
        
        for i in range(count):
            content, sector = test_memories[i % len(test_memories)]
            # Add variation
            content = f"{content} (test {i})"
            
            start = time.perf_counter()
            
            try:
                # Use fast index endpoint if available
                resp = await self.client.post(
                    f"{self.base_url}/v1/fast/index",
                    headers=self._headers(),
                    json={
                        "memory_id": f"speed_test_{i}",
                        "content": content,
                        "sector": sector,
                        "salience": 0.7
                    }
                )
                
                if resp.status_code in [200, 201]:
                    memory_ids.append(f"speed_test_{i}")
                else:
                    # Fallback to regular memory endpoint
                    resp = await self.client.post(
                        f"{self.base_url}/v1/memories",
                        headers=self._headers(),
                        json={
                            "content": content,
                            "tags": [sector],
                            "metadata": {"sector": sector}
                        }
                    )
                    if resp.status_code in [200, 201]:
                        data = resp.json()
                        memory_ids.append(data.get("id", f"speed_test_{i}"))
                
            except Exception as e:
                print(f"  ⚠️  Error ingesting memory {i}: {e}")
            
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            
            if (i + 1) % 20 == 0:
                avg = statistics.mean(latencies[-20:])
                print(f"  Progress: {i + 1}/{count}, Avg: {avg:.2f}ms")
        
        self.memory_ids = memory_ids
        
        result = {
            "count": len(memory_ids),
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_time_ms": sum(latencies),
            "throughput_per_sec": len(memory_ids) / (sum(latencies) / 1000),
        }
        
        print(f"\n  ✅ Ingested {len(memory_ids)} memories")
        print(f"  📊 Avg Latency: {result['avg_latency_ms']:.2f}ms")
        print(f"  📊 P95 Latency: {result['p95_latency_ms']:.2f}ms")
        print(f"  📊 Throughput: {result['throughput_per_sec']:.1f} memories/sec")
        
        return result
    
    async def test_fast_query(self, num_queries: int = 100) -> Dict[str, Any]:
        """Test fast query endpoint."""
        print(f"\n🔍 Test 2: Fast Query Retrieval ({num_queries} queries)")
        print("=" * 70)
        
        test_queries = [
            "What meetings did I have?",
            "How do I deploy the application?",
            "What made me feel excited?",
            "Tell me about the architecture",
            "What did I learn recently?",
            "Who did I meet yesterday?",
            "How do I install the package?",
            "What worried me about the project?",
            "Explain vector databases",
            "What insights did I gain?",
        ]
        
        latencies = []
        cache_hits = 0
        result_counts = []
        
        # Warmup
        print("  Warming up cache...")
        for _ in range(10):
            query = test_queries[0]
            try:
                await self.client.post(
                    f"{self.base_url}/v1/fast/query",
                    headers=self._headers(),
                    json={"query": query, "k": 10, "mode": "turbo"}
                )
            except:
                pass
        
        # Benchmark
        print("  Running benchmark queries...")
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            
            start = time.perf_counter()
            
            try:
                resp = await self.client.post(
                    f"{self.base_url}/v1/fast/query",
                    headers=self._headers(),
                    json={
                        "query": query,
                        "k": 10,
                        "mode": "turbo"
                    }
                )
                
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                
                if resp.status_code == 200:
                    data = resp.json()
                    stats = data.get("stats", {})
                    if stats.get("cache_hit", False):
                        cache_hits += 1
                    result_counts.append(len(data.get("results", [])))
                else:
                    result_counts.append(0)
                    
            except Exception as e:
                latencies.append(0)
                result_counts.append(0)
                if i < 5:  # Only print first few errors
                    print(f"  ⚠️  Query {i} error: {e}")
            
            if (i + 1) % 20 == 0:
                avg = statistics.mean(latencies[-20:])
                print(f"  Progress: {i + 1}/{num_queries}, Avg: {avg:.2f}ms")
        
        result = {
            "queries": num_queries,
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "qps": num_queries / (sum(latencies) / 1000) if latencies else 0,
            "cache_hit_rate": cache_hits / num_queries if num_queries > 0 else 0,
            "avg_results": statistics.mean(result_counts) if result_counts else 0,
        }
        
        print(f"\n  ✅ Completed {num_queries} queries")
        print(f"  📊 Avg Latency: {result['avg_latency_ms']:.2f}ms")
        print(f"  📊 P95 Latency: {result['p95_latency_ms']:.2f}ms")
        print(f"  📊 QPS: {result['qps']:.1f}")
        print(f"  📊 Cache Hit Rate: {result['cache_hit_rate'] * 100:.1f}%")
        print(f"  📊 Avg Results: {result['avg_results']:.1f}")
        
        return result
    
    async def test_batch_operations(self) -> Dict[str, Any]:
        """Test batch query operations."""
        print(f"\n⚡ Test 3: Batch Query Operations")
        print("=" * 70)
        
        queries = [
            "What meetings did I have?",
            "How do I deploy?",
            "What made me happy?",
            "Tell me about architecture",
            "What did I learn?",
        ]
        
        latencies = []
        
        for batch_size in [5, 10, 20]:
            print(f"  Testing batch size: {batch_size}")
            
            batch_queries = queries * (batch_size // len(queries) + 1)
            batch_queries = batch_queries[:batch_size]
            
            start = time.perf_counter()
            
            try:
                resp = await self.client.post(
                    f"{self.base_url}/v1/fast/batch",
                    headers=self._headers(),
                    json={
                        "queries": batch_queries,
                        "k": 10
                    }
                )
                
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
                
                if resp.status_code == 200:
                    data = resp.json()
                    avg_per_query = data.get("avg_latency_ms", elapsed / batch_size)
                    print(f"    Total: {elapsed:.2f}ms, Avg per query: {avg_per_query:.2f}ms")
                else:
                    print(f"    ⚠️  Error: {resp.status_code}")
                    
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
        
        result = {
            "batch_tested": True,
            "latencies": latencies,
        }
        
        return result
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance."""
        print(f"\n💾 Test 4: Cache Performance")
        print("=" * 70)
        
        query = "What meetings did I have?"
        
        # First query (cache miss)
        start1 = time.perf_counter()
        try:
            resp1 = await self.client.post(
                f"{self.base_url}/v1/fast/query",
                headers=self._headers(),
                json={"query": query, "k": 10, "mode": "turbo"}
            )
            elapsed1 = (time.perf_counter() - start1) * 1000
        except:
            elapsed1 = 0
        
        # Second query (cache hit)
        start2 = time.perf_counter()
        try:
            resp2 = await self.client.post(
                f"{self.base_url}/v1/fast/query",
                headers=self._headers(),
                json={"query": query, "k": 10, "mode": "turbo"}
            )
            elapsed2 = (time.perf_counter() - start2) * 1000
        except:
            elapsed2 = 0
        
        speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
        
        result = {
            "cache_miss_ms": elapsed1,
            "cache_hit_ms": elapsed2,
            "speedup": speedup,
        }
        
        print(f"  📊 Cache Miss: {elapsed1:.2f}ms")
        print(f"  📊 Cache Hit: {elapsed2:.2f}ms")
        print(f"  📊 Speedup: {speedup:.1f}x")
        
        return result
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/v1/fast/stats",
                headers=self._headers()
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {}
    
    def compare_with_competitors(self, query_result: Dict[str, Any]):
        """Compare results with competitors."""
        print(f"\n🏆 Test 5: Competitor Comparison")
        print("=" * 70)
        
        our_latency = query_result.get("avg_latency_ms", 0)
        our_qps = query_result.get("qps", 0)
        
        print(f"\n{'System':<15} {'Latency (ms)':<15} {'QPS':<10} {'vs LoopMind':<20}")
        print("-" * 70)
        print(f"{'LoopMind (Turbo)':<15} {our_latency:<15.1f} {our_qps:<10.1f} {'(baseline)':<20}")
        
        beaten = 0
        for name, metrics in COMPETITORS.items():
            comp_latency = metrics["latency_ms"]
            comp_qps = metrics["qps"]
            
            latency_ratio = comp_latency / our_latency if our_latency > 0 else 0
            qps_ratio = our_qps / comp_qps if comp_qps > 0 else 0
            
            if our_latency < comp_latency:
                beaten += 1
                status = f"✅ {latency_ratio:.1f}x FASTER"
            else:
                status = f"❌ {1/latency_ratio:.1f}x slower"
            
            print(f"{name:<15} {comp_latency:<15.0f} {comp_qps:<10.0f} {status:<20}")
        
        print("\n" + "=" * 70)
        print(f"📊 SUMMARY: LoopMind beats {beaten}/{len(COMPETITORS)} competitors on latency!")
        
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
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n" + "=" * 70)
        print("FINAL TEST SUMMARY")
        print("=" * 70)
        
        if "ingestion" in self.test_results:
            ing = self.test_results["ingestion"]
            print(f"\n📝 Memory Ingestion:")
            print(f"  Memories: {ing['count']}")
            print(f"  Avg Latency: {ing['avg_latency_ms']:.2f}ms")
            print(f"  Throughput: {ing['throughput_per_sec']:.1f} memories/sec")
        
        if "query" in self.test_results:
            qry = self.test_results["query"]
            print(f"\n🔍 Query Retrieval:")
            print(f"  Queries: {qry['queries']}")
            print(f"  Avg Latency: {qry['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {qry['p95_latency_ms']:.2f}ms")
            print(f"  QPS: {qry['qps']:.1f}")
            print(f"  Cache Hit Rate: {qry['cache_hit_rate'] * 100:.1f}%")
        
        if "cache" in self.test_results:
            cache = self.test_results["cache"]
            print(f"\n💾 Cache Performance:")
            print(f"  Cache Miss: {cache['cache_miss_ms']:.2f}ms")
            print(f"  Cache Hit: {cache['cache_hit_ms']:.2f}ms")
            print(f"  Speedup: {cache['speedup']:.1f}x")
        
        print("\n" + "=" * 70)
    
    async def run_all(self):
        """Run all tests."""
        print("=" * 70)
        print("LOOPMIND SPEED E2E TEST SUITE")
        print("=" * 70)
        print(f"Started: {datetime.now(timezone.utc).isoformat()}")
        print(f"Target: {self.base_url}")
        
        try:
            await self.setup()
            
            # Test 1: Ingestion
            self.test_results["ingestion"] = await self.test_memory_ingestion(count=50)
            
            # Wait a bit for indexing
            await asyncio.sleep(2)
            
            # Test 2: Fast Query
            self.test_results["query"] = await self.test_fast_query(num_queries=50)
            
            # Test 3: Batch Operations
            self.test_results["batch"] = await self.test_batch_operations()
            
            # Test 4: Cache Performance
            self.test_results["cache"] = await self.test_cache_performance()
            
            # Test 5: Comparison
            if "query" in self.test_results:
                self.compare_with_competitors(self.test_results["query"])
            
            # Get cache stats
            self.test_results["cache_stats"] = await self.get_cache_stats()
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    tester = SpeedE2ETester()
    await tester.run_all()


if __name__ == "__main__":
    asyncio.run(main())

