"""Accuracy Benchmarks.

Measures retrieval accuracy with:
- Recall @K (K=5, 10, 20)
- Precision @K
- Mean Reciprocal Rank (MRR)
- Cross-sector recall
"""

import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple, Set
import httpx
from dataclasses import dataclass

from benchmarks.base import BenchmarkRunner, Timer, BenchmarkResult, AccuracyStats
from benchmarks.config import BenchmarkConfig, get_default_config


@dataclass
class RetrievalTestCase:
    """A test case for retrieval accuracy."""
    query: str
    relevant_ids: Set[str]  # IDs of memories that should be retrieved
    keywords: List[str]     # Keywords that indicate relevance


# Curated test cases for accuracy evaluation
ACCURACY_TEST_CASES = [
    {
        "theme": "meetings",
        "memories": [
            "Meeting with John scheduled for Monday at 2pm in conference room A",
            "Quarterly review meeting with the team on Friday at 10am",
            "I met Sarah at the networking event last week",
            "The project kickoff meeting is tomorrow morning",
        ],
        "queries": [
            ("What meetings do I have?", [0, 1, 3]),
            ("When is my meeting with John?", [0]),
            ("Who did I meet recently?", [2]),
        ],
    },
    {
        "theme": "technology",
        "memories": [
            "Python 3.12 introduces new performance optimizations",
            "The database migration to PostgreSQL was completed successfully",
            "We're using Redis for caching to improve response times",
            "The new API endpoint supports batch operations",
        ],
        "queries": [
            ("What database are we using?", [1]),
            ("How are we improving performance?", [0, 2]),
            ("What's new with the API?", [3]),
        ],
    },
    {
        "theme": "people",
        "memories": [
            "John is the lead developer on the AI project",
            "Sarah handles customer support and feedback",
            "Mike manages the infrastructure team",
            "Lisa is the project manager for the mobile app",
        ],
        "queries": [
            ("Who is working on AI?", [0]),
            ("Who handles customer issues?", [1]),
            ("Who manages infrastructure?", [2]),
        ],
    },
    {
        "theme": "tasks",
        "memories": [
            "TODO: Complete the quarterly report by Friday",
            "Remember to call the client about the proposal",
            "Need to review the pull request from John",
            "Schedule a demo for the new feature next week",
        ],
        "queries": [
            ("What tasks are due?", [0]),
            ("What calls do I need to make?", [1]),
            ("What code reviews are pending?", [2]),
        ],
    },
]


class AccuracyBenchmark(BenchmarkRunner):
    """Benchmark retrieval accuracy."""
    
    def __init__(self, config: BenchmarkConfig = None, k_values: List[int] = None):
        super().__init__("accuracy", config or get_default_config())
        self.k_values = k_values or [5, 10, 20]
        self.client: Optional[httpx.AsyncClient] = None
        
        # Store test data
        self.memory_ids: Dict[str, str] = {}  # content -> memory_id
        self.test_cases: List[Tuple[str, Set[str]]] = []  # (query, relevant_ids)
        
        # Accuracy metrics
        self.recall_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        self.precision_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        self.reciprocal_ranks: List[float] = []
        
    async def setup(self):
        """Set up HTTP client and create test data."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Create memories from test cases
        for test_group in ACCURACY_TEST_CASES:
            memory_indices = []
            for memory_content in test_group["memories"]:
                payload = {
                    "content": memory_content,
                    "tags": ["accuracy_test", test_group["theme"]],
                }
                response = await self.client.post("/v1/memories", json=payload)
                if response.is_success:
                    data = response.json()
                    memory_id = data.get("memory_id", "")
                    self.memory_ids[memory_content] = memory_id
                    memory_indices.append(memory_id)
            
            # Build test cases with actual IDs
            for query, relevant_indices in test_group["queries"]:
                relevant_ids = set(
                    memory_indices[i]
                    for i in relevant_indices
                    if i < len(memory_indices)
                )
                if relevant_ids:
                    self.test_cases.append((query, relevant_ids))
        
        # Also add some noise memories
        noise_memories = [
            "The weather today is sunny with a high of 75 degrees",
            "My favorite color is blue",
            "The restaurant serves Italian cuisine",
            "Exercise is important for health",
            "Music helps with concentration",
        ]
        for memory in noise_memories:
            payload = {"content": memory, "tags": ["accuracy_test", "noise"]}
            await self.client.post("/v1/memories", json=payload)
        
    async def run_iteration(self, iteration: int) -> float:
        """Run a single accuracy test."""
        if not self.test_cases:
            raise Exception("No test cases available")
        
        # Pick a test case
        query, relevant_ids = self.test_cases[iteration % len(self.test_cases)]
        
        # Execute query
        with Timer() as timer:
            response = await self.client.post("/v1/query", json={
                "query_text": query,
                "limit": max(self.k_values),
            })
        
        if not response.is_success:
            raise Exception(f"Query failed: {response.status_code}")
        
        data = response.json()
        results = data.get("results", [])
        retrieved_ids = [r.get("memory_id", "") for r in results]
        
        # Calculate metrics
        for k in self.k_values:
            top_k = set(retrieved_ids[:k])
            
            # Recall@K = |relevant in top_k| / |relevant|
            recall = len(top_k & relevant_ids) / len(relevant_ids) if relevant_ids else 0
            self.recall_at_k[k].append(recall)
            
            # Precision@K = |relevant in top_k| / k
            precision = len(top_k & relevant_ids) / k
            self.precision_at_k[k].append(precision)
        
        # MRR: 1/rank of first relevant result
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                rr = 1.0 / rank
                break
        self.reciprocal_ranks.append(rr)
        
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()
    
    async def run(
        self,
        iterations: int = None,
        warmup_iterations: int = 0,
        progress_callback=None,
    ) -> BenchmarkResult:
        """Run accuracy benchmark."""
        # Use all test cases if iterations not specified
        if iterations is None:
            iterations = len(self.test_cases) * 3  # Run each test case 3 times
        
        result = await super().run(iterations, warmup_iterations, progress_callback)
        
        # Calculate final accuracy stats
        accuracy_stats = AccuracyStats(
            recall_at_k={
                k: sum(values) / len(values) if values else 0
                for k, values in self.recall_at_k.items()
            },
            precision_at_k={
                k: sum(values) / len(values) if values else 0
                for k, values in self.precision_at_k.items()
            },
            mrr=sum(self.reciprocal_ranks) / len(self.reciprocal_ranks) if self.reciprocal_ranks else 0,
            ndcg=0.0,  # Would need relevance scores to calculate properly
            sample_count=iterations,
        )
        
        result.accuracy = accuracy_stats
        
        # Check against targets
        result.meets_targets = {
            "recall_at_5": accuracy_stats.recall_at_k.get(5, 0) >= 0.95,
            "recall_at_10": accuracy_stats.recall_at_k.get(10, 0) >= 0.97,
            "recall_at_20": accuracy_stats.recall_at_k.get(20, 0) >= 0.99,
        }
        
        return result


class CrossSectorAccuracyBenchmark(BenchmarkRunner):
    """Benchmark cross-sector retrieval accuracy."""
    
    def __init__(self, config: BenchmarkConfig = None):
        super().__init__("cross_sector_accuracy", config or get_default_config())
        self.client: Optional[httpx.AsyncClient] = None
        self.sectors = ["episodic", "semantic", "procedural", "emotional", "workspace"]
        
    async def setup(self):
        """Create memories across sectors."""
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )
        
        # Create related memories in different sectors
        related_memories = [
            ("The team meeting is scheduled for Monday", "episodic"),
            ("Team meetings help with project coordination", "semantic"),
            ("To schedule a meeting, use the calendar app", "procedural"),
            ("I felt productive after the team meeting", "emotional"),
            ("Meeting notes are stored in the shared drive", "workspace"),
        ]
        
        for content, sector in related_memories:
            payload = {
                "content": content,
                "tags": ["cross_sector_test", sector],
            }
            await self.client.post("/v1/memories", json=payload)
    
    async def run_iteration(self, iteration: int) -> float:
        """Query across sectors."""
        query = "Tell me about team meetings"
        
        with Timer() as timer:
            response = await self.client.post("/v1/query", json={
                "query_text": query,
                "limit": 10,
                # Query across all sectors
            })
        
        if not response.is_success:
            raise Exception(f"Query failed: {response.status_code}")
        
        return timer.elapsed_ms
    
    async def teardown(self):
        """Close client."""
        if self.client:
            await self.client.aclose()


async def run_accuracy_benchmarks(config: BenchmarkConfig = None) -> Dict[str, BenchmarkResult]:
    """Run all accuracy benchmarks.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary of benchmark results
    """
    config = config or get_default_config()
    results = {}
    
    # Main accuracy benchmark
    print("Running accuracy benchmark...")
    bench = AccuracyBenchmark(config)
    results["accuracy"] = await bench.run()
    
    # Cross-sector accuracy
    print("Running cross-sector accuracy benchmark...")
    bench = CrossSectorAccuracyBenchmark(config)
    results["cross_sector"] = await bench.run(iterations=100)
    
    return results


if __name__ == "__main__":
    import asyncio
    results = asyncio.run(run_accuracy_benchmarks())
    
    print("\n=== Accuracy Benchmark Results ===")
    for name, result in results.items():
        print(f"\n{name}:")
        if result.accuracy:
            print(f"  Recall@5: {result.accuracy.recall_at_k.get(5, 0)*100:.1f}%")
            print(f"  Recall@10: {result.accuracy.recall_at_k.get(10, 0)*100:.1f}%")
            print(f"  Recall@20: {result.accuracy.recall_at_k.get(20, 0)*100:.1f}%")
            print(f"  MRR: {result.accuracy.mrr:.3f}")
        if result.meets_targets:
            print(f"  Meets targets: {result.meets_targets}")



