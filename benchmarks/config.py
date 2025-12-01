"""Benchmark Configuration and Competitor Baselines.

Contains configuration for benchmarks and competitor performance data
extracted from market research.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class MetricType(str, Enum):
    """Types of benchmark metrics."""
    LATENCY = "latency"          # Response time in ms
    THROUGHPUT = "throughput"    # Queries per second
    ACCURACY = "accuracy"        # Recall percentage
    SCALABILITY = "scalability"  # ms per item
    MEMORY = "memory"            # Memory usage


@dataclass
class CompetitorBaseline:
    """Baseline performance data from competitors."""
    name: str
    avg_response_time_ms: float
    throughput_qps: float
    recall_at_5: float
    scalability_ms_per_item: float
    consistency: str  # Low, Medium, High
    memory_model: str
    explainable: bool
    
    @classmethod
    def from_research(cls) -> Dict[str, "CompetitorBaseline"]:
        """Load competitor baselines from market research."""
        return {
            "openmemory": cls(
                name="OpenMemory",
                avg_response_time_ms=115,  # Nov 2025 benchmark
                throughput_qps=338,
                recall_at_5=1.00,  # 100% claimed
                scalability_ms_per_item=7.9,
                consistency="High",
                memory_model="HSG + Decay + Coactivation",
                explainable=True,  # Waypoint traces
            ),
            "zep": cls(
                name="Zep",
                avg_response_time_ms=310,
                throughput_qps=180,
                recall_at_5=0.91,
                scalability_ms_per_item=32,
                consistency="Medium",
                memory_model="Recency",
                explainable=False,
            ),
            "supermemory": cls(
                name="Supermemory",
                avg_response_time_ms=270,  # Average of 200-340
                throughput_qps=220,
                recall_at_5=0.93,
                scalability_ms_per_item=25,
                consistency="Medium",
                memory_model="Frequency",
                explainable=False,  # Labels only
            ),
            "mem0": cls(
                name="Mem0",
                avg_response_time_ms=250,
                throughput_qps=150,
                recall_at_5=0.89,  # Average of 88-90%
                scalability_ms_per_item=28,
                consistency="Low",  # Inconsistent
                memory_model="Static",
                explainable=False,
            ),
            "openai_memory": cls(
                name="OpenAI Memory",
                avg_response_time_ms=300,
                throughput_qps=180,
                recall_at_5=0.90,
                scalability_ms_per_item=40,
                consistency="Low",  # Inconsistent
                memory_model="Proprietary",
                explainable=False,
            ),
            "langchain": cls(
                name="LangChain Memory",
                avg_response_time_ms=200,
                throughput_qps=140,
                recall_at_5=0.875,  # Average of 85-90%
                scalability_ms_per_item=20,
                consistency="Low",  # Volatile
                memory_model="Distance",
                explainable=False,
            ),
            "vector_dbs": cls(
                name="Vector DBs (Avg)",
                avg_response_time_ms=160,
                throughput_qps=250,
                recall_at_5=0.90,
                scalability_ms_per_item=18,
                consistency="Medium",  # Backend dependent
                memory_model="Distance",
                explainable=False,
            ),
        }


@dataclass
class BenchmarkTargets:
    """Target metrics to beat competitors (Updated for OpenMemory parity).
    
    OpenMemory benchmarks (Nov 2025):
    - Avg Response Time: 115ms @100k
    - Throughput: 338 QPS
    - Recall@5: 100% (avg 6.7ms)
    - Scalability: 7.9 ms/item @10k+
    """
    # Latency targets - Beat OpenMemory's 115ms
    avg_response_time_ms: float = 100.0  # Target <100ms
    p95_response_time_ms: float = 200.0
    p99_response_time_ms: float = 350.0
    
    # Throughput targets - Beat OpenMemory's 338
    throughput_qps: float = 350.0  # Target >350 QPS
    
    # Accuracy targets - Match OpenMemory's 100%
    recall_at_5: float = 0.98  # Target >98%
    recall_at_10: float = 0.99
    recall_at_20: float = 0.995
    
    # Scalability targets - Beat OpenMemory's 7.9ms
    scalability_ms_per_item: float = 8.0  # Target <8ms
    
    # Consistency
    consistency_target: str = "High"  # Better than all competitors


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Test sizes
    small_dataset: int = 1000
    medium_dataset: int = 10000
    large_dataset: int = 100000
    xlarge_dataset: int = 1000000
    
    # Concurrency settings
    concurrent_users: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    
    # Duration settings
    warmup_seconds: int = 5
    test_duration_seconds: int = 60
    cooldown_seconds: int = 5
    
    # Accuracy test settings
    accuracy_sample_size: int = 1000
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # API settings
    api_base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_detailed_logs: bool = True
    
    # Targets
    targets: BenchmarkTargets = field(default_factory=BenchmarkTargets)
    
    # Competitor baselines
    competitors: Dict[str, CompetitorBaseline] = field(
        default_factory=CompetitorBaseline.from_research
    )


# Test data generators
SAMPLE_MEMORIES = [
    "I met John at the coffee shop today. He mentioned he's working on a new AI project.",
    "The quarterly report shows 15% revenue growth compared to last year.",
    "Remember to call Mom on her birthday next Tuesday at 3pm.",
    "Python 3.12 introduces several performance improvements and new typing features.",
    "The team decided to use PostgreSQL for the main database and Redis for caching.",
    "Customer feedback indicates users want better mobile app performance.",
    "The meeting with investors is scheduled for Friday at 2pm in the main conference room.",
    "Machine learning models require large amounts of training data for best results.",
    "Sarah recommended the Italian restaurant on Main Street for the team dinner.",
    "The new feature should reduce page load times by approximately 40%.",
]

SAMPLE_QUERIES = [
    "What meetings do I have scheduled?",
    "Who did I meet recently?",
    "What are the key findings from the quarterly report?",
    "What database technologies are we using?",
    "What did customers say about the mobile app?",
    "When is Mom's birthday?",
    "What restaurants were recommended?",
    "What performance improvements are expected?",
    "Who is working on AI projects?",
    "What are the Python 3.12 features?",
]


def get_default_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig()

