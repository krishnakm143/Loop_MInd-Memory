"""LoopMind Benchmarking Framework.

Comprehensive benchmarks to measure performance against competitors:
- Zep, Supermemory, Mem0, OpenAI Memory, LangChain Memory, Vector DBs

Target Metrics:
- Response Time: <150ms (beat Vector DBs' 160ms)
- Throughput: >300 QPS (beat Vector DBs' 250)
- Recall @5: >95% (beat Supermemory's 93%)
- Scalability: <15ms/item (beat Vector DBs' 18ms)
"""

from benchmarks.config import BenchmarkConfig, CompetitorBaseline
from benchmarks.base import BenchmarkRunner, BenchmarkResult

__all__ = [
    "BenchmarkConfig",
    "CompetitorBaseline",
    "BenchmarkRunner",
    "BenchmarkResult",
]



