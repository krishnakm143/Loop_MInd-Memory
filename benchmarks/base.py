"""Base Benchmark Runner and Result Classes.

Provides the foundation for all benchmark tests with:
- Timing utilities
- Statistics calculation
- Result aggregation
- Progress tracking
"""

import asyncio
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()


class BenchmarkStatus(str, Enum):
    """Status of a benchmark run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LatencyStats:
    """Latency statistics."""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    sample_count: int
    
    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        """Calculate statistics from latency samples (in ms)."""
        if not samples:
            return cls(
                min_ms=0, max_ms=0, mean_ms=0, median_ms=0,
                p50_ms=0, p75_ms=0, p90_ms=0, p95_ms=0, p99_ms=0,
                std_dev_ms=0, sample_count=0,
            )
        
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_samples[min(idx, n - 1)]
        
        return cls(
            min_ms=min(samples),
            max_ms=max(samples),
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            p50_ms=percentile(50),
            p75_ms=percentile(75),
            p90_ms=percentile(90),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            std_dev_ms=statistics.stdev(samples) if len(samples) > 1 else 0,
            sample_count=n,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p75_ms": round(self.p75_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "sample_count": self.sample_count,
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    duration_seconds: float
    ops_per_second: float
    success_rate: float
    
    @classmethod
    def calculate(
        cls,
        total: int,
        successful: int,
        duration_seconds: float,
    ) -> "ThroughputStats":
        """Calculate throughput statistics."""
        return cls(
            total_operations=total,
            successful_operations=successful,
            failed_operations=total - successful,
            duration_seconds=duration_seconds,
            ops_per_second=successful / duration_seconds if duration_seconds > 0 else 0,
            success_rate=successful / total if total > 0 else 0,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "duration_seconds": round(self.duration_seconds, 2),
            "ops_per_second": round(self.ops_per_second, 2),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class AccuracyStats:
    """Accuracy/recall statistics."""
    recall_at_k: Dict[int, float]  # k -> recall percentage
    precision_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "sample_count": self.sample_count,
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    benchmark_name: str
    status: BenchmarkStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    
    # Different types of stats
    latency: Optional[LatencyStats] = None
    throughput: Optional[ThroughputStats] = None
    accuracy: Optional[AccuracyStats] = None
    
    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Comparison to targets
    meets_targets: Dict[str, bool] = field(default_factory=dict)
    
    # Error info if failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "latency": self.latency.to_dict() if self.latency else None,
            "throughput": self.throughput.to_dict() if self.throughput else None,
            "accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "metrics": self.metrics,
            "meets_targets": self.meets_targets,
            "error": self.error,
        }
    
    def save(self, path: str):
        """Save result to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        
    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""
    
    def __init__(self, name: str, config: Any = None):
        """Initialize benchmark runner.
        
        Args:
            name: Name of the benchmark
            config: Benchmark configuration
        """
        self.name = name
        self.config = config
        self.status = BenchmarkStatus.PENDING
        self.latency_samples: List[float] = []
        self.errors: List[str] = []
        
    @abstractmethod
    async def setup(self):
        """Set up the benchmark (create test data, connections, etc)."""
        pass
    
    @abstractmethod
    async def run_iteration(self, iteration: int) -> float:
        """Run a single iteration of the benchmark.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Latency in milliseconds
        """
        pass
    
    @abstractmethod
    async def teardown(self):
        """Clean up after the benchmark."""
        pass
    
    async def warmup(self, iterations: int = 10):
        """Run warmup iterations."""
        logger.info(f"Running {iterations} warmup iterations", benchmark=self.name)
        for i in range(iterations):
            try:
                await self.run_iteration(i)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed", error=str(e))
    
    async def run(
        self,
        iterations: int = 1000,
        warmup_iterations: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Run the full benchmark.
        
        Args:
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
            progress_callback: Optional callback for progress updates
            
        Returns:
            BenchmarkResult with statistics
        """
        started_at = datetime.utcnow()
        self.status = BenchmarkStatus.RUNNING
        self.latency_samples = []
        self.errors = []
        
        try:
            # Setup
            logger.info(f"Setting up benchmark", benchmark=self.name)
            await self.setup()
            
            # Warmup
            await self.warmup(warmup_iterations)
            
            # Main test
            logger.info(
                f"Running benchmark",
                benchmark=self.name,
                iterations=iterations,
            )
            
            start_time = time.perf_counter()
            successful = 0
            
            for i in range(iterations):
                try:
                    latency = await self.run_iteration(i)
                    self.latency_samples.append(latency)
                    successful += 1
                except Exception as e:
                    self.errors.append(str(e))
                    
                if progress_callback and i % 100 == 0:
                    progress_callback(i, iterations)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Calculate statistics
            latency_stats = LatencyStats.from_samples(self.latency_samples)
            throughput_stats = ThroughputStats.calculate(
                total=iterations,
                successful=successful,
                duration_seconds=duration,
            )
            
            # Teardown
            await self.teardown()
            
            self.status = BenchmarkStatus.COMPLETED
            completed_at = datetime.utcnow()
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                status=self.status,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                latency=latency_stats,
                throughput=throughput_stats,
                metrics={
                    "error_count": len(self.errors),
                    "sample_errors": self.errors[:10],  # First 10 errors
                },
            )
            
            logger.info(
                f"Benchmark completed",
                benchmark=self.name,
                mean_latency_ms=latency_stats.mean_ms,
                p95_latency_ms=latency_stats.p95_ms,
                throughput_qps=throughput_stats.ops_per_second,
            )
            
            return result
            
        except Exception as e:
            self.status = BenchmarkStatus.FAILED
            logger.error(f"Benchmark failed", benchmark=self.name, error=str(e))
            
            return BenchmarkResult(
                benchmark_name=self.name,
                status=self.status,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=0,
                error=str(e),
            )


class ConcurrentBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for concurrent load testing."""
    
    def __init__(self, name: str, config: Any = None, concurrency: int = 10):
        """Initialize concurrent benchmark runner.
        
        Args:
            name: Name of the benchmark
            config: Benchmark configuration
            concurrency: Number of concurrent workers
        """
        super().__init__(name, config)
        self.concurrency = concurrency
        
    async def run_concurrent(
        self,
        duration_seconds: int = 60,
        warmup_seconds: int = 5,
    ) -> BenchmarkResult:
        """Run benchmark with concurrent load.
        
        Args:
            duration_seconds: How long to run the test
            warmup_seconds: Warmup period
            
        Returns:
            BenchmarkResult with statistics
        """
        started_at = datetime.utcnow()
        self.status = BenchmarkStatus.RUNNING
        self.latency_samples = []
        self.errors = []
        
        try:
            await self.setup()
            
            # Warmup
            logger.info(f"Warmup for {warmup_seconds}s", benchmark=self.name)
            warmup_end = time.perf_counter() + warmup_seconds
            while time.perf_counter() < warmup_end:
                await self.run_iteration(0)
            
            # Concurrent test
            logger.info(
                f"Running concurrent benchmark",
                benchmark=self.name,
                concurrency=self.concurrency,
                duration_seconds=duration_seconds,
            )
            
            stop_event = asyncio.Event()
            results_queue = asyncio.Queue()
            
            async def worker(worker_id: int):
                iteration = 0
                while not stop_event.is_set():
                    try:
                        latency = await self.run_iteration(iteration)
                        await results_queue.put(("success", latency))
                    except Exception as e:
                        await results_queue.put(("error", str(e)))
                    iteration += 1
            
            # Start workers
            workers = [
                asyncio.create_task(worker(i))
                for i in range(self.concurrency)
            ]
            
            # Run for duration
            await asyncio.sleep(duration_seconds)
            stop_event.set()
            
            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)
            
            # Collect results
            successful = 0
            while not results_queue.empty():
                status, value = await results_queue.get()
                if status == "success":
                    self.latency_samples.append(value)
                    successful += 1
                else:
                    self.errors.append(value)
            
            await self.teardown()
            
            # Calculate statistics
            latency_stats = LatencyStats.from_samples(self.latency_samples)
            throughput_stats = ThroughputStats.calculate(
                total=len(self.latency_samples) + len(self.errors),
                successful=successful,
                duration_seconds=duration_seconds,
            )
            
            self.status = BenchmarkStatus.COMPLETED
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                status=self.status,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=duration_seconds,
                latency=latency_stats,
                throughput=throughput_stats,
                metrics={
                    "concurrency": self.concurrency,
                    "error_count": len(self.errors),
                },
            )
            
            logger.info(
                f"Concurrent benchmark completed",
                benchmark=self.name,
                concurrency=self.concurrency,
                throughput_qps=throughput_stats.ops_per_second,
            )
            
            return result
            
        except Exception as e:
            self.status = BenchmarkStatus.FAILED
            return BenchmarkResult(
                benchmark_name=self.name,
                status=self.status,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=0,
                error=str(e),
            )



