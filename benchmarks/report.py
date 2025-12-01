"""Benchmark Report Generator.

Generates comprehensive benchmark reports with:
- Comparison against competitor baselines
- Performance summary tables
- Pass/fail against targets
- Recommendations
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from benchmarks.base import BenchmarkResult, LatencyStats, ThroughputStats, AccuracyStats
from benchmarks.config import CompetitorBaseline, BenchmarkTargets, get_default_config


@dataclass
class ComparisonResult:
    """Result of comparing against a competitor."""
    competitor_name: str
    metric: str
    our_value: float
    competitor_value: float
    unit: str
    better: bool  # True if we're better
    difference_percent: float


class BenchmarkReporter:
    """Generate benchmark reports with competitor comparison."""
    
    def __init__(self, results: Dict[str, BenchmarkResult]):
        """Initialize reporter.
        
        Args:
            results: Dictionary of benchmark results
        """
        self.results = results
        self.config = get_default_config()
        self.competitors = CompetitorBaseline.from_research()
        self.targets = BenchmarkTargets()
        self.comparisons: List[ComparisonResult] = []
        
    def _extract_metrics(self) -> Dict[str, float]:
        """Extract key metrics from results."""
        metrics = {
            "avg_response_time_ms": 0.0,
            "p95_response_time_ms": 0.0,
            "throughput_qps": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "ms_per_item": 0.0,
        }
        
        # Aggregate from various benchmarks
        latency_samples = []
        throughput_samples = []
        
        for name, result in self.results.items():
            if result.latency:
                latency_samples.append(result.latency.mean_ms)
            if result.throughput:
                throughput_samples.append(result.throughput.ops_per_second)
            if result.accuracy:
                if result.accuracy.recall_at_k.get(5):
                    metrics["recall_at_5"] = max(
                        metrics["recall_at_5"],
                        result.accuracy.recall_at_k[5],
                    )
                if result.accuracy.recall_at_k.get(10):
                    metrics["recall_at_10"] = max(
                        metrics["recall_at_10"],
                        result.accuracy.recall_at_k[10],
                    )
            if "ms_per_item" in result.metrics:
                metrics["ms_per_item"] = result.metrics["ms_per_item"]
        
        if latency_samples:
            metrics["avg_response_time_ms"] = sum(latency_samples) / len(latency_samples)
        if throughput_samples:
            metrics["throughput_qps"] = max(throughput_samples)
        
        return metrics
    
    def compare_to_competitors(self) -> List[ComparisonResult]:
        """Compare our metrics against competitors."""
        our_metrics = self._extract_metrics()
        self.comparisons = []
        
        for comp_name, competitor in self.competitors.items():
            # Response time (lower is better)
            if our_metrics["avg_response_time_ms"] > 0:
                diff = ((competitor.avg_response_time_ms - our_metrics["avg_response_time_ms"]) 
                       / competitor.avg_response_time_ms * 100)
                self.comparisons.append(ComparisonResult(
                    competitor_name=competitor.name,
                    metric="Avg Response Time",
                    our_value=our_metrics["avg_response_time_ms"],
                    competitor_value=competitor.avg_response_time_ms,
                    unit="ms",
                    better=our_metrics["avg_response_time_ms"] < competitor.avg_response_time_ms,
                    difference_percent=diff,
                ))
            
            # Throughput (higher is better)
            if our_metrics["throughput_qps"] > 0:
                diff = ((our_metrics["throughput_qps"] - competitor.throughput_qps) 
                       / competitor.throughput_qps * 100)
                self.comparisons.append(ComparisonResult(
                    competitor_name=competitor.name,
                    metric="Throughput",
                    our_value=our_metrics["throughput_qps"],
                    competitor_value=competitor.throughput_qps,
                    unit="QPS",
                    better=our_metrics["throughput_qps"] > competitor.throughput_qps,
                    difference_percent=diff,
                ))
            
            # Recall @5 (higher is better)
            if our_metrics["recall_at_5"] > 0:
                diff = ((our_metrics["recall_at_5"] - competitor.recall_at_5) 
                       / competitor.recall_at_5 * 100)
                self.comparisons.append(ComparisonResult(
                    competitor_name=competitor.name,
                    metric="Recall @5",
                    our_value=our_metrics["recall_at_5"] * 100,
                    competitor_value=competitor.recall_at_5 * 100,
                    unit="%",
                    better=our_metrics["recall_at_5"] > competitor.recall_at_5,
                    difference_percent=diff,
                ))
            
            # Scalability (lower is better)
            if our_metrics["ms_per_item"] > 0:
                diff = ((competitor.scalability_ms_per_item - our_metrics["ms_per_item"]) 
                       / competitor.scalability_ms_per_item * 100)
                self.comparisons.append(ComparisonResult(
                    competitor_name=competitor.name,
                    metric="Scalability",
                    our_value=our_metrics["ms_per_item"],
                    competitor_value=competitor.scalability_ms_per_item,
                    unit="ms/item",
                    better=our_metrics["ms_per_item"] < competitor.scalability_ms_per_item,
                    difference_percent=diff,
                ))
        
        return self.comparisons
    
    def check_targets(self) -> Dict[str, Dict[str, Any]]:
        """Check if we meet our targets."""
        our_metrics = self._extract_metrics()
        
        return {
            "avg_response_time": {
                "target": f"< {self.targets.avg_response_time_ms}ms",
                "actual": f"{our_metrics['avg_response_time_ms']:.2f}ms",
                "pass": our_metrics["avg_response_time_ms"] < self.targets.avg_response_time_ms,
            },
            "throughput": {
                "target": f"> {self.targets.throughput_qps} QPS",
                "actual": f"{our_metrics['throughput_qps']:.2f} QPS",
                "pass": our_metrics["throughput_qps"] > self.targets.throughput_qps,
            },
            "recall_at_5": {
                "target": f"> {self.targets.recall_at_5 * 100}%",
                "actual": f"{our_metrics['recall_at_5'] * 100:.1f}%",
                "pass": our_metrics["recall_at_5"] >= self.targets.recall_at_5,
            },
            "recall_at_10": {
                "target": f"> {self.targets.recall_at_10 * 100}%",
                "actual": f"{our_metrics['recall_at_10'] * 100:.1f}%",
                "pass": our_metrics["recall_at_10"] >= self.targets.recall_at_10,
            },
            "scalability": {
                "target": f"< {self.targets.scalability_ms_per_item} ms/item",
                "actual": f"{our_metrics['ms_per_item']:.4f} ms/item",
                "pass": our_metrics["ms_per_item"] < self.targets.scalability_ms_per_item,
            },
        }
    
    def generate_summary(self) -> str:
        """Generate a text summary of the benchmark results."""
        our_metrics = self._extract_metrics()
        comparisons = self.compare_to_competitors()
        targets = self.check_targets()
        
        lines = [
            "=" * 70,
            "LOOPMIND BENCHMARK REPORT",
            f"Generated: {datetime.utcnow().isoformat()}",
            "=" * 70,
            "",
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"  Avg Response Time: {our_metrics['avg_response_time_ms']:.2f} ms",
            f"  P95 Response Time: {our_metrics['p95_response_time_ms']:.2f} ms",
            f"  Throughput: {our_metrics['throughput_qps']:.2f} QPS",
            f"  Recall @5: {our_metrics['recall_at_5'] * 100:.1f}%",
            f"  Recall @10: {our_metrics['recall_at_10'] * 100:.1f}%",
            f"  Scalability: {our_metrics['ms_per_item']:.4f} ms/item",
            "",
            "TARGET COMPLIANCE",
            "-" * 40,
        ]
        
        for metric, data in targets.items():
            status = "PASS" if data["pass"] else "FAIL"
            lines.append(f"  {metric}: {data['actual']} (target: {data['target']}) [{status}]")
        
        lines.extend([
            "",
            "COMPETITOR COMPARISON",
            "-" * 40,
        ])
        
        # Group by competitor
        by_competitor: Dict[str, List[ComparisonResult]] = {}
        for comp in comparisons:
            if comp.competitor_name not in by_competitor:
                by_competitor[comp.competitor_name] = []
            by_competitor[comp.competitor_name].append(comp)
        
        for competitor_name, comps in by_competitor.items():
            wins = sum(1 for c in comps if c.better)
            total = len(comps)
            lines.append(f"\n  vs {competitor_name}: {wins}/{total} metrics better")
            for comp in comps:
                indicator = "+" if comp.better else "-"
                lines.append(
                    f"    {indicator} {comp.metric}: {comp.our_value:.2f} vs {comp.competitor_value:.2f} {comp.unit} "
                    f"({comp.difference_percent:+.1f}%)"
                )
        
        # Overall wins
        total_wins = sum(1 for c in comparisons if c.better)
        total_comparisons = len(comparisons)
        win_rate = total_wins / total_comparisons * 100 if total_comparisons > 0 else 0
        
        lines.extend([
            "",
            "OVERALL COMPETITIVE POSITION",
            "-" * 40,
            f"  Winning metrics: {total_wins}/{total_comparisons} ({win_rate:.0f}%)",
            "",
        ])
        
        # Recommendations
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        
        if our_metrics["avg_response_time_ms"] > self.targets.avg_response_time_ms:
            lines.append("  - Consider adding response caching to reduce latency")
        if our_metrics["throughput_qps"] < self.targets.throughput_qps:
            lines.append("  - Scale horizontally or optimize batch operations")
        if our_metrics["recall_at_5"] < self.targets.recall_at_5:
            lines.append("  - Tune embedding model or adjust retrieval ranking")
        if our_metrics["ms_per_item"] > self.targets.scalability_ms_per_item:
            lines.append("  - Optimize indexing pipeline for better scalability")
        
        if all(data["pass"] for data in targets.values()):
            lines.append("  - All targets met! System is performing well.")
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a JSON report."""
        our_metrics = self._extract_metrics()
        comparisons = self.compare_to_competitors()
        targets = self.check_targets()
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": our_metrics,
            "targets": targets,
            "competitor_comparisons": [
                {
                    "competitor": c.competitor_name,
                    "metric": c.metric,
                    "our_value": c.our_value,
                    "competitor_value": c.competitor_value,
                    "unit": c.unit,
                    "better": c.better,
                    "difference_percent": c.difference_percent,
                }
                for c in comparisons
            ],
            "benchmark_results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
        }
    
    def save_report(self, output_dir: str = "benchmark_results"):
        """Save reports to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Text summary
        summary_path = output_path / f"benchmark_report_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(self.generate_summary())
        print(f"Saved text report to {summary_path}")
        
        # JSON report
        json_path = output_path / f"benchmark_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.generate_json_report(), f, indent=2)
        print(f"Saved JSON report to {json_path}")
        
        return {
            "summary": str(summary_path),
            "json": str(json_path),
        }


def generate_comparison_table() -> str:
    """Generate a markdown comparison table with our targets vs competitors."""
    competitors = CompetitorBaseline.from_research()
    targets = BenchmarkTargets()
    
    lines = [
        "| Metric | Target | " + " | ".join(c.name for c in competitors.values()) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(competitors)) + " |",
    ]
    
    # Response time
    comp_values = " | ".join(f"{c.avg_response_time_ms}ms" for c in competitors.values())
    lines.append(f"| Avg Response Time | <{targets.avg_response_time_ms}ms | {comp_values} |")
    
    # Throughput
    comp_values = " | ".join(f"{c.throughput_qps}" for c in competitors.values())
    lines.append(f"| Throughput (QPS) | >{targets.throughput_qps} | {comp_values} |")
    
    # Recall
    comp_values = " | ".join(f"{c.recall_at_5*100:.0f}%" for c in competitors.values())
    lines.append(f"| Recall @5 | >{targets.recall_at_5*100:.0f}% | {comp_values} |")
    
    # Scalability
    comp_values = " | ".join(f"{c.scalability_ms_per_item}ms" for c in competitors.values())
    lines.append(f"| Scalability (ms/item) | <{targets.scalability_ms_per_item}ms | {comp_values} |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Competitor Comparison Table:")
    print(generate_comparison_table())



