"""Run All Benchmarks Script.

CLI tool to run comprehensive benchmarks and generate reports.

Usage:
    python scripts/run_benchmarks.py --all
    python scripts/run_benchmarks.py --quick
    python scripts/run_benchmarks.py --memory
    python scripts/run_benchmarks.py --query
    python scripts/run_benchmarks.py --accuracy
    python scripts/run_benchmarks.py --scale --max-size 10000
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.config import get_default_config, BenchmarkConfig
from benchmarks.memory_bench import run_memory_benchmarks
from benchmarks.query_bench import run_query_benchmarks
from benchmarks.accuracy_bench import run_accuracy_benchmarks
from benchmarks.scale_bench import run_scale_benchmarks
from benchmarks.report import BenchmarkReporter, generate_comparison_table


async def run_quick_benchmarks(config: BenchmarkConfig):
    """Run a quick subset of benchmarks."""
    results = {}
    
    print("\n=== Quick Memory Benchmark ===")
    from benchmarks.memory_bench import MemoryCreationBenchmark
    bench = MemoryCreationBenchmark(config)
    results["memory_quick"] = await bench.run(iterations=100, warmup_iterations=5)
    
    print("\n=== Quick Query Benchmark ===")
    from benchmarks.query_bench import QueryBenchmark
    bench = QueryBenchmark(config)
    results["query_quick"] = await bench.run(iterations=100, warmup_iterations=5)
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="Run LoopMind Benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory benchmarks")
    parser.add_argument("--query", action="store_true", help="Run query benchmarks")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy benchmarks")
    parser.add_argument("--scale", action="store_true", help="Run scalability benchmarks")
    parser.add_argument("--max-size", type=int, default=10000, help="Max dataset size for scale tests")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key/token")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Show competitor comparison table")
    
    args = parser.parse_args()
    
    # Show comparison table
    if args.compare:
        print("\n=== Competitor Comparison Table ===\n")
        print(generate_comparison_table())
        return
    
    # Configure benchmarks
    config = get_default_config()
    config.api_base_url = args.api_url
    config.api_key = args.api_key
    config.output_dir = args.output_dir
    
    all_results = {}
    start_time = datetime.utcnow()
    
    print("=" * 60)
    print("LOOPMIND BENCHMARK SUITE")
    print(f"Started: {start_time.isoformat()}")
    print(f"API: {config.api_base_url}")
    print("=" * 60)
    
    # Determine which benchmarks to run
    run_all = args.all or not any([args.quick, args.memory, args.query, args.accuracy, args.scale])
    
    try:
        if args.quick:
            print("\n>>> Running quick benchmarks...")
            results = await run_quick_benchmarks(config)
            all_results.update(results)
            
        if args.memory or run_all:
            print("\n>>> Running memory benchmarks...")
            results = await run_memory_benchmarks(config)
            all_results.update(results)
            
        if args.query or run_all:
            print("\n>>> Running query benchmarks...")
            results = await run_query_benchmarks(config)
            all_results.update(results)
            
        if args.accuracy or run_all:
            print("\n>>> Running accuracy benchmarks...")
            results = await run_accuracy_benchmarks(config)
            all_results.update(results)
            
        if args.scale or run_all:
            print("\n>>> Running scalability benchmarks...")
            results = await run_scale_benchmarks(config, max_size=args.max_size)
            all_results.update(results)
            
    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate report
    if all_results:
        print("\n" + "=" * 60)
        print("GENERATING REPORT")
        print("=" * 60)
        
        reporter = BenchmarkReporter(all_results)
        
        # Print summary
        print(reporter.generate_summary())
        
        # Save reports
        paths = reporter.save_report(args.output_dir)
        print(f"\nReports saved to: {args.output_dir}/")
        
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print(f"Completed in {duration:.1f} seconds")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code or 0)



