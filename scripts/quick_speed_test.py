"""Quick speed test for synthetic embeddings."""
import sys
import time
import statistics

sys.path.insert(0, ".")

print("Starting speed test...", flush=True)

from services.fast_embeddings import FastEmbeddingGenerator

gen = FastEmbeddingGenerator(768)
text = "What did I discuss in the meeting yesterday about the project deployment?"

# Warmup
print("Warming up...", flush=True)
for _ in range(10):
    gen.generate(text, "semantic")

# Benchmark
print("Running benchmark...", flush=True)
times = []
for i in range(100):
    start = time.perf_counter()
    vec = gen.generate(text, "semantic")
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

print("\n" + "=" * 50, flush=True)
print("SYNTHETIC EMBEDDING SPEED TEST", flush=True)
print("=" * 50, flush=True)
print(f"Samples: 100", flush=True)
print(f"Avg: {statistics.mean(times):.3f}ms", flush=True)
print(f"P50: {statistics.median(times):.3f}ms", flush=True)
print(f"P95: {sorted(times)[95]:.3f}ms", flush=True)
print(f"Min/Max: {min(times):.3f}ms / {max(times):.3f}ms", flush=True)
print(f"QPS: {100/(sum(times)/1000):.0f}", flush=True)
print("=" * 50, flush=True)

# Compare to competitors
our_latency = statistics.mean(times)
competitors = {
    "OpenMemory": 115,
    "Zep": 310,
    "Supermemory": 270,
    "Mem0": 250,
}

print("\nCOMPARISON (embedding generation only):", flush=True)
print(f"LoopMind Synthetic: {our_latency:.3f}ms", flush=True)
for name, latency in competitors.items():
    ratio = latency / our_latency
    print(f"  vs {name}: {ratio:.0f}x FASTER", flush=True)

print("\nDone!", flush=True)



