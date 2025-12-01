"""Quick test to check API imports."""
import sys
print("Python:", sys.version, flush=True)

try:
    print("Testing imports...", flush=True)
    from api.main import app
    print("SUCCESS: api.main imported", flush=True)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

