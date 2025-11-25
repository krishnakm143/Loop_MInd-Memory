"""Quick verification that sandbox is working."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 50)
print("Verifying Sandbox Setup")
print("=" * 50)

# Test 1: Import sandbox module
print("\n1. Importing sandbox module...")
try:
    from sandbox import InteractiveSandbox, SandboxTestRunner, MockDataGenerator
    print("   [OK] Sandbox module imported")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 2: Generate mock data
print("\n2. Generating mock data...")
try:
    generator = MockDataGenerator(seed=42)
    memories = generator.generate_memories(3)
    queries = generator.generate_queries(2)
    adversarial = generator.generate_adversarial_inputs()
    print(f"   [OK] Generated {len(memories)} memories, {len(queries)} queries, {len(adversarial)} adversarial inputs")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 3: Create sandbox instance
print("\n3. Creating sandbox instance...")
try:
    sandbox = InteractiveSandbox()
    print(f"   [OK] Sandbox created for tenant={sandbox.tenant_id}")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 4: Create test runner
print("\n4. Creating test runner...")
try:
    runner = SandboxTestRunner()
    print("   [OK] Test runner created")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Sandbox Setup VERIFIED")
print("=" * 50)

print("""
Usage:
  python run_sandbox.py              # Interactive mode
  python run_sandbox.py test         # Run all tests  
  python run_sandbox.py quick        # Quick smoke test
  python run_sandbox.py security     # Security tests
""")

