#!/usr/bin/env python
"""Quick launcher for LoopMind Sandbox.

Usage:
    python run_sandbox.py              # Interactive mode
    python run_sandbox.py test         # Run all tests
    python run_sandbox.py quick        # Quick smoke test
    python run_sandbox.py security     # Security tests only
"""

import asyncio
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


async def interactive_mode():
    """Run interactive sandbox."""
    from sandbox.interactive import run_sandbox
    await run_sandbox()


async def run_tests():
    """Run all automated tests."""
    from sandbox.test_runner import SandboxTestRunner
    runner = SandboxTestRunner()
    result = await runner.run_all()
    return result["total_failed"] == 0


async def quick_test():
    """Run quick smoke test."""
    print("\n" + "=" * 50)
    print("Quick Smoke Test")
    print("=" * 50)
    
    from services.preprocessor import preprocess_text
    from services.antiviral import scan_content
    from services.sector_classifier import classify_sector
    
    test_text = "Yesterday I learned about machine learning with Python"
    
    print("\n[1/3] Preprocessing...")
    result = await preprocess_text(test_text)
    assert result["normalized_text"], "Preprocessing failed"
    print("   OK - Text normalized")
    
    print("\n[2/3] ANTIVIRAL scan...")
    scan = await scan_content(test_text)
    assert scan["is_safe"], "Content should be safe"
    print("   OK - Content is safe")
    
    print("\n[3/3] Sector classification...")
    sector = await classify_sector(test_text)
    assert sector["sector"] in ["episodic", "semantic"], "Unexpected sector"
    print(f"   OK - Classified as {sector['sector']}")
    
    print("\n" + "=" * 50)
    print("Quick Test PASSED")
    print("=" * 50)
    return True


async def security_test():
    """Run security-focused tests."""
    print("\n" + "=" * 50)
    print("Security Tests")
    print("=" * 50)
    
    from services.antiviral import scan_content
    from sandbox.mock_data import generate_adversarial_tests
    
    tests = generate_adversarial_tests()
    passed = 0
    failed = 0
    
    for test in tests:
        result = await scan_content(test["input"])
        
        if test["expected_action"] == "quarantine":
            success = not result["is_safe"]
        elif test["expected_action"] == "low_trust":
            success = result["threat_score"] > 0.2
        else:  # pass
            success = result["is_safe"]
        
        status = "PASS" if success else "FAIL"
        print(f"   [{status}] {test['type']}: threat={result['threat_score']:.2f}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 50)
    print(f"   Results: {passed}/{passed + failed} passed")
    print("=" * 50)
    
    return failed == 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        mode = "interactive"
    else:
        mode = sys.argv[1].lower()
    
    if mode == "interactive" or mode == "i":
        asyncio.run(interactive_mode())
    
    elif mode == "test" or mode == "tests":
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    
    elif mode == "quick" or mode == "q":
        success = asyncio.run(quick_test())
        sys.exit(0 if success else 1)
    
    elif mode == "security" or mode == "sec":
        success = asyncio.run(security_test())
        sys.exit(0 if success else 1)
    
    elif mode == "help" or mode == "-h" or mode == "--help":
        print(__doc__)
    
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

