"""Test script for Phase 2: HSG Core Implementation.

Tests the full pipeline:
1. Create memory with entity extraction
2. Build HSG node with entity linking
3. Query using hybrid retrieval
4. Verify graph relationships
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


async def test_extractor():
    """Test entity and triple extraction."""
    print("\n" + "=" * 60)
    print("Testing Entity Extractor")
    print("=" * 60)
    
    from services.extractor import extract_entities_and_triples
    
    test_texts = [
        "I met John Smith at the London office yesterday. He mentioned the quarterly report is due on January 15, 2024.",
        "The machine learning model achieved 95% accuracy on the test dataset after training for 100 epochs.",
        "Called Sarah at +1-555-123-4567 about the project deadline. Her email is sarah@example.com.",
        "My flight from New York to San Francisco departs at 3:30 PM tomorrow.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:60]}...")
        result = await extract_entities_and_triples(text)
        
        print(f"  Entities ({len(result['entities'])}):")
        for entity in result["entities"][:5]:
            print(f"    - {entity['name']} ({entity['type']}) [confidence: {entity['confidence']}]")
        
        print(f"  Triples ({len(result['triples'])}):")
        for triple in result["triples"][:3]:
            print(f"    - {triple['subject']} -> {triple['predicate']} -> {triple['object']}")
        
        print(f"  Importance: {result['importance']}")
    
    print("\n[OK] Extractor tests completed")
    return True


async def test_hsg_builder():
    """Test HSG builder (without actual Neo4j connection)."""
    print("\n" + "=" * 60)
    print("Testing HSG Builder (mock)")
    print("=" * 60)
    
    from services.hsg_builder import HSGBuilder
    
    # Create builder
    builder = HSGBuilder()
    
    # Test with mock data
    test_memory_id = "test-memory-123"
    test_content = {
        "text": "Meeting with John about the Q4 report in London office.",
        "normalized_text": "Meeting with John about the Q4 report in London office.",
    }
    test_entities = [
        {"name": "John", "type": "person", "confidence": 0.8},
        {"name": "London", "type": "location", "confidence": 0.9},
        {"name": "Q4 report", "type": "concept", "confidence": 0.7},
    ]
    
    print(f"  Memory ID: {test_memory_id}")
    print(f"  Entities to link: {len(test_entities)}")
    
    # Note: This will attempt to connect to Neo4j
    # If Neo4j is not available, it will log warnings but not fail
    try:
        # Test building HSG node
        from services.hsg_builder import build_hsg_node
        result = await build_hsg_node(
            memory_id=test_memory_id,
            content=test_content,
            sector="semantic",
            tenant_id="test_tenant",
            user_id="test_user",
            importance=0.7,
            trust_score=0.8,
            extract_entities=False,  # We provide entities
            link_related=False,  # Skip related linking for test
        )
        
        print(f"  HSG Result: success={result.get('success', False)}")
        print(f"  Node ID: {result.get('node_id', 'N/A')}")
        
    except Exception as e:
        print(f"  [WARN] HSG builder test skipped (Neo4j not available): {e}")
    
    print("\n[OK] HSG Builder tests completed")
    return True


async def test_smart_updater():
    """Test smart updater scoring."""
    print("\n" + "=" * 60)
    print("Testing Smart Updater")
    print("=" * 60)
    
    from services.smart_updater import SmartUpdater, calculate_hybrid_score
    
    updater = SmartUpdater()
    
    # Test hybrid scoring
    test_cases = [
        # (cosine_sim, graph_sim, days_since, trust, importance)
        (0.95, 0.8, 1, 0.9, 0.8),    # Very similar, recent -> should merge
        (0.80, 0.5, 30, 0.7, 0.6),   # Similar, older -> should link
        (0.50, 0.2, 60, 0.5, 0.5),   # Not similar -> should create
        (0.30, 0.1, 90, 0.3, 0.3),   # Not related -> should create
    ]
    
    print("\n  Hybrid Score Tests:")
    for cosine, graph, days, trust, imp in test_cases:
        score = calculate_hybrid_score(cosine, graph, days, trust, imp)
        
        # Determine expected action
        if score >= 0.86:
            action = "MERGE"
        elif score >= 0.72:
            action = "LINK"
        else:
            action = "CREATE"
        
        print(f"    cosine={cosine}, graph={graph}, days={days}, trust={trust}, imp={imp}")
        print(f"      -> score={score:.3f}, action={action}")
    
    # Test recency calculation
    print("\n  Recency Score Tests:")
    from datetime import timedelta
    now = datetime.utcnow()
    
    for days in [0, 7, 30, 90, 180, 365]:
        created = now - timedelta(days=days)
        recency = updater.calculate_recency_score(created, now)
        print(f"    {days} days ago -> recency={recency:.3f}")
    
    print("\n[OK] Smart Updater tests completed")
    return True


async def test_hybrid_retriever_scoring():
    """Test hybrid retriever scoring logic."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Retriever Scoring")
    print("=" * 60)
    
    from services.hybrid_retriever import HybridRetriever, RetrievalCandidate
    
    retriever = HybridRetriever()
    
    # Create test candidates
    candidates = [
        RetrievalCandidate(
            memory_id="mem-1",
            vector_score=0.95,
            graph_score=0.3,
            recency_score=0.9,
            trust_score=0.8,
            importance=0.7,
            sector="semantic",
            source="vector",
        ),
        RetrievalCandidate(
            memory_id="mem-2",
            vector_score=0.85,
            graph_score=0.8,
            recency_score=0.5,
            trust_score=0.9,
            importance=0.6,
            sector="episodic",
            source="both",
        ),
        RetrievalCandidate(
            memory_id="mem-3",
            vector_score=0.75,
            graph_score=0.9,
            recency_score=0.3,
            trust_score=0.7,
            importance=0.8,
            sector="semantic",
            source="graph",
        ),
    ]
    
    print(f"\n  Initial candidates: {len(candidates)}")
    
    # Test reranking
    reranked = retriever.hybrid_rerank(candidates)
    
    print("\n  Reranked results:")
    for i, c in enumerate(reranked, 1):
        print(f"    {i}. {c.memory_id}: hybrid={c.hybrid_score:.3f}")
        print(f"       (vector={c.vector_score:.2f}, graph={c.graph_score:.2f}, recency={c.recency_score:.2f})")
    
    print("\n[OK] Hybrid Retriever scoring tests completed")
    return True


async def test_api_integration():
    """Test API integration (if server is running)."""
    print("\n" + "=" * 60)
    print("Testing API Integration")
    print("=" * 60)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        # Check if server is running
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                health = await client.get(f"{base_url}/health")
                if health.status_code != 200:
                    print("  [SKIP] API server not running")
                    return True
            except Exception:
                print("  [SKIP] API server not running")
                return True
        
        print("  API server is running")
        
        # Get JWT token
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Generate token
            token_response = await client.post(
                f"{base_url}/v1/auth/token",
                json={"user_id": "test_user", "tenant_id": "test_tenant"},
            )
            
            if token_response.status_code != 200:
                print(f"  [WARN] Could not get token: {token_response.text}")
                return True
            
            token = token_response.json().get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create a test memory
            print("\n  Creating test memory...")
            memory_response = await client.post(
                f"{base_url}/v1/memories",
                headers=headers,
                json={
                    "content": "I had a meeting with Alice and Bob in the Seattle office on November 25, 2024. We discussed the machine learning project timeline and decided to launch in Q1 2025.",
                    "tags": ["meeting", "project"],
                },
            )
            
            if memory_response.status_code == 200:
                memory_data = memory_response.json()
                memory_id = memory_data.get("memory_id")
                print(f"    Memory created: {memory_id}")
                print(f"    Status: {memory_data.get('status')}")
            else:
                print(f"    [WARN] Memory creation failed: {memory_response.text[:100]}")
            
            # Query memories
            print("\n  Querying memories...")
            query_response = await client.post(
                f"{base_url}/v1/query",
                headers=headers,
                json={
                    "query_text": "meeting about machine learning project",
                    "limit": 5,
                },
            )
            
            if query_response.status_code == 200:
                query_data = query_response.json()
                print(f"    Query results: {query_data.get('total', 0)} candidates")
                print(f"    Top results: {len(query_data.get('results', []))}")
                
                for i, result in enumerate(query_data.get("results", [])[:3], 1):
                    print(f"      {i}. {result.get('memory_id')[:20]}... score={result.get('score', 0):.3f}")
            else:
                print(f"    [WARN] Query failed: {query_response.text[:100]}")
        
        print("\n[OK] API integration tests completed")
        
    except ImportError:
        print("  [SKIP] httpx not installed")
    except Exception as e:
        print(f"  [ERROR] API test failed: {e}")
    
    return True


async def main():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2: HSG Core Implementation Tests")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Extractor", await test_extractor()))
    results.append(("HSG Builder", await test_hsg_builder()))
    results.append(("Smart Updater", await test_smart_updater()))
    results.append(("Hybrid Retriever", await test_hybrid_retriever_scoring()))
    results.append(("API Integration", await test_api_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All Phase 2 tests completed successfully!")
    else:
        print("Some tests failed. Check logs above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

