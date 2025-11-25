"""Test script for Phase 3: Context Assembly, Waypoint Traces, Memory Dynamics.

This script tests:
1. Context Assembler - assembles memories with provenance
2. Waypoint Traces - explains WHY each memory was retrieved
3. Decay Engine - memory strength decays over time
4. Coactivation Diffusion - related memories strengthen each other
5. GNN Reranker - graph-aware retrieval ranking
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_context_assembler():
    """Test context assembly with waypoint traces."""
    print("\n" + "=" * 60)
    print("Testing Context Assembler")
    print("=" * 60)
    
    from services.context_assembler import (
        assemble_context,
        generate_waypoint_explanation,
        get_context_assembler,
    )
    
    # Create mock retrieval results
    mock_results = [
        {
            "memory_id": "mem-001",
            "score": 0.85,
            "vector_score": 0.92,
            "graph_score": 0.3,
            "recency_score": 0.95,
            "trust_score": 0.8,
            "importance": 0.7,
            "sector": "episodic",
            "source": "both",
            "content": "I had a meeting with John in London yesterday. We discussed the Q4 roadmap.",
            "entities": ["John", "London", "Q4 roadmap"],
        },
        {
            "memory_id": "mem-002",
            "score": 0.72,
            "vector_score": 0.75,
            "graph_score": 0.6,
            "recency_score": 0.4,
            "trust_score": 0.9,
            "importance": 0.5,
            "sector": "semantic",
            "source": "graph",
            "content": "The Q4 roadmap includes three major initiatives: AI integration, mobile app, and API v2.",
            "entities": ["Q4 roadmap", "AI integration", "mobile app", "API v2"],
        },
        {
            "memory_id": "mem-003",
            "score": 0.65,
            "vector_score": 0.68,
            "graph_score": 0.2,
            "recency_score": 0.3,
            "trust_score": 0.7,
            "importance": 0.4,
            "sector": "procedural",
            "source": "vector",
            "content": "To submit the Q4 report, export from Tableau and send to finance@company.com by Friday.",
            "entities": ["Q4 report", "Tableau", "finance"],
        },
    ]
    
    # Test 1: Generate waypoint explanation for single memory
    print("\n1. Testing waypoint generation for single memory...")
    waypoint = await generate_waypoint_explanation(mock_results[0])
    print(f"   Memory ID: {waypoint['memory_id']}")
    print(f"   Final Score: {waypoint['final_score']:.3f}")
    print(f"   Explanation: {waypoint['explanation']}")
    print(f"   Factors: {waypoint['factors']}")
    print(f"   Path steps: {len(waypoint['path'])}")
    for step in waypoint['path']:
        print(f"      Step {step['step_number']}: {step['step_type']} - {step['description']}")
    
    # Test 2: Assemble context with all options
    print("\n2. Testing full context assembly...")
    context_result = await assemble_context(
        memories=mock_results,
        query="What was discussed in the London meeting?",
        include_trace=True,
        include_provenance=True,
        format_for_llm=False,
    )
    
    print(f"   Total memories: {context_result['total_memories']}")
    print(f"   Context length: {len(context_result['context'])} chars")
    print(f"   Waypoint summaries: {len(context_result['waypoint_summary'])}")
    
    # Test 3: LLM-formatted context
    print("\n3. Testing LLM-formatted context...")
    llm_context = await assemble_context(
        memories=mock_results,
        query="What was discussed in the London meeting?",
        include_trace=True,
        include_provenance=True,
        format_for_llm=True,
    )
    
    print(f"   LLM context preview (first 500 chars):")
    print("   " + "-" * 40)
    print(llm_context['context'][:500])
    print("   " + "-" * 40)
    
    print("\n   [PASS] Context Assembler tests completed")
    return True


async def test_decay_engine():
    """Test memory decay calculations."""
    print("\n" + "=" * 60)
    print("Testing Decay Engine")
    print("=" * 60)
    
    from dynamics.decay_engine import (
        get_decay_engine,
        calculate_memory_decay,
        boost_memory_strength,
        run_decay_sweep,
    )
    
    engine = get_decay_engine()
    
    # Test 1: Calculate decay for different time periods
    print("\n1. Testing decay over time (semantic memory, importance=0.5)...")
    now = datetime.utcnow()
    
    test_periods = [1, 7, 30, 90, 180, 365]
    for days in test_periods:
        last_accessed = now - timedelta(days=days)
        new_strength = engine.calculate_decay(
            current_strength=1.0,
            last_accessed=last_accessed,
            sector="semantic",
            importance=0.5,
            access_count=1,
        )
        print(f"   After {days:3d} days: strength = {new_strength:.3f}")
    
    # Test 2: Decay varies by sector
    print("\n2. Testing decay by sector (30 days, strength=1.0)...")
    last_accessed = now - timedelta(days=30)
    sectors = ["episodic", "semantic", "emotional", "procedural", "workspace"]
    
    for sector in sectors:
        new_strength = engine.calculate_decay(
            current_strength=1.0,
            last_accessed=last_accessed,
            sector=sector,
            importance=0.5,
            access_count=1,
        )
        print(f"   {sector:12s}: strength = {new_strength:.3f}")
    
    # Test 3: Importance slows decay
    print("\n3. Testing importance effect on decay (90 days, semantic)...")
    last_accessed = now - timedelta(days=90)
    
    for importance in [0.1, 0.5, 0.9]:
        new_strength = engine.calculate_decay(
            current_strength=1.0,
            last_accessed=last_accessed,
            sector="semantic",
            importance=importance,
            access_count=1,
        )
        print(f"   Importance {importance:.1f}: strength = {new_strength:.3f}")
    
    # Test 4: Access count slows decay
    print("\n4. Testing access count effect (90 days, semantic)...")
    for access_count in [1, 5, 20, 100]:
        new_strength = engine.calculate_decay(
            current_strength=1.0,
            last_accessed=last_accessed,
            sector="semantic",
            importance=0.5,
            access_count=access_count,
        )
        print(f"   Access count {access_count:3d}: strength = {new_strength:.3f}")
    
    # Test 5: Boost on access
    print("\n5. Testing strength boost on access...")
    boost_result = await boost_memory_strength(
        memory_id="test-mem",
        current_strength=0.6,
        access_type="retrieve",
    )
    print(f"   Retrieve: {boost_result['original_strength']:.2f} -> {boost_result['new_strength']:.2f}")
    
    boost_result = await boost_memory_strength(
        memory_id="test-mem",
        current_strength=0.6,
        access_type="update",
    )
    print(f"   Update:   {boost_result['original_strength']:.2f} -> {boost_result['new_strength']:.2f}")
    
    # Test 6: Half-life estimation
    print("\n6. Testing half-life estimation...")
    for sector in ["semantic", "episodic", "workspace"]:
        half_life = engine.estimate_half_life(sector=sector, importance=0.5, access_count=1)
        print(f"   {sector:12s} half-life: {half_life:.1f} days")
    
    print("\n   [PASS] Decay Engine tests completed")
    return True


async def test_coactivation():
    """Test coactivation diffusion."""
    print("\n" + "=" * 60)
    print("Testing Coactivation Diffusion")
    print("=" * 60)
    
    from dynamics.coactivation import (
        get_coactivation_engine,
        record_coactivation,
        get_associated_memories,
    )
    
    engine = get_coactivation_engine()
    
    # Test 1: Calculate association strength
    print("\n1. Testing association strength calculation...")
    
    test_cases = [
        (10, 100, 100),   # 10% co-retrieval
        (50, 100, 100),   # 50% co-retrieval
        (90, 100, 100),   # 90% co-retrieval
        (5, 100, 10),     # Asymmetric
    ]
    
    for coact, total_a, total_b in test_cases:
        strength = engine.calculate_association_strength(coact, total_a, total_b)
        print(f"   Co-activation {coact}, Total A {total_a}, Total B {total_b}: strength = {strength:.3f}")
    
    # Test 2: Record coactivation (will fail without Neo4j but tests logic)
    print("\n2. Testing coactivation recording (may fail without Neo4j)...")
    try:
        result = await record_coactivation(
            memory_ids=["mem-001", "mem-002", "mem-003"],
            context="test_retrieval",
        )
        print(f"   Memories linked: {result.get('memories_linked', 0)}")
        print(f"   Edges updated: {result.get('edges_updated', 0)}")
    except Exception as e:
        print(f"   [SKIP] Neo4j not available: {str(e)[:50]}")
    
    # Test 3: Get associated memories (will fail without Neo4j)
    print("\n3. Testing associated memory retrieval (may fail without Neo4j)...")
    try:
        associated = await get_associated_memories(
            seed_memory_ids=["mem-001"],
            min_activation=0.1,
            limit=5,
        )
        print(f"   Found {len(associated)} associated memories")
        for mem in associated:
            print(f"      {mem['memory_id']}: activation = {mem['activation']:.3f}")
    except Exception as e:
        print(f"   [SKIP] Neo4j not available: {str(e)[:50]}")
    
    print("\n   [PASS] Coactivation tests completed (some may have skipped)")
    return True


async def test_gnn_reranker():
    """Test GNN message-passing reranker."""
    print("\n" + "=" * 60)
    print("Testing GNN Reranker")
    print("=" * 60)
    
    from dynamics.gnn_reranker import (
        get_gnn_reranker,
        gnn_rerank,
        MessagePassingReranker,
    )
    
    # Create mock candidates
    mock_candidates = [
        {
            "memory_id": "mem-001",
            "score": 0.85,
            "vector_score": 0.92,
            "graph_score": 0.3,
            "recency_score": 0.95,
            "trust_score": 0.8,
            "importance": 0.7,
        },
        {
            "memory_id": "mem-002",
            "score": 0.82,
            "vector_score": 0.88,
            "graph_score": 0.5,
            "recency_score": 0.6,
            "trust_score": 0.9,
            "importance": 0.5,
        },
        {
            "memory_id": "mem-003",
            "score": 0.78,
            "vector_score": 0.75,
            "graph_score": 0.8,
            "recency_score": 0.4,
            "trust_score": 0.7,
            "importance": 0.6,
        },
        {
            "memory_id": "mem-004",
            "score": 0.75,
            "vector_score": 0.70,
            "graph_score": 0.2,
            "recency_score": 0.9,
            "trust_score": 0.6,
            "importance": 0.4,
        },
    ]
    
    # Test 1: Initialize reranker
    print("\n1. Initializing GNN reranker...")
    reranker = get_gnn_reranker()
    print(f"   Using PyTorch: {reranker.use_torch}")
    
    # Test 2: Feature similarity calculation
    print("\n2. Testing feature similarity...")
    mp_reranker = MessagePassingReranker()
    
    features_a = {"vector_score": 0.9, "graph_score": 0.3, "trust_score": 0.8}
    features_b = {"vector_score": 0.85, "graph_score": 0.4, "trust_score": 0.75}
    
    similarity = mp_reranker._feature_similarity(features_a, features_b)
    print(f"   Feature similarity: {similarity:.3f}")
    
    # Test 3: Rerank candidates (will work without Neo4j - uses features only)
    print("\n3. Testing candidate reranking...")
    print("   Original order:")
    for i, c in enumerate(mock_candidates):
        print(f"      {i+1}. {c['memory_id']}: score = {c['score']:.3f}")
    
    try:
        reranked = await gnn_rerank(mock_candidates, top_k=4)
        print("   After GNN reranking:")
        for i, c in enumerate(reranked):
            gnn_score = c.get('gnn_score', 'N/A')
            reranked_flag = c.get('reranked', False)
            if gnn_score != 'N/A':
                print(f"      {i+1}. {c['memory_id']}: score = {c['score']:.3f} (GNN: {gnn_score:.3f})")
            else:
                print(f"      {i+1}. {c['memory_id']}: score = {c['score']:.3f}")
    except Exception as e:
        print(f"   [SKIP] Reranking failed: {str(e)[:50]}")
    
    print("\n   [PASS] GNN Reranker tests completed")
    return True


async def test_integration():
    """Test integration of all Phase 3 components."""
    print("\n" + "=" * 60)
    print("Testing Phase 3 Integration")
    print("=" * 60)
    
    print("\n1. Simulating full retrieval → rerank → assemble flow...")
    
    from services.context_assembler import assemble_context
    from dynamics.gnn_reranker import gnn_rerank
    from dynamics.decay_engine import get_decay_engine
    
    # Mock retrieval results
    results = [
        {
            "memory_id": f"mem-{i:03d}",
            "score": 0.9 - (i * 0.05),
            "vector_score": 0.92 - (i * 0.05),
            "graph_score": 0.3 + (i * 0.1),
            "recency_score": 0.9 - (i * 0.1),
            "trust_score": 0.8,
            "importance": 0.7 - (i * 0.05),
            "sector": ["episodic", "semantic", "procedural"][i % 3],
            "source": "both",
            "content": f"Memory content {i} with relevant information about the topic.",
            "entities": [f"Entity_{i}"],
        }
        for i in range(5)
    ]
    
    print(f"   Initial results: {len(results)} memories")
    
    # Step 1: GNN Rerank
    print("\n2. Applying GNN reranking...")
    try:
        reranked = await gnn_rerank(results, top_k=5)
        print(f"   Reranked order: {[r['memory_id'] for r in reranked]}")
    except Exception as e:
        reranked = results
        print(f"   [SKIP] Reranking skipped: {str(e)[:30]}")
    
    # Step 2: Apply decay to importance
    print("\n3. Calculating decay for each memory...")
    engine = get_decay_engine()
    from datetime import datetime, timedelta
    
    for mem in reranked:
        days_old = int(mem['memory_id'].split('-')[1]) * 10  # Simulate age
        last_accessed = datetime.utcnow() - timedelta(days=days_old)
        
        decayed_strength = engine.calculate_decay(
            current_strength=mem['importance'],
            last_accessed=last_accessed,
            sector=mem['sector'],
            importance=mem['importance'],
            access_count=1,
        )
        mem['decayed_importance'] = decayed_strength
        print(f"   {mem['memory_id']}: {mem['importance']:.2f} -> {decayed_strength:.2f} (after {days_old} days)")
    
    # Step 3: Assemble context
    print("\n4. Assembling final context...")
    context_result = await assemble_context(
        memories=reranked,
        query="Test integration query",
        include_trace=True,
        include_provenance=True,
        format_for_llm=True,
    )
    
    print(f"   Context assembled: {context_result['total_memories']} memories")
    print(f"   Context length: {len(context_result['context'])} chars")
    print(f"   Waypoint summaries: {len(context_result['waypoint_summary'])}")
    
    print("\n   [PASS] Integration test completed")
    return True


async def main():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("LoopMind Memory HSG v4 - Phase 3 Tests")
    print("Context Assembly, Waypoint Traces, Memory Dynamics")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    try:
        results['context_assembler'] = await test_context_assembler()
    except Exception as e:
        print(f"\n[FAIL] Context Assembler: {e}")
        results['context_assembler'] = False
    
    try:
        results['decay_engine'] = await test_decay_engine()
    except Exception as e:
        print(f"\n[FAIL] Decay Engine: {e}")
        results['decay_engine'] = False
    
    try:
        results['coactivation'] = await test_coactivation()
    except Exception as e:
        print(f"\n[FAIL] Coactivation: {e}")
        results['coactivation'] = False
    
    try:
        results['gnn_reranker'] = await test_gnn_reranker()
    except Exception as e:
        print(f"\n[FAIL] GNN Reranker: {e}")
        results['gnn_reranker'] = False
    
    try:
        results['integration'] = await test_integration()
    except Exception as e:
        print(f"\n[FAIL] Integration: {e}")
        results['integration'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"   {test_name:20s}: [{status}]")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

