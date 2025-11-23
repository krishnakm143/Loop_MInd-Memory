"""HSG graph builder service (stub)."""

from typing import Dict, Any, List


async def build_hsg_node(
    memory_id: str, content: Dict[str, Any], sector: str
) -> Dict[str, Any]:
    """Build HSG node (stub implementation).

    Args:
        memory_id: Memory UUID
        content: Memory content
        sector: Sector classification

    Returns:
        HSG node structure
    """
    # TODO: Implement:
    # - Build canonical memory node protobuf
    # - Add versioning and provenance
    # - Create edges (coactivation, mentions, temporal)
    return {
        "node_id": memory_id,
        "sector": sector,
        "version": 1,
        "provenance": [],
    }

