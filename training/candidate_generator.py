"""Candidate Generator for Adapter Training.

Collects high-quality training examples from memories for future adapter training.
This prepares data for when GPU training resources are available.

Features:
- Quality Filter: Only trust_score >= threshold
- Diversity Sampler: Balance across memory sectors
- User Corrections: Prioritize human-edited content
- Export Format: JSON/JSONL for Colab/cloud training
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog

logger = structlog.get_logger()


class CandidateQuality(str, Enum):
    """Quality classification for training candidates."""
    EXCELLENT = "excellent"  # trust >= 0.9, human-verified
    GOOD = "good"           # trust >= 0.7, multiple sources
    ACCEPTABLE = "acceptable"  # trust >= 0.5
    REJECTED = "rejected"   # Below threshold or flagged


class SampleStrategy(str, Enum):
    """Sampling strategies for candidate selection."""
    UNIFORM = "uniform"           # Equal from each sector
    PROPORTIONAL = "proportional" # Based on sector distribution
    WEIGHTED = "weighted"         # Based on quality scores
    RECENT = "recent"             # Prioritize recent memories
    CORRECTIONS = "corrections"   # Prioritize user corrections


@dataclass
class TrainingCandidate:
    """A candidate example for training."""
    candidate_id: str
    memory_id: str
    content: str
    sector: str
    quality: CandidateQuality
    trust_score: float
    importance: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Training-specific fields
    input_text: str = ""
    output_text: str = ""
    instruction: str = ""
    is_correction: bool = False
    correction_source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "candidate_id": self.candidate_id,
            "memory_id": self.memory_id,
            "content": self.content,
            "sector": self.sector,
            "quality": self.quality.value,
            "trust_score": self.trust_score,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "instruction": self.instruction,
            "is_correction": self.is_correction,
            "correction_source": self.correction_source,
        }
    
    def to_training_format(self, format_type: str = "alpaca") -> Dict[str, Any]:
        """Convert to specific training format.
        
        Formats:
        - alpaca: instruction, input, output
        - sharegpt: conversations list
        - simple: text only
        """
        if format_type == "alpaca":
            return {
                "instruction": self.instruction or "Remember and recall this information.",
                "input": self.input_text or self.content,
                "output": self.output_text or self.content,
            }
        elif format_type == "sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": self.input_text or f"Remember: {self.content}"},
                    {"from": "gpt", "value": self.output_text or f"I've stored this memory: {self.content[:100]}..."},
                ]
            }
        else:  # simple
            return {"text": self.content}


@dataclass
class CandidateStats:
    """Statistics about generated candidates."""
    total_candidates: int = 0
    by_sector: Dict[str, int] = field(default_factory=dict)
    by_quality: Dict[str, int] = field(default_factory=dict)
    corrections_count: int = 0
    avg_trust_score: float = 0.0
    avg_importance: float = 0.0
    date_range: Tuple[Optional[datetime], Optional[datetime]] = (None, None)


class QualityFilter:
    """Filter candidates based on quality criteria."""
    
    def __init__(
        self,
        min_trust_score: float = 0.7,
        min_importance: float = 0.3,
        min_content_length: int = 10,
        max_content_length: int = 10000,
        exclude_flagged: bool = True,
        require_provenance: bool = True,
    ):
        """Initialize quality filter.
        
        Args:
            min_trust_score: Minimum trust score (0-1)
            min_importance: Minimum importance score (0-1)
            min_content_length: Minimum content length in characters
            max_content_length: Maximum content length
            exclude_flagged: Exclude flagged/quarantined memories
            require_provenance: Require provenance information
        """
        self.min_trust_score = min_trust_score
        self.min_importance = min_importance
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.exclude_flagged = exclude_flagged
        self.require_provenance = require_provenance
        
    def evaluate(self, memory: Dict[str, Any]) -> Tuple[bool, CandidateQuality, List[str]]:
        """Evaluate a memory for training candidacy.
        
        Args:
            memory: Memory data dictionary
            
        Returns:
            Tuple of (passes_filter, quality_level, rejection_reasons)
        """
        reasons = []
        
        # Extract values
        trust_score = memory.get("trust_score", 0.0)
        importance = memory.get("importance", 0.0)
        content = memory.get("content", "")
        status = memory.get("status", "")
        provenance = memory.get("provenance", [])
        
        # Check content length
        content_text = content if isinstance(content, str) else json.dumps(content)
        if len(content_text) < self.min_content_length:
            reasons.append(f"Content too short ({len(content_text)} chars)")
        if len(content_text) > self.max_content_length:
            reasons.append(f"Content too long ({len(content_text)} chars)")
            
        # Check trust score
        if trust_score < self.min_trust_score:
            reasons.append(f"Trust score too low ({trust_score:.2f} < {self.min_trust_score})")
            
        # Check importance
        if importance < self.min_importance:
            reasons.append(f"Importance too low ({importance:.2f} < {self.min_importance})")
            
        # Check status
        if self.exclude_flagged and status in ["flagged", "quarantined", "deleted"]:
            reasons.append(f"Memory status is {status}")
            
        # Check provenance
        if self.require_provenance and not provenance:
            reasons.append("No provenance information")
            
        # Determine quality level
        if reasons:
            return False, CandidateQuality.REJECTED, reasons
            
        # Calculate quality based on scores
        if trust_score >= 0.9 and importance >= 0.7:
            quality = CandidateQuality.EXCELLENT
        elif trust_score >= 0.7:
            quality = CandidateQuality.GOOD
        else:
            quality = CandidateQuality.ACCEPTABLE
            
        return True, quality, []


class DiversitySampler:
    """Sample candidates to ensure diversity across sectors."""
    
    def __init__(
        self,
        strategy: SampleStrategy = SampleStrategy.UNIFORM,
        target_per_sector: int = 100,
        max_total: int = 1000,
    ):
        """Initialize diversity sampler.
        
        Args:
            strategy: Sampling strategy to use
            target_per_sector: Target samples per sector (for uniform)
            max_total: Maximum total samples
        """
        self.strategy = strategy
        self.target_per_sector = target_per_sector
        self.max_total = max_total
        
    def sample(
        self,
        candidates: List[TrainingCandidate],
        sector_distribution: Optional[Dict[str, float]] = None,
    ) -> List[TrainingCandidate]:
        """Sample candidates according to strategy.
        
        Args:
            candidates: List of all candidates
            sector_distribution: Target distribution by sector (for proportional)
            
        Returns:
            Sampled candidates
        """
        if not candidates:
            return []
            
        if self.strategy == SampleStrategy.UNIFORM:
            return self._sample_uniform(candidates)
        elif self.strategy == SampleStrategy.PROPORTIONAL:
            return self._sample_proportional(candidates, sector_distribution or {})
        elif self.strategy == SampleStrategy.WEIGHTED:
            return self._sample_weighted(candidates)
        elif self.strategy == SampleStrategy.RECENT:
            return self._sample_recent(candidates)
        elif self.strategy == SampleStrategy.CORRECTIONS:
            return self._sample_corrections(candidates)
        else:
            return candidates[:self.max_total]
            
    def _sample_uniform(self, candidates: List[TrainingCandidate]) -> List[TrainingCandidate]:
        """Sample uniformly from each sector."""
        by_sector: Dict[str, List[TrainingCandidate]] = {}
        for c in candidates:
            if c.sector not in by_sector:
                by_sector[c.sector] = []
            by_sector[c.sector].append(c)
            
        result = []
        for sector, sector_candidates in by_sector.items():
            # Sort by quality (trust * importance)
            sorted_candidates = sorted(
                sector_candidates,
                key=lambda x: x.trust_score * x.importance,
                reverse=True,
            )
            result.extend(sorted_candidates[:self.target_per_sector])
            
        return result[:self.max_total]
        
    def _sample_proportional(
        self,
        candidates: List[TrainingCandidate],
        distribution: Dict[str, float],
    ) -> List[TrainingCandidate]:
        """Sample proportionally based on target distribution."""
        by_sector: Dict[str, List[TrainingCandidate]] = {}
        for c in candidates:
            if c.sector not in by_sector:
                by_sector[c.sector] = []
            by_sector[c.sector].append(c)
            
        result = []
        for sector, target_ratio in distribution.items():
            target_count = int(self.max_total * target_ratio)
            sector_candidates = by_sector.get(sector, [])
            sorted_candidates = sorted(
                sector_candidates,
                key=lambda x: x.trust_score * x.importance,
                reverse=True,
            )
            result.extend(sorted_candidates[:target_count])
            
        return result[:self.max_total]
        
    def _sample_weighted(self, candidates: List[TrainingCandidate]) -> List[TrainingCandidate]:
        """Sample weighted by quality scores."""
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.trust_score * x.importance * (1.5 if x.is_correction else 1.0),
            reverse=True,
        )
        return sorted_candidates[:self.max_total]
        
    def _sample_recent(self, candidates: List[TrainingCandidate]) -> List[TrainingCandidate]:
        """Sample prioritizing recent memories."""
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.created_at,
            reverse=True,
        )
        return sorted_candidates[:self.max_total]
        
    def _sample_corrections(self, candidates: List[TrainingCandidate]) -> List[TrainingCandidate]:
        """Prioritize user corrections."""
        corrections = [c for c in candidates if c.is_correction]
        non_corrections = [c for c in candidates if not c.is_correction]
        
        # Take all corrections, then fill with non-corrections
        result = corrections[:self.max_total]
        remaining = self.max_total - len(result)
        if remaining > 0:
            sorted_non = sorted(
                non_corrections,
                key=lambda x: x.trust_score * x.importance,
                reverse=True,
            )
            result.extend(sorted_non[:remaining])
            
        return result


class CandidateGenerator:
    """Main candidate generator for training data collection."""
    
    def __init__(
        self,
        quality_filter: Optional[QualityFilter] = None,
        diversity_sampler: Optional[DiversitySampler] = None,
    ):
        """Initialize candidate generator.
        
        Args:
            quality_filter: Quality filter instance
            diversity_sampler: Diversity sampler instance
        """
        self.quality_filter = quality_filter or QualityFilter()
        self.diversity_sampler = diversity_sampler or DiversitySampler()
        self.candidates: List[TrainingCandidate] = []
        self.rejected: List[Dict[str, Any]] = []
        
    def process_memory(
        self,
        memory: Dict[str, Any],
        instruction_template: Optional[str] = None,
    ) -> Optional[TrainingCandidate]:
        """Process a memory into a training candidate.
        
        Args:
            memory: Memory data dictionary
            instruction_template: Optional instruction template
            
        Returns:
            TrainingCandidate if passes filter, None otherwise
        """
        passes, quality, reasons = self.quality_filter.evaluate(memory)
        
        if not passes:
            self.rejected.append({
                "memory_id": memory.get("memory_id", "unknown"),
                "reasons": reasons,
            })
            logger.debug(
                "Memory rejected for training",
                memory_id=memory.get("memory_id"),
                reasons=reasons,
            )
            return None
            
        # Extract content
        content = memory.get("content", "")
        if isinstance(content, dict):
            content = content.get("text", json.dumps(content))
            
        # Create candidate
        candidate = TrainingCandidate(
            candidate_id=str(uuid.uuid4()),
            memory_id=str(memory.get("memory_id", "")),
            content=content,
            sector=memory.get("type", "episodic"),
            quality=quality,
            trust_score=memory.get("trust_score", 0.5),
            importance=memory.get("importance", 0.5),
            created_at=datetime.fromisoformat(memory["created_at"]) if isinstance(memory.get("created_at"), str) else memory.get("created_at", datetime.utcnow()),
            metadata={
                "tags": memory.get("tags", []),
                "user_id": memory.get("user_id", ""),
                "tenant_id": memory.get("tenant_id", ""),
            },
            input_text=content,
            output_text=content,  # For recall tasks
            instruction=instruction_template or f"Recall information about: {memory.get('tags', ['this topic'])[0] if memory.get('tags') else 'this topic'}",
            is_correction=memory.get("is_correction", False),
            correction_source=memory.get("correction_source"),
        )
        
        self.candidates.append(candidate)
        return candidate
        
    def process_batch(
        self,
        memories: List[Dict[str, Any]],
        instruction_template: Optional[str] = None,
    ) -> List[TrainingCandidate]:
        """Process a batch of memories.
        
        Args:
            memories: List of memory dictionaries
            instruction_template: Optional instruction template
            
        Returns:
            List of valid training candidates
        """
        results = []
        for memory in memories:
            candidate = self.process_memory(memory, instruction_template)
            if candidate:
                results.append(candidate)
                
        logger.info(
            "Batch processed",
            total=len(memories),
            accepted=len(results),
            rejected=len(memories) - len(results),
        )
        
        return results
        
    def sample_candidates(
        self,
        sector_distribution: Optional[Dict[str, float]] = None,
    ) -> List[TrainingCandidate]:
        """Sample from collected candidates.
        
        Args:
            sector_distribution: Optional target sector distribution
            
        Returns:
            Sampled candidates
        """
        return self.diversity_sampler.sample(self.candidates, sector_distribution)
        
    def get_stats(self) -> CandidateStats:
        """Get statistics about collected candidates."""
        if not self.candidates:
            return CandidateStats()
            
        by_sector: Dict[str, int] = {}
        by_quality: Dict[str, int] = {}
        corrections = 0
        total_trust = 0.0
        total_importance = 0.0
        min_date = None
        max_date = None
        
        for c in self.candidates:
            by_sector[c.sector] = by_sector.get(c.sector, 0) + 1
            by_quality[c.quality.value] = by_quality.get(c.quality.value, 0) + 1
            if c.is_correction:
                corrections += 1
            total_trust += c.trust_score
            total_importance += c.importance
            
            if min_date is None or c.created_at < min_date:
                min_date = c.created_at
            if max_date is None or c.created_at > max_date:
                max_date = c.created_at
                
        return CandidateStats(
            total_candidates=len(self.candidates),
            by_sector=by_sector,
            by_quality=by_quality,
            corrections_count=corrections,
            avg_trust_score=total_trust / len(self.candidates),
            avg_importance=total_importance / len(self.candidates),
            date_range=(min_date, max_date),
        )
        
    def export_jsonl(
        self,
        output_path: str,
        format_type: str = "alpaca",
        sample_first: bool = True,
    ) -> int:
        """Export candidates to JSONL file.
        
        Args:
            output_path: Path to output file
            format_type: Training format (alpaca, sharegpt, simple)
            sample_first: Whether to sample before export
            
        Returns:
            Number of exported candidates
        """
        candidates = self.sample_candidates() if sample_first else self.candidates
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for candidate in candidates:
                line = json.dumps(candidate.to_training_format(format_type))
                f.write(line + '\n')
                
        logger.info(
            "Candidates exported",
            path=output_path,
            count=len(candidates),
            format=format_type,
        )
        
        return len(candidates)
        
    def export_json(
        self,
        output_path: str,
        format_type: str = "alpaca",
        sample_first: bool = True,
        include_metadata: bool = False,
    ) -> int:
        """Export candidates to JSON file.
        
        Args:
            output_path: Path to output file
            format_type: Training format
            sample_first: Whether to sample before export
            include_metadata: Include full metadata in export
            
        Returns:
            Number of exported candidates
        """
        candidates = self.sample_candidates() if sample_first else self.candidates
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_metadata:
            data = [c.to_dict() for c in candidates]
        else:
            data = [c.to_training_format(format_type) for c in candidates]
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(
            "Candidates exported",
            path=output_path,
            count=len(candidates),
            format=format_type,
        )
        
        return len(candidates)
        
    def clear(self):
        """Clear all collected candidates."""
        self.candidates = []
        self.rejected = []


# Global candidate generator instance
_candidate_generator: Optional[CandidateGenerator] = None


def get_candidate_generator() -> CandidateGenerator:
    """Get or create the global candidate generator."""
    global _candidate_generator
    if _candidate_generator is None:
        _candidate_generator = CandidateGenerator()
    return _candidate_generator


async def collect_training_candidates(
    memories: List[Dict[str, Any]],
    min_trust_score: float = 0.7,
    export_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to collect training candidates.
    
    Args:
        memories: List of memory dictionaries
        min_trust_score: Minimum trust score threshold
        export_path: Optional path to export JSONL
        
    Returns:
        Collection statistics
    """
    generator = CandidateGenerator(
        quality_filter=QualityFilter(min_trust_score=min_trust_score),
    )
    
    generator.process_batch(memories)
    stats = generator.get_stats()
    
    if export_path:
        generator.export_jsonl(export_path)
        
    return {
        "total_processed": len(memories),
        "total_accepted": stats.total_candidates,
        "by_sector": stats.by_sector,
        "by_quality": stats.by_quality,
        "corrections": stats.corrections_count,
        "avg_trust": stats.avg_trust_score,
        "avg_importance": stats.avg_importance,
    }
