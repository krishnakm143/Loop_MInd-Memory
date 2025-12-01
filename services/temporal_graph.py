"""
Temporal Knowledge Graph Service.

Provides time-aware fact storage and retrieval:
- Add facts with temporal validity (auto-closes previous facts)
- Point-in-time queries (what was true on date X?)
- Timeline reconstruction for entities
- Temporal comparison between dates
- Volatile fact detection
- Confidence decay over time

OpenMemory-competitive feature for temporal reasoning.
"""

import structlog
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.temporal import TemporalFact, FactChange

logger = structlog.get_logger(__name__)


# Decay configuration per fact type
CONFIDENCE_DECAY_RATES = {
    "has_CEO": 0.02,        # CEO changes infrequently
    "works_at": 0.01,       # Employment is stable
    "lives_in": 0.015,      # Residence changes occasionally
    "has_status": 0.05,     # Status changes frequently
    "has_price": 0.1,       # Prices change rapidly
    "default": 0.03,        # Default decay rate
}


class TemporalGraphService:
    """
    Service for managing the Temporal Knowledge Graph.
    
    Key features:
    - Automatic fact evolution (new facts close old ones)
    - Point-in-time queries
    - Timeline reconstruction
    - Confidence decay
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logger.bind(service="temporal_graph")

    async def add_fact(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        object_value: str,
        valid_from: Optional[datetime] = None,
        confidence: float = 1.0,
        source_memory_id: Optional[UUID] = None,
        source_type: str = "extracted",
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> TemporalFact:
        """
        Add a new temporal fact.
        
        Automatically closes any existing fact with the same subject-predicate
        by setting its valid_to to the new fact's valid_from.
        
        Args:
            tenant_id: Tenant identifier
            subject: Entity subject (e.g., "CompanyX")
            predicate: Relationship type (e.g., "has_CEO")
            object_value: Object value (e.g., "Alice")
            valid_from: When the fact became true (default: now)
            confidence: Initial confidence score (0.0-1.0)
            source_memory_id: Source memory this was extracted from
            source_type: How the fact was created
            extra_data: Additional data
            
        Returns:
            Created TemporalFact
        """
        valid_from = valid_from or datetime.now(timezone.utc)
        
        self.logger.info(
            "adding_temporal_fact",
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            object=object_value,
            valid_from=valid_from.isoformat(),
        )

        # Find and close existing current fact with same subject-predicate
        existing_fact = await self._get_current_fact(tenant_id, subject, predicate)
        change_count = 0

        if existing_fact and existing_fact.object != object_value:
            # Close the existing fact
            existing_fact.valid_to = valid_from
            existing_fact.updated_at = datetime.now(timezone.utc)
            change_count = existing_fact.change_count + 1

            self.logger.info(
                "closing_existing_fact",
                fact_id=str(existing_fact.id),
                old_value=existing_fact.object,
                new_value=object_value,
            )

        # Create new fact
        new_fact = TemporalFact(
            id=uuid4(),
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            object=object_value,
            valid_from=valid_from,
            valid_to=None,  # Currently active
            confidence=confidence,
            initial_confidence=confidence,
            source_memory_id=source_memory_id,
            source_type=source_type,
            extra_data=extra_data or {},
            change_count=change_count,
        )

        self.session.add(new_fact)
        
        # Flush the new fact first so it has an ID for foreign key references
        await self.session.flush()

        # Record the change if there was a previous value
        if existing_fact and existing_fact.object != object_value:
            existing_fact.superseded_by = new_fact.id
            
            change_record = FactChange(
                id=uuid4(),
                tenant_id=tenant_id,
                subject=subject,
                predicate=predicate,
                old_fact_id=existing_fact.id,
                new_fact_id=new_fact.id,
                old_value=existing_fact.object,
                new_value=object_value,
                changed_at=valid_from,
                change_reason="new_information",
            )
            self.session.add(change_record)
            await self.session.flush()

        return new_fact

    async def _get_current_fact(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
    ) -> Optional[TemporalFact]:
        """Get the currently active fact for a subject-predicate pair."""
        result = await self.session.execute(
            select(TemporalFact)
            .where(
                and_(
                    TemporalFact.tenant_id == tenant_id,
                    TemporalFact.subject == subject,
                    TemporalFact.predicate == predicate,
                    TemporalFact.valid_to.is_(None),  # Currently active
                )
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def query_point_in_time(
        self,
        tenant_id: str,
        subject: str,
        predicate: Optional[str] = None,
        at_date: Optional[datetime] = None,
    ) -> List[TemporalFact]:
        """
        Query what was true at a specific point in time.
        
        Args:
            tenant_id: Tenant identifier
            subject: Entity to query
            predicate: Optional specific predicate (None = all predicates)
            at_date: Point in time to query (default: now)
            
        Returns:
            List of facts that were valid at that time
        """
        at_date = at_date or datetime.now(timezone.utc)

        self.logger.info(
            "point_in_time_query",
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            at_date=at_date.isoformat(),
        )

        conditions = [
            TemporalFact.tenant_id == tenant_id,
            TemporalFact.subject == subject,
            TemporalFact.valid_from <= at_date,
            or_(
                TemporalFact.valid_to.is_(None),
                TemporalFact.valid_to > at_date,
            ),
        ]

        if predicate:
            conditions.append(TemporalFact.predicate == predicate)

        result = await self.session.execute(
            select(TemporalFact)
            .where(and_(*conditions))
            .order_by(TemporalFact.predicate)
        )

        facts = list(result.scalars().all())
        
        self.logger.info(
            "point_in_time_result",
            subject=subject,
            at_date=at_date.isoformat(),
            facts_found=len(facts),
        )

        return facts

    async def get_timeline(
        self,
        tenant_id: str,
        subject: str,
        predicate: Optional[str] = None,
        limit: int = 100,
    ) -> List[TemporalFact]:
        """
        Get the full timeline of an entity's facts.
        
        Args:
            tenant_id: Tenant identifier
            subject: Entity to get timeline for
            predicate: Optional filter by predicate
            limit: Maximum facts to return
            
        Returns:
            List of facts ordered by valid_from (newest first)
        """
        self.logger.info(
            "get_timeline",
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
        )

        conditions = [
            TemporalFact.tenant_id == tenant_id,
            TemporalFact.subject == subject,
        ]

        if predicate:
            conditions.append(TemporalFact.predicate == predicate)

        result = await self.session.execute(
            select(TemporalFact)
            .where(and_(*conditions))
            .order_by(desc(TemporalFact.valid_from))
            .limit(limit)
        )

        return list(result.scalars().all())

    async def compare_dates(
        self,
        tenant_id: str,
        subject: str,
        date1: datetime,
        date2: datetime,
    ) -> Dict[str, Any]:
        """
        Compare what was true about an entity at two different dates.
        
        Args:
            tenant_id: Tenant identifier
            subject: Entity to compare
            date1: First date
            date2: Second date
            
        Returns:
            Dict with 'added', 'removed', 'changed', 'unchanged' facts
        """
        self.logger.info(
            "comparing_dates",
            tenant_id=tenant_id,
            subject=subject,
            date1=date1.isoformat(),
            date2=date2.isoformat(),
        )

        # Get facts at both dates
        facts_date1 = await self.query_point_in_time(tenant_id, subject, at_date=date1)
        facts_date2 = await self.query_point_in_time(tenant_id, subject, at_date=date2)

        # Convert to dicts keyed by predicate
        facts1_by_pred = {f.predicate: f for f in facts_date1}
        facts2_by_pred = {f.predicate: f for f in facts_date2}

        added = []
        removed = []
        changed = []
        unchanged = []

        # Find added facts (in date2 but not date1)
        for pred, fact in facts2_by_pred.items():
            if pred not in facts1_by_pred:
                added.append({
                    "predicate": pred,
                    "value": fact.object,
                    "valid_from": fact.valid_from.isoformat(),
                })

        # Find removed or changed facts
        for pred, fact1 in facts1_by_pred.items():
            if pred not in facts2_by_pred:
                removed.append({
                    "predicate": pred,
                    "value": fact1.object,
                    "valid_to": fact1.valid_to.isoformat() if fact1.valid_to else None,
                })
            else:
                fact2 = facts2_by_pred[pred]
                if fact1.object != fact2.object:
                    changed.append({
                        "predicate": pred,
                        "old_value": fact1.object,
                        "new_value": fact2.object,
                        "changed_at": fact2.valid_from.isoformat(),
                    })
                else:
                    unchanged.append({
                        "predicate": pred,
                        "value": fact1.object,
                    })

        return {
            "subject": subject,
            "date1": date1.isoformat(),
            "date2": date2.isoformat(),
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "summary": {
                "total_added": len(added),
                "total_removed": len(removed),
                "total_changed": len(changed),
                "total_unchanged": len(unchanged),
            },
        }

    async def get_volatile_facts(
        self,
        tenant_id: str,
        min_changes: int = 3,
        time_window_days: int = 90,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find facts that change frequently (volatile).
        
        Args:
            tenant_id: Tenant identifier
            min_changes: Minimum number of changes to be considered volatile
            time_window_days: Look at changes within this window
            limit: Maximum results
            
        Returns:
            List of volatile subject-predicate pairs with change stats
        """
        self.logger.info(
            "finding_volatile_facts",
            tenant_id=tenant_id,
            min_changes=min_changes,
            time_window_days=time_window_days,
        )

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)

        # Count changes per subject-predicate
        result = await self.session.execute(
            select(
                FactChange.subject,
                FactChange.predicate,
                func.count(FactChange.id).label("change_count"),
                func.min(FactChange.changed_at).label("first_change"),
                func.max(FactChange.changed_at).label("last_change"),
            )
            .where(
                and_(
                    FactChange.tenant_id == tenant_id,
                    FactChange.changed_at >= cutoff_date,
                )
            )
            .group_by(FactChange.subject, FactChange.predicate)
            .having(func.count(FactChange.id) >= min_changes)
            .order_by(desc(func.count(FactChange.id)))
            .limit(limit)
        )

        volatile = []
        for row in result:
            volatile.append({
                "subject": row.subject,
                "predicate": row.predicate,
                "change_count": row.change_count,
                "first_change": row.first_change.isoformat() if row.first_change else None,
                "last_change": row.last_change.isoformat() if row.last_change else None,
                "volatility_score": row.change_count / time_window_days,
            })

        return volatile

    async def decay_confidence(
        self,
        tenant_id: str,
        batch_size: int = 1000,
    ) -> int:
        """
        Apply confidence decay to old facts.
        
        Reduces confidence based on age and predicate type.
        
        Args:
            tenant_id: Tenant to process
            batch_size: Number of facts to update per batch
            
        Returns:
            Number of facts updated
        """
        self.logger.info(
            "running_confidence_decay",
            tenant_id=tenant_id,
            batch_size=batch_size,
        )

        now = datetime.now(timezone.utc)
        total_updated = 0

        # Get facts that need decay
        result = await self.session.execute(
            select(TemporalFact)
            .where(
                and_(
                    TemporalFact.tenant_id == tenant_id,
                    TemporalFact.confidence > 0.1,  # Don't decay below threshold
                )
            )
            .limit(batch_size)
        )

        facts = list(result.scalars().all())

        for fact in facts:
            age_days = (now - fact.valid_from).days
            decay_rate = CONFIDENCE_DECAY_RATES.get(
                fact.predicate,
                CONFIDENCE_DECAY_RATES["default"]
            )

            # Exponential decay: confidence = initial * e^(-rate * age)
            import math
            new_confidence = fact.initial_confidence * math.exp(-decay_rate * age_days / 365)
            new_confidence = max(0.1, min(1.0, new_confidence))  # Clamp to [0.1, 1.0]

            if abs(fact.confidence - new_confidence) > 0.01:
                fact.confidence = new_confidence
                fact.updated_at = now
                total_updated += 1

        await self.session.flush()

        self.logger.info(
            "confidence_decay_complete",
            tenant_id=tenant_id,
            facts_updated=total_updated,
        )

        return total_updated

    async def get_facts_by_memory(
        self,
        source_memory_id: UUID,
    ) -> List[TemporalFact]:
        """Get all facts extracted from a specific memory."""
        result = await self.session.execute(
            select(TemporalFact)
            .where(TemporalFact.source_memory_id == source_memory_id)
            .order_by(TemporalFact.created_at)
        )
        return list(result.scalars().all())

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        include_historical: bool = False,
        limit: int = 50,
    ) -> List[TemporalFact]:
        """
        Search facts by subject or object content.
        
        Args:
            tenant_id: Tenant identifier
            query: Search string
            include_historical: Include expired facts
            limit: Maximum results
            
        Returns:
            Matching facts
        """
        conditions = [
            TemporalFact.tenant_id == tenant_id,
            or_(
                TemporalFact.subject.ilike(f"%{query}%"),
                TemporalFact.object.ilike(f"%{query}%"),
            ),
        ]

        if not include_historical:
            conditions.append(TemporalFact.valid_to.is_(None))

        result = await self.session.execute(
            select(TemporalFact)
            .where(and_(*conditions))
            .order_by(desc(TemporalFact.confidence))
            .limit(limit)
        )

        return list(result.scalars().all())

    async def get_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get temporal graph statistics for a tenant."""
        # Total facts
        total_result = await self.session.execute(
            select(func.count(TemporalFact.id))
            .where(TemporalFact.tenant_id == tenant_id)
        )
        total_facts = total_result.scalar() or 0

        # Current facts
        current_result = await self.session.execute(
            select(func.count(TemporalFact.id))
            .where(
                and_(
                    TemporalFact.tenant_id == tenant_id,
                    TemporalFact.valid_to.is_(None),
                )
            )
        )
        current_facts = current_result.scalar() or 0

        # Unique subjects
        subjects_result = await self.session.execute(
            select(func.count(func.distinct(TemporalFact.subject)))
            .where(TemporalFact.tenant_id == tenant_id)
        )
        unique_subjects = subjects_result.scalar() or 0

        # Unique predicates
        predicates_result = await self.session.execute(
            select(func.count(func.distinct(TemporalFact.predicate)))
            .where(TemporalFact.tenant_id == tenant_id)
        )
        unique_predicates = predicates_result.scalar() or 0

        # Average confidence
        confidence_result = await self.session.execute(
            select(func.avg(TemporalFact.confidence))
            .where(
                and_(
                    TemporalFact.tenant_id == tenant_id,
                    TemporalFact.valid_to.is_(None),
                )
            )
        )
        avg_confidence = confidence_result.scalar() or 0

        # Total changes
        changes_result = await self.session.execute(
            select(func.count(FactChange.id))
            .where(FactChange.tenant_id == tenant_id)
        )
        total_changes = changes_result.scalar() or 0

        return {
            "total_facts": total_facts,
            "current_facts": current_facts,
            "historical_facts": total_facts - current_facts,
            "unique_subjects": unique_subjects,
            "unique_predicates": unique_predicates,
            "average_confidence": round(avg_confidence, 3),
            "total_changes": total_changes,
        }


# Convenience functions for direct use

async def add_temporal_fact(
    session: AsyncSession,
    tenant_id: str,
    subject: str,
    predicate: str,
    object_value: str,
    valid_from: Optional[datetime] = None,
    **kwargs,
) -> TemporalFact:
    """Add a temporal fact (convenience function)."""
    service = TemporalGraphService(session)
    return await service.add_fact(
        tenant_id=tenant_id,
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        valid_from=valid_from,
        **kwargs,
    )


async def query_temporal_facts(
    session: AsyncSession,
    tenant_id: str,
    subject: str,
    at_date: Optional[datetime] = None,
) -> List[TemporalFact]:
    """Query temporal facts at a point in time (convenience function)."""
    service = TemporalGraphService(session)
    return await service.query_point_in_time(
        tenant_id=tenant_id,
        subject=subject,
        at_date=at_date,
    )



