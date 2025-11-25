"""Webhooks API Endpoints.

Event notification system for memory events:
- POST /v1/webhooks - Register a webhook
- GET /v1/webhooks - List webhooks
- DELETE /v1/webhooks/{id} - Remove webhook
- POST /v1/webhooks/{id}/test - Test webhook

Events: memory.created, memory.updated, memory.deleted, memory.merged, action.approved
"""

import uuid
import hmac
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
import httpx
import structlog

from api.dependencies import get_current_user

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


# In-memory webhook storage (use database in production)
_webhooks: Dict[str, Dict[str, Any]] = {}
_webhook_deliveries: Dict[str, List[Dict[str, Any]]] = {}


class WebhookEventType(str):
    """Webhook event types."""
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_MERGED = "memory.merged"
    MEMORY_QUARANTINED = "memory.quarantined"
    QUERY_EXECUTED = "query.executed"
    ACTION_SUBMITTED = "action.submitted"
    ACTION_APPROVED = "action.approved"
    ACTION_DENIED = "action.denied"


ALL_EVENTS = [
    "memory.created",
    "memory.updated",
    "memory.deleted",
    "memory.merged",
    "memory.quarantined",
    "query.executed",
    "action.submitted",
    "action.approved",
    "action.denied",
]


class WebhookCreate(BaseModel):
    """Request to create a webhook."""
    url: HttpUrl
    events: List[str] = Field(default=["memory.created"])
    secret: Optional[str] = None  # For HMAC signature verification
    active: bool = True
    description: str = ""
    headers: Dict[str, str] = {}  # Custom headers to include


class WebhookResponse(BaseModel):
    """Webhook response model."""
    webhook_id: str
    url: str
    events: List[str]
    active: bool
    description: str
    created_at: datetime
    last_triggered: Optional[datetime] = None
    delivery_count: int = 0
    failure_count: int = 0


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    delivery_id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: datetime
    success: bool
    error: Optional[str] = None


class WebhookTestResponse(BaseModel):
    """Response from webhook test."""
    webhook_id: str
    success: bool
    status_code: Optional[int] = None
    response_time_ms: int
    error: Optional[str] = None


async def deliver_webhook(
    webhook_id: str,
    event: str,
    payload: Dict[str, Any],
):
    """Deliver a webhook event."""
    if webhook_id not in _webhooks:
        return
    
    webhook = _webhooks[webhook_id]
    
    if not webhook["active"]:
        return
    
    if event not in webhook["events"]:
        return
    
    delivery_id = str(uuid.uuid4())
    
    # Prepare payload
    delivery_payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "webhook_id": webhook_id,
        "data": payload,
    }
    
    # Generate signature if secret is set
    headers = dict(webhook.get("headers", {}))
    headers["Content-Type"] = "application/json"
    
    if webhook.get("secret"):
        signature = hmac.new(
            webhook["secret"].encode(),
            msg=str(delivery_payload).encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"
    
    try:
        import time
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                str(webhook["url"]),
                json=delivery_payload,
                headers=headers,
            )
        
        duration_ms = int((time.time() - start_time) * 1000)
        success = response.is_success
        
        # Update webhook stats
        _webhooks[webhook_id]["delivery_count"] = _webhooks[webhook_id].get("delivery_count", 0) + 1
        _webhooks[webhook_id]["last_triggered"] = datetime.utcnow().isoformat()
        
        if not success:
            _webhooks[webhook_id]["failure_count"] = _webhooks[webhook_id].get("failure_count", 0) + 1
        
        # Record delivery
        if webhook_id not in _webhook_deliveries:
            _webhook_deliveries[webhook_id] = []
        
        _webhook_deliveries[webhook_id].append({
            "delivery_id": delivery_id,
            "event": event,
            "payload": payload,
            "status_code": response.status_code,
            "response_body": response.text[:500] if response.text else None,
            "delivered_at": datetime.utcnow().isoformat(),
            "success": success,
            "duration_ms": duration_ms,
        })
        
        # Keep only last 100 deliveries
        _webhook_deliveries[webhook_id] = _webhook_deliveries[webhook_id][-100:]
        
        logger.info(
            "Webhook delivered",
            webhook_id=webhook_id,
            event=event,
            status_code=response.status_code,
            success=success,
            duration_ms=duration_ms,
        )
        
    except Exception as e:
        logger.error(
            "Webhook delivery failed",
            webhook_id=webhook_id,
            event=event,
            error=str(e),
        )
        
        _webhooks[webhook_id]["failure_count"] = _webhooks[webhook_id].get("failure_count", 0) + 1
        
        if webhook_id not in _webhook_deliveries:
            _webhook_deliveries[webhook_id] = []
        
        _webhook_deliveries[webhook_id].append({
            "delivery_id": delivery_id,
            "event": event,
            "payload": payload,
            "status_code": None,
            "delivered_at": datetime.utcnow().isoformat(),
            "success": False,
            "error": str(e)[:200],
        })


async def trigger_webhook_event(
    tenant_id: str,
    event: str,
    payload: Dict[str, Any],
):
    """Trigger webhook event for all matching webhooks."""
    for webhook_id, webhook in _webhooks.items():
        if webhook.get("tenant_id") == tenant_id:
            # Deliver in background
            asyncio.create_task(deliver_webhook(webhook_id, event, payload))


@router.post("", response_model=WebhookResponse)
async def create_webhook(
    request: WebhookCreate,
    user: dict = Depends(get_current_user),
) -> WebhookResponse:
    """Register a new webhook.
    
    Args:
        request: Webhook configuration
        
    Returns:
        Created webhook
    """
    webhook_id = str(uuid.uuid4())
    tenant_id = user.get("tenant_id", "default")
    
    # Validate events
    invalid_events = [e for e in request.events if e not in ALL_EVENTS]
    if invalid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid events: {invalid_events}. Valid events: {ALL_EVENTS}",
        )
    
    # Create webhook
    webhook = {
        "webhook_id": webhook_id,
        "tenant_id": tenant_id,
        "url": str(request.url),
        "events": request.events,
        "secret": request.secret,
        "active": request.active,
        "description": request.description,
        "headers": request.headers,
        "created_at": datetime.utcnow().isoformat(),
        "delivery_count": 0,
        "failure_count": 0,
    }
    
    _webhooks[webhook_id] = webhook
    
    logger.info(
        "Webhook created",
        webhook_id=webhook_id,
        tenant_id=tenant_id,
        events=request.events,
    )
    
    return WebhookResponse(
        webhook_id=webhook_id,
        url=str(request.url),
        events=request.events,
        active=request.active,
        description=request.description,
        created_at=datetime.utcnow(),
    )


@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    user: dict = Depends(get_current_user),
) -> List[WebhookResponse]:
    """List all webhooks for the tenant.
    
    Returns:
        List of webhooks
    """
    tenant_id = user.get("tenant_id", "default")
    
    webhooks = [
        WebhookResponse(
            webhook_id=w["webhook_id"],
            url=w["url"],
            events=w["events"],
            active=w["active"],
            description=w.get("description", ""),
            created_at=datetime.fromisoformat(w["created_at"]),
            last_triggered=datetime.fromisoformat(w["last_triggered"]) if w.get("last_triggered") else None,
            delivery_count=w.get("delivery_count", 0),
            failure_count=w.get("failure_count", 0),
        )
        for w in _webhooks.values()
        if w.get("tenant_id") == tenant_id
    ]
    
    return webhooks


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    user: dict = Depends(get_current_user),
) -> WebhookResponse:
    """Get a specific webhook.
    
    Args:
        webhook_id: Webhook ID
        
    Returns:
        Webhook details
    """
    tenant_id = user.get("tenant_id", "default")
    
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    w = _webhooks[webhook_id]
    
    if w.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return WebhookResponse(
        webhook_id=w["webhook_id"],
        url=w["url"],
        events=w["events"],
        active=w["active"],
        description=w.get("description", ""),
        created_at=datetime.fromisoformat(w["created_at"]),
        last_triggered=datetime.fromisoformat(w["last_triggered"]) if w.get("last_triggered") else None,
        delivery_count=w.get("delivery_count", 0),
        failure_count=w.get("failure_count", 0),
    )


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a webhook.
    
    Args:
        webhook_id: Webhook ID
        
    Returns:
        Deletion confirmation
    """
    tenant_id = user.get("tenant_id", "default")
    
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    if _webhooks[webhook_id].get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del _webhooks[webhook_id]
    
    # Clean up deliveries
    if webhook_id in _webhook_deliveries:
        del _webhook_deliveries[webhook_id]
    
    logger.info("Webhook deleted", webhook_id=webhook_id)
    
    return {"message": "Webhook deleted", "webhook_id": webhook_id}


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: str,
    user: dict = Depends(get_current_user),
) -> WebhookTestResponse:
    """Test a webhook with a sample event.
    
    Args:
        webhook_id: Webhook ID
        
    Returns:
        Test result
    """
    tenant_id = user.get("tenant_id", "default")
    
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook = _webhooks[webhook_id]
    
    if webhook.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Send test event
    test_payload = {
        "event": "test",
        "timestamp": datetime.utcnow().isoformat(),
        "webhook_id": webhook_id,
        "data": {
            "message": "This is a test webhook delivery",
            "tenant_id": tenant_id,
        },
    }
    
    headers = dict(webhook.get("headers", {}))
    headers["Content-Type"] = "application/json"
    
    if webhook.get("secret"):
        signature = hmac.new(
            webhook["secret"].encode(),
            msg=str(test_payload).encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"
    
    try:
        import time
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                str(webhook["url"]),
                json=test_payload,
                headers=headers,
            )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return WebhookTestResponse(
            webhook_id=webhook_id,
            success=response.is_success,
            status_code=response.status_code,
            response_time_ms=duration_ms,
        )
        
    except Exception as e:
        return WebhookTestResponse(
            webhook_id=webhook_id,
            success=False,
            response_time_ms=0,
            error=str(e)[:200],
        )


@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDelivery])
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 50,
    user: dict = Depends(get_current_user),
) -> List[WebhookDelivery]:
    """Get recent deliveries for a webhook.
    
    Args:
        webhook_id: Webhook ID
        limit: Maximum number of deliveries to return
        
    Returns:
        List of recent deliveries
    """
    tenant_id = user.get("tenant_id", "default")
    
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    if _webhooks[webhook_id].get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    deliveries = _webhook_deliveries.get(webhook_id, [])[-limit:]
    
    return [
        WebhookDelivery(
            delivery_id=d["delivery_id"],
            webhook_id=webhook_id,
            event=d["event"],
            payload=d["payload"],
            status_code=d.get("status_code"),
            response_body=d.get("response_body"),
            delivered_at=datetime.fromisoformat(d["delivered_at"]),
            success=d["success"],
            error=d.get("error"),
        )
        for d in reversed(deliveries)  # Most recent first
    ]


# Utility function to trigger events from other parts of the app
def emit_webhook_event(tenant_id: str, event: str, data: Dict[str, Any]):
    """Emit a webhook event (non-async wrapper)."""
    asyncio.create_task(trigger_webhook_event(tenant_id, event, data))

