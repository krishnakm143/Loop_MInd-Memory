"""Action submission endpoints for NeuralGuard."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.session import get_async_session
from api.dependencies import get_current_user, verify_agent_signature

router = APIRouter(prefix="/v1/actions", tags=["actions"])


class ActionRequest(BaseModel):
    """Request model for submitting an action."""

    action_type: str
    target: str
    parameters: Dict[str, Any] = {}
    signature: str = ""


class ActionResponse(BaseModel):
    """Response model for action submission."""

    action_id: str
    status: str
    risk_score: Optional[float] = None
    requires_approval: bool = False
    message: str = "Action submitted for NeuralGuard review"


@router.post("/submit", response_model=ActionResponse)
async def submit_action(
    request: ActionRequest,
    session: AsyncSession = Depends(get_async_session),
    user: dict = Depends(get_current_user),
):
    """Submit an action for NeuralGuard approval (stub implementation).

    This endpoint:
    1. Verifies agent signature (Ed25519)
    2. Runs NeuralGuard policy engine
    3. Executes simulator dry-run
    4. Calculates risk score
    5. Approves/denies/requires approval
    6. Logs to audit ledger

    Args:
        request: Action request with signature
        session: Database session
        user: Current user from JWT token

    Returns:
        Action response with approval status
    """
    # TODO: Implement NeuralGuard pipeline
    # 1. Verify Ed25519 signature
    # 2. Run policy engine checks
    # 3. Execute simulator dry-run
    # 4. Calculate risk score
    # 5. Determine approval status
    # 6. Log to audit ledger
    # 7. If approved, queue for Action Broker

    import uuid

    action_id = str(uuid.uuid4())

    # Verify signature if provided
    if request.signature:
        message_bytes = f"{request.action_type}:{request.target}".encode()
        is_valid = await verify_agent_signature(request.signature, message_bytes)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid agent signature",
            )

    return ActionResponse(
        action_id=action_id,
        status="pending_review",
        risk_score=0.5,
        requires_approval=True,
        message="Action submitted for NeuralGuard review",
    )

