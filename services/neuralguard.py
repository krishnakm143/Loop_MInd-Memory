"""NeuralGuard Policy Engine - Safe AI Action Approval System.

This module implements a comprehensive policy engine that:
1. Evaluates AI agent actions for safety
2. Calculates risk scores based on action type, target, and history
3. Simulates actions before execution (dry-run)
4. Flags high-risk actions for human approval
5. Logs all decisions to the audit ledger
"""

import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


class ActionType(str, Enum):
    """Types of actions that can be submitted to NeuralGuard."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    API_CALL = "api_call"
    DB_QUERY = "db_query"
    DB_WRITE = "db_write"
    SHELL_EXEC = "shell_exec"
    EMAIL_SEND = "email_send"
    WEBHOOK_CALL = "webhook_call"
    MEMORY_CREATE = "memory_create"
    MEMORY_DELETE = "memory_delete"
    MEMORY_MERGE = "memory_merge"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"           # Auto-approve
    MEDIUM = "medium"     # Log and approve
    HIGH = "high"         # Require review
    CRITICAL = "critical" # Block and alert


class ApprovalStatus(str, Enum):
    """Action approval status."""
    APPROVED = "approved"
    DENIED = "denied"
    PENDING_REVIEW = "pending_review"
    REQUIRES_HUMAN = "requires_human"


@dataclass
class PolicyRule:
    """A single policy rule for action evaluation."""
    name: str
    action_types: List[ActionType]
    pattern: Optional[str] = None  # Regex pattern for target matching
    max_risk_score: float = 0.5
    requires_approval: bool = False
    blocked: bool = False
    reason: str = ""
    
    def matches(self, action_type: ActionType, target: str) -> bool:
        """Check if this rule matches the given action."""
        if action_type not in self.action_types:
            return False
        if self.pattern:
            return bool(re.search(self.pattern, target, re.IGNORECASE))
        return True


@dataclass
class EvaluationResult:
    """Result of policy evaluation."""
    action_id: str
    risk_score: float
    risk_level: RiskLevel
    status: ApprovalStatus
    matched_rules: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    simulation_result: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "matched_rules": self.matched_rules,
            "reasons": self.reasons,
            "simulation_result": self.simulation_result,
            "recommendations": self.recommendations,
        }


class PolicyEngine:
    """Core policy engine for evaluating actions."""
    
    # Default risk thresholds
    LOW_THRESHOLD = 0.3
    MEDIUM_THRESHOLD = 0.6
    HIGH_THRESHOLD = 0.85
    
    def __init__(self, tenant_id: str = "default"):
        """Initialize the policy engine.
        
        Args:
            tenant_id: Tenant identifier for tenant-specific policies
        """
        self.tenant_id = tenant_id
        self.rules: List[PolicyRule] = []
        self.action_history: List[Dict[str, Any]] = []
        self._load_default_rules()
        
    def _load_default_rules(self):
        """Load default security policies."""
        self.rules = [
            # File operations
            PolicyRule(
                name="block_system_files",
                action_types=[ActionType.FILE_WRITE, ActionType.FILE_DELETE],
                pattern=r"(\/etc\/|\/sys\/|\/boot\/|C:\\Windows\\|\.env|\.ssh|\.git)",
                blocked=True,
                reason="System-critical paths are protected",
            ),
            PolicyRule(
                name="sensitive_file_write",
                action_types=[ActionType.FILE_WRITE],
                pattern=r"\.(key|pem|cert|crt|password|secret|credentials)",
                max_risk_score=0.9,
                requires_approval=True,
                reason="Writing to sensitive file types requires approval",
            ),
            PolicyRule(
                name="config_file_write",
                action_types=[ActionType.FILE_WRITE],
                pattern=r"\.(json|yaml|yml|toml|ini|conf|config)$",
                max_risk_score=0.7,
                requires_approval=True,
                reason="Configuration file changes require review",
            ),
            
            # Database operations
            PolicyRule(
                name="destructive_db_ops",
                action_types=[ActionType.DB_QUERY, ActionType.DB_WRITE],
                pattern=r"\b(DROP|TRUNCATE|DELETE\s+FROM|ALTER\s+TABLE)\b",
                max_risk_score=0.95,
                requires_approval=True,
                reason="Destructive database operations require approval",
            ),
            PolicyRule(
                name="bulk_update",
                action_types=[ActionType.DB_WRITE],
                pattern=r"\bUPDATE\b.*\bSET\b(?!.*\bWHERE\b)",
                max_risk_score=0.85,
                requires_approval=True,
                reason="UPDATE without WHERE clause affects all rows",
            ),
            
            # Shell execution
            PolicyRule(
                name="block_dangerous_shell",
                action_types=[ActionType.SHELL_EXEC],
                pattern=r"(rm\s+-rf|chmod\s+777|mkfs|dd\s+if=|>\s*\/dev\/|curl.*\|\s*(ba)?sh)",
                blocked=True,
                reason="Dangerous shell commands are blocked",
            ),
            PolicyRule(
                name="all_shell_commands",
                action_types=[ActionType.SHELL_EXEC],
                max_risk_score=0.8,
                requires_approval=True,
                reason="Shell execution requires approval by default",
            ),
            
            # API calls
            PolicyRule(
                name="external_api_calls",
                action_types=[ActionType.API_CALL],
                pattern=r"^https?://(?!localhost|127\.0\.0\.1|internal\.|api\.loopmind\.)",
                max_risk_score=0.6,
                reason="External API calls have elevated risk",
            ),
            
            # Email
            PolicyRule(
                name="bulk_email",
                action_types=[ActionType.EMAIL_SEND],
                max_risk_score=0.7,
                requires_approval=True,
                reason="Email sending requires approval",
            ),
            
            # Memory operations (lower risk - internal)
            PolicyRule(
                name="memory_delete",
                action_types=[ActionType.MEMORY_DELETE],
                max_risk_score=0.5,
                reason="Memory deletion is moderately risky",
            ),
            PolicyRule(
                name="memory_create",
                action_types=[ActionType.MEMORY_CREATE],
                max_risk_score=0.2,
                reason="Memory creation is low risk",
            ),
        ]
        
    def add_rule(self, rule: PolicyRule):
        """Add a custom policy rule."""
        self.rules.append(rule)
        logger.info("Policy rule added", rule_name=rule.name, tenant_id=self.tenant_id)
        
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a policy rule by name."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        removed = len(self.rules) < original_count
        if removed:
            logger.info("Policy rule removed", rule_name=rule_name, tenant_id=self.tenant_id)
        return removed
    
    def _calculate_base_risk(self, action_type: ActionType) -> float:
        """Calculate base risk score for action type."""
        risk_map = {
            ActionType.FILE_READ: 0.1,
            ActionType.FILE_WRITE: 0.4,
            ActionType.FILE_DELETE: 0.6,
            ActionType.API_CALL: 0.3,
            ActionType.DB_QUERY: 0.2,
            ActionType.DB_WRITE: 0.5,
            ActionType.SHELL_EXEC: 0.7,
            ActionType.EMAIL_SEND: 0.5,
            ActionType.WEBHOOK_CALL: 0.4,
            ActionType.MEMORY_CREATE: 0.1,
            ActionType.MEMORY_DELETE: 0.4,
            ActionType.MEMORY_MERGE: 0.2,
            ActionType.CUSTOM: 0.5,
        }
        return risk_map.get(action_type, 0.5)
    
    def _calculate_history_modifier(self, user_id: str, action_type: ActionType) -> float:
        """Calculate risk modifier based on user's action history."""
        # Count recent similar actions (last 100)
        recent_actions = [
            a for a in self.action_history[-100:]
            if a.get("user_id") == user_id and a.get("action_type") == action_type.value
        ]
        
        # High frequency of similar actions might indicate automation abuse
        if len(recent_actions) > 10:
            return 0.1  # Increase risk by 10%
        
        # Check for recent denials
        recent_denials = [
            a for a in recent_actions
            if a.get("status") == ApprovalStatus.DENIED.value
        ]
        if len(recent_denials) > 2:
            return 0.2  # Recent denials increase risk
            
        return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score < self.LOW_THRESHOLD:
            return RiskLevel.LOW
        elif risk_score < self.MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM
        elif risk_score < self.HIGH_THRESHOLD:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def evaluate(
        self,
        action_type: ActionType,
        target: str,
        parameters: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate an action against all policies.
        
        Args:
            action_type: Type of action being requested
            target: Target of the action (file path, URL, query, etc.)
            parameters: Action parameters
            user_id: User requesting the action
            context: Additional context for evaluation
            
        Returns:
            EvaluationResult with risk assessment and approval status
        """
        action_id = str(uuid.uuid4())
        matched_rules: List[str] = []
        reasons: List[str] = []
        recommendations: List[str] = []
        
        # Check for blocked actions first
        for rule in self.rules:
            if rule.blocked and rule.matches(action_type, target):
                logger.warning(
                    "Action blocked by policy",
                    action_id=action_id,
                    rule=rule.name,
                    action_type=action_type.value,
                    target=target,
                )
                return EvaluationResult(
                    action_id=action_id,
                    risk_score=1.0,
                    risk_level=RiskLevel.CRITICAL,
                    status=ApprovalStatus.DENIED,
                    matched_rules=[rule.name],
                    reasons=[rule.reason, "Action is blocked by security policy"],
                    recommendations=["Contact administrator if this action is required"],
                )
        
        # Calculate risk score
        base_risk = self._calculate_base_risk(action_type)
        history_modifier = self._calculate_history_modifier(user_id, action_type)
        
        # Apply rule-based risk adjustments
        max_rule_risk = base_risk
        requires_approval = False
        
        for rule in self.rules:
            if rule.matches(action_type, target):
                matched_rules.append(rule.name)
                reasons.append(rule.reason)
                max_rule_risk = max(max_rule_risk, rule.max_risk_score)
                if rule.requires_approval:
                    requires_approval = True
        
        # Final risk score
        risk_score = min(1.0, max_rule_risk + history_modifier)
        risk_level = self._determine_risk_level(risk_score)
        
        # Determine approval status
        if risk_level == RiskLevel.CRITICAL:
            status = ApprovalStatus.REQUIRES_HUMAN
            recommendations.append("This action requires manual review by an administrator")
        elif risk_level == RiskLevel.HIGH or requires_approval:
            status = ApprovalStatus.PENDING_REVIEW
            recommendations.append("Action queued for review due to elevated risk")
        elif risk_level == RiskLevel.MEDIUM:
            status = ApprovalStatus.APPROVED
            recommendations.append("Action approved with logging")
        else:
            status = ApprovalStatus.APPROVED
        
        # Record in history
        self.action_history.append({
            "action_id": action_id,
            "user_id": user_id,
            "action_type": action_type.value,
            "target": target,
            "risk_score": risk_score,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        logger.info(
            "Action evaluated",
            action_id=action_id,
            action_type=action_type.value,
            risk_score=risk_score,
            risk_level=risk_level.value,
            status=status.value,
            matched_rules=matched_rules,
        )
        
        return EvaluationResult(
            action_id=action_id,
            risk_score=risk_score,
            risk_level=risk_level,
            status=status,
            matched_rules=matched_rules,
            reasons=reasons,
            recommendations=recommendations,
        )


class ActionSimulator:
    """Simulator for dry-running actions before execution."""
    
    def __init__(self):
        """Initialize the action simulator."""
        self.simulations: Dict[str, Dict[str, Any]] = {}
        
    async def simulate(
        self,
        action_type: ActionType,
        target: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate an action without executing it.
        
        Args:
            action_type: Type of action to simulate
            target: Target of the action
            parameters: Action parameters
            
        Returns:
            Simulation result with predicted outcomes
        """
        simulation_id = str(uuid.uuid4())
        
        result = {
            "simulation_id": simulation_id,
            "action_type": action_type.value,
            "target": target,
            "success": True,
            "predicted_effects": [],
            "warnings": [],
            "estimated_duration_ms": 0,
        }
        
        # Simulate based on action type
        if action_type in [ActionType.FILE_WRITE, ActionType.FILE_DELETE]:
            result["predicted_effects"].append(f"File operation on: {target}")
            result["estimated_duration_ms"] = 50
            
            # Check if path looks dangerous
            if any(p in target.lower() for p in ["/etc", "/sys", "system32", ".env"]):
                result["warnings"].append("Target path may affect system stability")
                
        elif action_type == ActionType.DB_QUERY:
            query = parameters.get("query", "")
            
            # Estimate query complexity
            if "JOIN" in query.upper():
                result["warnings"].append("Query contains JOINs - may be slow on large tables")
                result["estimated_duration_ms"] = 500
            else:
                result["estimated_duration_ms"] = 100
                
            if "SELECT *" in query.upper():
                result["warnings"].append("SELECT * may return large result sets")
                
        elif action_type == ActionType.DB_WRITE:
            query = parameters.get("query", "")
            
            # Estimate affected rows
            if "WHERE" not in query.upper() and "UPDATE" in query.upper():
                result["warnings"].append("UPDATE without WHERE affects ALL rows")
                result["predicted_effects"].append("Potentially affects entire table")
                
            result["estimated_duration_ms"] = 200
            
        elif action_type == ActionType.API_CALL:
            result["predicted_effects"].append(f"HTTP request to: {target}")
            result["estimated_duration_ms"] = 1000
            
            if "http://" in target:
                result["warnings"].append("Using HTTP instead of HTTPS")
                
        elif action_type == ActionType.SHELL_EXEC:
            command = parameters.get("command", target)
            result["predicted_effects"].append(f"Shell execution: {command[:100]}")
            result["estimated_duration_ms"] = 5000
            
            # Check for piped commands
            if "|" in command:
                result["warnings"].append("Command contains pipes - output may be modified")
            if ">" in command or ">>" in command:
                result["warnings"].append("Command redirects output to file")
                
        elif action_type == ActionType.EMAIL_SEND:
            recipients = parameters.get("recipients", [])
            result["predicted_effects"].append(f"Email to {len(recipients)} recipient(s)")
            result["estimated_duration_ms"] = 2000
            
            if len(recipients) > 10:
                result["warnings"].append("Bulk email - verify recipient list")
                
        elif action_type == ActionType.WEBHOOK_CALL:
            result["predicted_effects"].append(f"Webhook POST to: {target}")
            result["estimated_duration_ms"] = 1500
            
        elif action_type in [ActionType.MEMORY_CREATE, ActionType.MEMORY_DELETE, ActionType.MEMORY_MERGE]:
            result["predicted_effects"].append(f"Memory operation: {action_type.value}")
            result["estimated_duration_ms"] = 100
            
        # Store simulation
        self.simulations[simulation_id] = result
        
        logger.debug(
            "Action simulated",
            simulation_id=simulation_id,
            action_type=action_type.value,
            warnings=result["warnings"],
        )
        
        return result


class NeuralGuard:
    """Main NeuralGuard service combining policy engine and simulator."""
    
    def __init__(self, tenant_id: str = "default"):
        """Initialize NeuralGuard.
        
        Args:
            tenant_id: Tenant identifier
        """
        self.tenant_id = tenant_id
        self.policy_engine = PolicyEngine(tenant_id)
        self.simulator = ActionSimulator()
        self.pending_reviews: Dict[str, Dict[str, Any]] = {}
        
    async def evaluate_action(
        self,
        action_type: str,
        target: str,
        parameters: Dict[str, Any],
        user_id: str,
        run_simulation: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an action through the full NeuralGuard pipeline.
        
        Args:
            action_type: Type of action (string or ActionType)
            target: Target of the action
            parameters: Action parameters
            user_id: User requesting the action
            run_simulation: Whether to run simulation
            context: Additional context
            
        Returns:
            Complete evaluation result
        """
        # Parse action type
        try:
            action_enum = ActionType(action_type) if isinstance(action_type, str) else action_type
        except ValueError:
            action_enum = ActionType.CUSTOM
            
        # Run policy evaluation
        eval_result = self.policy_engine.evaluate(
            action_type=action_enum,
            target=target,
            parameters=parameters,
            user_id=user_id,
            context=context,
        )
        
        # Run simulation if requested and not blocked
        if run_simulation and eval_result.status != ApprovalStatus.DENIED:
            simulation = await self.simulator.simulate(
                action_type=action_enum,
                target=target,
                parameters=parameters,
            )
            eval_result.simulation_result = simulation
            
            # Add simulation warnings to recommendations
            if simulation.get("warnings"):
                eval_result.recommendations.extend(
                    [f"Simulation warning: {w}" for w in simulation["warnings"]]
                )
        
        # Track pending reviews
        if eval_result.status in [ApprovalStatus.PENDING_REVIEW, ApprovalStatus.REQUIRES_HUMAN]:
            self.pending_reviews[eval_result.action_id] = {
                "evaluation": eval_result.to_dict(),
                "action_type": action_type,
                "target": target,
                "parameters": parameters,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
            }
            
        return eval_result.to_dict()
    
    async def approve_action(self, action_id: str, approver_id: str, reason: str = "") -> Dict[str, Any]:
        """Manually approve a pending action.
        
        Args:
            action_id: ID of the action to approve
            approver_id: ID of the approver
            reason: Reason for approval
            
        Returns:
            Updated action status
        """
        if action_id not in self.pending_reviews:
            return {
                "success": False,
                "error": "Action not found in pending reviews",
            }
            
        pending = self.pending_reviews.pop(action_id)
        
        logger.info(
            "Action manually approved",
            action_id=action_id,
            approver_id=approver_id,
            reason=reason,
        )
        
        return {
            "success": True,
            "action_id": action_id,
            "status": ApprovalStatus.APPROVED.value,
            "approved_by": approver_id,
            "approved_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "original_action": pending,
        }
    
    async def deny_action(self, action_id: str, denier_id: str, reason: str) -> Dict[str, Any]:
        """Manually deny a pending action.
        
        Args:
            action_id: ID of the action to deny
            denier_id: ID of the denier
            reason: Reason for denial
            
        Returns:
            Updated action status
        """
        if action_id not in self.pending_reviews:
            return {
                "success": False,
                "error": "Action not found in pending reviews",
            }
            
        pending = self.pending_reviews.pop(action_id)
        
        logger.info(
            "Action manually denied",
            action_id=action_id,
            denier_id=denier_id,
            reason=reason,
        )
        
        return {
            "success": True,
            "action_id": action_id,
            "status": ApprovalStatus.DENIED.value,
            "denied_by": denier_id,
            "denied_at": datetime.utcnow().isoformat(),
            "reason": reason,
        }
    
    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all pending review actions."""
        return [
            {"action_id": action_id, **data}
            for action_id, data in self.pending_reviews.items()
        ]
    
    def add_policy(self, rule: PolicyRule):
        """Add a custom policy rule."""
        self.policy_engine.add_rule(rule)
        
    def remove_policy(self, rule_name: str) -> bool:
        """Remove a policy rule."""
        return self.policy_engine.remove_rule(rule_name)
    
    def get_policies(self) -> List[Dict[str, Any]]:
        """Get all active policies."""
        return [
            {
                "name": r.name,
                "action_types": [a.value for a in r.action_types],
                "pattern": r.pattern,
                "max_risk_score": r.max_risk_score,
                "requires_approval": r.requires_approval,
                "blocked": r.blocked,
                "reason": r.reason,
            }
            for r in self.policy_engine.rules
        ]


# Global NeuralGuard instance cache
_neuralguard_instances: Dict[str, NeuralGuard] = {}


def get_neuralguard(tenant_id: str = "default") -> NeuralGuard:
    """Get or create a NeuralGuard instance for a tenant.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        NeuralGuard instance
    """
    if tenant_id not in _neuralguard_instances:
        _neuralguard_instances[tenant_id] = NeuralGuard(tenant_id)
    return _neuralguard_instances[tenant_id]


# Convenience function for direct evaluation
async def evaluate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate action through NeuralGuard.

    Args:
        action: Action to evaluate with keys:
            - action_type: Type of action
            - target: Target of action
            - parameters: Action parameters
            - user_id: User ID
            - tenant_id: Tenant ID (optional)

    Returns:
        Evaluation result with risk score and approval status
    """
    tenant_id = action.get("tenant_id", "default")
    neuralguard = get_neuralguard(tenant_id)
    
    return await neuralguard.evaluate_action(
        action_type=action.get("action_type", "custom"),
        target=action.get("target", ""),
        parameters=action.get("parameters", {}),
        user_id=action.get("user_id", "unknown"),
        run_simulation=action.get("run_simulation", True),
        context=action.get("context"),
    )
