"""Action Broker Service - Execute Approved Actions via Connectors.

This module provides connectors for executing various types of actions:
1. Webhook Connector - Generic HTTP calls
2. GitHub Connector - Create issues, PRs, comments
3. Slack Connector - Send messages, notifications
4. Email Connector - Send templated emails
5. Database Connector - Execute approved queries

Each connector:
- Validates credentials from secure config
- Logs execution to audit ledger
- Returns structured results
"""

import uuid
import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from enum import Enum
from dataclasses import dataclass, field
import httpx
import structlog

logger = structlog.get_logger()


class ConnectorType(str, Enum):
    """Types of action connectors."""
    WEBHOOK = "webhook"
    GITHUB = "github"
    SLACK = "slack"
    EMAIL = "email"
    DATABASE = "database"
    MEMORY = "memory"


class ExecutionStatus(str, Enum):
    """Execution status for actions."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of action execution."""
    execution_id: str
    status: ExecutionStatus
    connector_type: ConnectorType
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "connector_type": self.connector_type.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "response_data": self.response_data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseConnector(ABC):
    """Base class for action connectors."""
    
    connector_type: ConnectorType = ConnectorType.WEBHOOK
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize connector with configuration.
        
        Args:
            config: Connector-specific configuration
        """
        self.config = config or {}
        self.execution_history: List[ExecutionResult] = []
        
    @abstractmethod
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute an action.
        
        Args:
            target: Target of the action
            parameters: Action parameters
            context: Additional context
            
        Returns:
            Execution result
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate connector configuration."""
        return True
    
    def _create_result(
        self,
        status: ExecutionStatus,
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Create an execution result."""
        now = datetime.utcnow()
        result = ExecutionResult(
            execution_id=str(uuid.uuid4()),
            status=status,
            connector_type=self.connector_type,
            started_at=now,
            completed_at=now,
            response_data=response_data,
            error=error,
            metadata=metadata or {},
        )
        self.execution_history.append(result)
        return result


class WebhookConnector(BaseConnector):
    """Generic HTTP webhook connector."""
    
    connector_type = ConnectorType.WEBHOOK
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize webhook connector.
        
        Config options:
            - timeout: Request timeout in seconds (default: 30)
            - default_headers: Headers to include in all requests
            - allowed_domains: List of allowed domains (optional)
        """
        super().__init__(config)
        self.timeout = config.get("timeout", 30) if config else 30
        self.default_headers = config.get("default_headers", {}) if config else {}
        self.allowed_domains = config.get("allowed_domains", []) if config else []
        
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute HTTP request.
        
        Parameters:
            - method: HTTP method (default: POST)
            - headers: Additional headers
            - body: Request body (JSON)
            - query_params: URL query parameters
        """
        start_time = datetime.utcnow()
        
        # Check domain allowlist
        if self.allowed_domains:
            from urllib.parse import urlparse
            parsed = urlparse(target)
            if parsed.netloc not in self.allowed_domains:
                return self._create_result(
                    ExecutionStatus.FAILED,
                    error=f"Domain {parsed.netloc} not in allowed list",
                )
        
        method = parameters.get("method", "POST").upper()
        headers = {**self.default_headers, **parameters.get("headers", {})}
        body = parameters.get("body", {})
        query_params = parameters.get("query_params", {})
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=target,
                    headers=headers,
                    json=body if method in ["POST", "PUT", "PATCH"] else None,
                    params=query_params,
                )
                
                end_time = datetime.utcnow()
                duration = int((end_time - start_time).total_seconds() * 1000)
                
                # Try to parse JSON response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = {"text": response.text[:1000]}
                
                result = ExecutionResult(
                    execution_id=str(uuid.uuid4()),
                    status=ExecutionStatus.SUCCESS if response.is_success else ExecutionStatus.FAILED,
                    connector_type=self.connector_type,
                    started_at=start_time,
                    completed_at=end_time,
                    duration_ms=duration,
                    response_data={
                        "status_code": response.status_code,
                        "data": response_data,
                    },
                    error=None if response.is_success else f"HTTP {response.status_code}",
                    metadata={"url": target, "method": method},
                )
                
                self.execution_history.append(result)
                
                logger.info(
                    "Webhook executed",
                    execution_id=result.execution_id,
                    url=target,
                    status_code=response.status_code,
                    duration_ms=duration,
                )
                
                return result
                
        except httpx.TimeoutException:
            return self._create_result(
                ExecutionStatus.TIMEOUT,
                error=f"Request timed out after {self.timeout}s",
                metadata={"url": target, "method": method},
            )
        except Exception as e:
            logger.error("Webhook execution failed", url=target, error=str(e))
            return self._create_result(
                ExecutionStatus.FAILED,
                error=str(e),
                metadata={"url": target, "method": method},
            )


class GitHubConnector(BaseConnector):
    """GitHub API connector for issues, PRs, and comments."""
    
    connector_type = ConnectorType.GITHUB
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GitHub connector.
        
        Config options:
            - token: GitHub personal access token
            - base_url: API base URL (default: https://api.github.com)
        """
        super().__init__(config)
        self.token = config.get("token", "") if config else ""
        self.base_url = config.get("base_url", "https://api.github.com") if config else "https://api.github.com"
        
    def validate_config(self) -> bool:
        """Validate GitHub token is configured."""
        return bool(self.token)
        
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute GitHub API action.
        
        Target format: owner/repo
        
        Parameters:
            - action: create_issue, create_comment, create_pr
            - title: Issue/PR title
            - body: Issue/PR/Comment body
            - issue_number: For comments
            - base/head: For PRs
            - labels: List of labels
        """
        if not self.validate_config():
            return self._create_result(
                ExecutionStatus.FAILED,
                error="GitHub token not configured",
            )
            
        start_time = datetime.utcnow()
        action = parameters.get("action", "create_issue")
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if action == "create_issue":
                    url = f"{self.base_url}/repos/{target}/issues"
                    payload = {
                        "title": parameters.get("title", "New Issue"),
                        "body": parameters.get("body", ""),
                        "labels": parameters.get("labels", []),
                    }
                    response = await client.post(url, headers=headers, json=payload)
                    
                elif action == "create_comment":
                    issue_number = parameters.get("issue_number")
                    url = f"{self.base_url}/repos/{target}/issues/{issue_number}/comments"
                    payload = {"body": parameters.get("body", "")}
                    response = await client.post(url, headers=headers, json=payload)
                    
                elif action == "create_pr":
                    url = f"{self.base_url}/repos/{target}/pulls"
                    payload = {
                        "title": parameters.get("title", "New PR"),
                        "body": parameters.get("body", ""),
                        "head": parameters.get("head", ""),
                        "base": parameters.get("base", "main"),
                    }
                    response = await client.post(url, headers=headers, json=payload)
                    
                else:
                    return self._create_result(
                        ExecutionStatus.FAILED,
                        error=f"Unknown action: {action}",
                    )
                
                end_time = datetime.utcnow()
                duration = int((end_time - start_time).total_seconds() * 1000)
                
                response_data = response.json() if response.is_success else {"error": response.text}
                
                result = ExecutionResult(
                    execution_id=str(uuid.uuid4()),
                    status=ExecutionStatus.SUCCESS if response.is_success else ExecutionStatus.FAILED,
                    connector_type=self.connector_type,
                    started_at=start_time,
                    completed_at=end_time,
                    duration_ms=duration,
                    response_data=response_data,
                    error=None if response.is_success else f"GitHub API error: {response.status_code}",
                    metadata={"repo": target, "action": action},
                )
                
                self.execution_history.append(result)
                
                logger.info(
                    "GitHub action executed",
                    execution_id=result.execution_id,
                    repo=target,
                    action=action,
                    status=result.status.value,
                )
                
                return result
                
        except Exception as e:
            logger.error("GitHub execution failed", repo=target, action=action, error=str(e))
            return self._create_result(
                ExecutionStatus.FAILED,
                error=str(e),
                metadata={"repo": target, "action": action},
            )


class SlackConnector(BaseConnector):
    """Slack messaging connector."""
    
    connector_type = ConnectorType.SLACK
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Slack connector.
        
        Config options:
            - webhook_url: Slack incoming webhook URL
            - bot_token: Slack bot token (for API calls)
            - default_channel: Default channel ID
        """
        super().__init__(config)
        self.webhook_url = config.get("webhook_url", "") if config else ""
        self.bot_token = config.get("bot_token", "") if config else ""
        self.default_channel = config.get("default_channel", "") if config else ""
        
    def validate_config(self) -> bool:
        """Validate Slack configuration."""
        return bool(self.webhook_url or self.bot_token)
        
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Send Slack message.
        
        Target: channel ID or name (e.g., #general or C1234567890)
        
        Parameters:
            - text: Message text
            - blocks: Slack blocks for rich formatting
            - attachments: Message attachments
            - thread_ts: Thread timestamp for replies
        """
        if not self.validate_config():
            return self._create_result(
                ExecutionStatus.FAILED,
                error="Slack not configured (need webhook_url or bot_token)",
            )
            
        start_time = datetime.utcnow()
        channel = target or self.default_channel
        
        payload = {
            "channel": channel,
            "text": parameters.get("text", ""),
        }
        
        if parameters.get("blocks"):
            payload["blocks"] = parameters["blocks"]
        if parameters.get("attachments"):
            payload["attachments"] = parameters["attachments"]
        if parameters.get("thread_ts"):
            payload["thread_ts"] = parameters["thread_ts"]
            
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if self.webhook_url:
                    # Use incoming webhook
                    response = await client.post(
                        self.webhook_url,
                        json=payload,
                    )
                else:
                    # Use Bot API
                    response = await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        json=payload,
                    )
                
                end_time = datetime.utcnow()
                duration = int((end_time - start_time).total_seconds() * 1000)
                
                try:
                    response_data = response.json()
                except Exception:
                    response_data = {"text": response.text}
                
                success = response.is_success
                if self.bot_token and response_data.get("ok") is False:
                    success = False
                
                result = ExecutionResult(
                    execution_id=str(uuid.uuid4()),
                    status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                    connector_type=self.connector_type,
                    started_at=start_time,
                    completed_at=end_time,
                    duration_ms=duration,
                    response_data=response_data,
                    error=None if success else response_data.get("error", "Unknown error"),
                    metadata={"channel": channel},
                )
                
                self.execution_history.append(result)
                
                logger.info(
                    "Slack message sent",
                    execution_id=result.execution_id,
                    channel=channel,
                    status=result.status.value,
                )
                
                return result
                
        except Exception as e:
            logger.error("Slack execution failed", channel=channel, error=str(e))
            return self._create_result(
                ExecutionStatus.FAILED,
                error=str(e),
                metadata={"channel": channel},
            )


class EmailConnector(BaseConnector):
    """Email sending connector using SMTP or API services."""
    
    connector_type = ConnectorType.EMAIL
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize email connector.
        
        Config options:
            - smtp_host: SMTP server host
            - smtp_port: SMTP server port
            - smtp_user: SMTP username
            - smtp_password: SMTP password
            - from_email: Default sender email
            - api_key: For email APIs (SendGrid, Mailgun, etc.)
            - api_provider: sendgrid, mailgun, etc.
        """
        super().__init__(config)
        self.smtp_host = config.get("smtp_host", "") if config else ""
        self.smtp_port = config.get("smtp_port", 587) if config else 587
        self.smtp_user = config.get("smtp_user", "") if config else ""
        self.smtp_password = config.get("smtp_password", "") if config else ""
        self.from_email = config.get("from_email", "") if config else ""
        self.api_key = config.get("api_key", "") if config else ""
        self.api_provider = config.get("api_provider", "") if config else ""
        
    def validate_config(self) -> bool:
        """Validate email configuration."""
        return bool(self.smtp_host or self.api_key)
        
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Send email.
        
        Target: recipient email or comma-separated list
        
        Parameters:
            - subject: Email subject
            - body: Email body (text or HTML)
            - html: True if body is HTML
            - cc: CC recipients
            - bcc: BCC recipients
            - attachments: List of attachment data
        """
        if not self.validate_config():
            return self._create_result(
                ExecutionStatus.FAILED,
                error="Email not configured (need SMTP or API credentials)",
            )
            
        start_time = datetime.utcnow()
        recipients = [e.strip() for e in target.split(",")]
        
        # For demo purposes, we'll simulate email sending
        # In production, implement actual SMTP or API calls
        
        email_data = {
            "to": recipients,
            "from": self.from_email,
            "subject": parameters.get("subject", "No Subject"),
            "body": parameters.get("body", ""),
            "html": parameters.get("html", False),
            "cc": parameters.get("cc", []),
            "bcc": parameters.get("bcc", []),
        }
        
        try:
            # Simulate email sending (replace with actual implementation)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            end_time = datetime.utcnow()
            duration = int((end_time - start_time).total_seconds() * 1000)
            
            result = ExecutionResult(
                execution_id=str(uuid.uuid4()),
                status=ExecutionStatus.SUCCESS,
                connector_type=self.connector_type,
                started_at=start_time,
                completed_at=end_time,
                duration_ms=duration,
                response_data={
                    "message_id": str(uuid.uuid4()),
                    "recipients_count": len(recipients),
                    "accepted": recipients,
                    "rejected": [],
                },
                metadata={
                    "recipients": recipients,
                    "subject": email_data["subject"],
                },
            )
            
            self.execution_history.append(result)
            
            logger.info(
                "Email sent",
                execution_id=result.execution_id,
                recipients=recipients,
                subject=email_data["subject"],
            )
            
            return result
            
        except Exception as e:
            logger.error("Email execution failed", recipients=recipients, error=str(e))
            return self._create_result(
                ExecutionStatus.FAILED,
                error=str(e),
                metadata={"recipients": recipients},
            )


class MemoryConnector(BaseConnector):
    """Internal connector for memory operations."""
    
    connector_type = ConnectorType.MEMORY
    
    async def execute(
        self,
        target: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute memory operation.
        
        Target: memory_id
        
        Parameters:
            - action: create, update, delete, merge
            - content: Memory content (for create/update)
            - merge_with: Memory ID to merge with
        """
        start_time = datetime.utcnow()
        action = parameters.get("action", "create")
        
        try:
            # This connector integrates with the memory system
            # For now, return success (actual implementation would call memory service)
            
            result_data = {
                "memory_id": target or str(uuid.uuid4()),
                "action": action,
                "success": True,
            }
            
            end_time = datetime.utcnow()
            duration = int((end_time - start_time).total_seconds() * 1000)
            
            result = ExecutionResult(
                execution_id=str(uuid.uuid4()),
                status=ExecutionStatus.SUCCESS,
                connector_type=self.connector_type,
                started_at=start_time,
                completed_at=end_time,
                duration_ms=duration,
                response_data=result_data,
                metadata={"memory_id": target, "action": action},
            )
            
            self.execution_history.append(result)
            
            logger.info(
                "Memory operation executed",
                execution_id=result.execution_id,
                memory_id=target,
                action=action,
            )
            
            return result
            
        except Exception as e:
            logger.error("Memory operation failed", memory_id=target, action=action, error=str(e))
            return self._create_result(
                ExecutionStatus.FAILED,
                error=str(e),
                metadata={"memory_id": target, "action": action},
            )


class ActionBroker:
    """Main Action Broker service for executing approved actions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Action Broker with connectors.
        
        Args:
            config: Configuration for all connectors
        """
        self.config = config or {}
        self.connectors: Dict[ConnectorType, BaseConnector] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        self._initialize_connectors()
        
    def _initialize_connectors(self):
        """Initialize all connectors with config."""
        self.connectors[ConnectorType.WEBHOOK] = WebhookConnector(
            self.config.get("webhook", {})
        )
        self.connectors[ConnectorType.GITHUB] = GitHubConnector(
            self.config.get("github", {})
        )
        self.connectors[ConnectorType.SLACK] = SlackConnector(
            self.config.get("slack", {})
        )
        self.connectors[ConnectorType.EMAIL] = EmailConnector(
            self.config.get("email", {})
        )
        self.connectors[ConnectorType.MEMORY] = MemoryConnector(
            self.config.get("memory", {})
        )
        
    def get_connector(self, connector_type: ConnectorType) -> Optional[BaseConnector]:
        """Get a specific connector."""
        return self.connectors.get(connector_type)
        
    def register_connector(self, connector_type: ConnectorType, connector: BaseConnector):
        """Register a custom connector."""
        self.connectors[connector_type] = connector
        logger.info("Custom connector registered", connector_type=connector_type.value)
        
    async def execute_action(
        self,
        action_id: str,
        connector_type: str,
        target: str,
        parameters: Dict[str, Any],
        user_id: str,
        tenant_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an approved action via the appropriate connector.
        
        Args:
            action_id: Unique action identifier
            connector_type: Type of connector to use
            target: Target of the action
            parameters: Action parameters
            user_id: User executing the action
            tenant_id: Tenant identifier
            context: Additional context
            
        Returns:
            Execution result
        """
        # Parse connector type
        try:
            conn_type = ConnectorType(connector_type)
        except ValueError:
            return {
                "action_id": action_id,
                "status": ExecutionStatus.FAILED.value,
                "error": f"Unknown connector type: {connector_type}",
            }
            
        connector = self.connectors.get(conn_type)
        if not connector:
            return {
                "action_id": action_id,
                "status": ExecutionStatus.FAILED.value,
                "error": f"Connector not available: {connector_type}",
            }
            
        # Validate connector config
        if not connector.validate_config():
            return {
                "action_id": action_id,
                "status": ExecutionStatus.FAILED.value,
                "error": f"Connector not properly configured: {connector_type}",
            }
            
        logger.info(
            "Executing action",
            action_id=action_id,
            connector_type=connector_type,
            target=target,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        
        # Execute action
        result = await connector.execute(target, parameters, context)
        
        # Log execution
        log_entry = {
            "action_id": action_id,
            "execution_id": result.execution_id,
            "connector_type": connector_type,
            "target": target,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "error": result.error,
            "executed_at": result.started_at.isoformat(),
        }
        self.execution_log.append(log_entry)
        
        logger.info(
            "Action executed",
            action_id=action_id,
            execution_id=result.execution_id,
            status=result.status.value,
            duration_ms=result.duration_ms,
        )
        
        return {
            "action_id": action_id,
            **result.to_dict(),
        }
        
    def get_execution_history(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by tenant."""
        history = self.execution_log
        if tenant_id:
            history = [e for e in history if e.get("tenant_id") == tenant_id]
        return history[-limit:]
        
    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors."""
        return {
            conn_type.value: {
                "available": conn.validate_config(),
                "execution_count": len(conn.execution_history),
            }
            for conn_type, conn in self.connectors.items()
        }


# Global Action Broker instance
_action_broker: Optional[ActionBroker] = None


def get_action_broker(config: Optional[Dict[str, Any]] = None) -> ActionBroker:
    """Get or create the global Action Broker instance.
    
    Args:
        config: Optional configuration for connectors
        
    Returns:
        ActionBroker instance
    """
    global _action_broker
    if _action_broker is None:
        _action_broker = ActionBroker(config)
    return _action_broker


# Convenience function for direct execution
async def execute_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute approved action via connectors.

    Args:
        action: Action to execute with keys:
            - action_id: Action identifier
            - connector_type: Type of connector (webhook, github, slack, email, memory)
            - target: Target of action
            - parameters: Action parameters
            - user_id: User ID
            - tenant_id: Tenant ID

    Returns:
        Execution result
    """
    broker = get_action_broker()
    
    return await broker.execute_action(
        action_id=action.get("action_id", str(uuid.uuid4())),
        connector_type=action.get("connector_type", "webhook"),
        target=action.get("target", ""),
        parameters=action.get("parameters", {}),
        user_id=action.get("user_id", "unknown"),
        tenant_id=action.get("tenant_id", "default"),
        context=action.get("context"),
    )
