"""Role-Based Access Control (RBAC) Middleware.

Implements role-based access control for API endpoints:
- Roles: admin, editor, viewer, agent
- Permissions: Per-endpoint access control
- Tenant isolation: Users can only access their tenant's data
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from functools import wraps
from fastapi import HTTPException, Request, status, Depends
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


class Role(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"         # Full access
    EDITOR = "editor"       # Create, read, update
    VIEWER = "viewer"       # Read only
    AGENT = "agent"         # Agent-specific permissions
    SYSTEM = "system"       # Internal system operations


class Permission(str, Enum):
    """Permissions for various operations."""
    # Memory permissions
    MEMORY_CREATE = "memory:create"
    MEMORY_READ = "memory:read"
    MEMORY_UPDATE = "memory:update"
    MEMORY_DELETE = "memory:delete"
    MEMORY_EXPORT = "memory:export"
    MEMORY_IMPORT = "memory:import"
    
    # Query permissions
    QUERY_EXECUTE = "query:execute"
    QUERY_BATCH = "query:batch"
    
    # Action permissions
    ACTION_SUBMIT = "action:submit"
    ACTION_APPROVE = "action:approve"
    ACTION_EXECUTE = "action:execute"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_TENANTS = "admin:tenants"
    ADMIN_POLICIES = "admin:policies"
    ADMIN_ANALYTICS = "admin:analytics"
    ADMIN_WEBHOOKS = "admin:webhooks"
    
    # System permissions
    SYSTEM_HEALTH = "system:health"
    SYSTEM_METRICS = "system:metrics"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Admin has all permissions
        Permission.MEMORY_CREATE,
        Permission.MEMORY_READ,
        Permission.MEMORY_UPDATE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_EXPORT,
        Permission.MEMORY_IMPORT,
        Permission.QUERY_EXECUTE,
        Permission.QUERY_BATCH,
        Permission.ACTION_SUBMIT,
        Permission.ACTION_APPROVE,
        Permission.ACTION_EXECUTE,
        Permission.ADMIN_USERS,
        Permission.ADMIN_TENANTS,
        Permission.ADMIN_POLICIES,
        Permission.ADMIN_ANALYTICS,
        Permission.ADMIN_WEBHOOKS,
        Permission.SYSTEM_HEALTH,
        Permission.SYSTEM_METRICS,
    },
    Role.EDITOR: {
        Permission.MEMORY_CREATE,
        Permission.MEMORY_READ,
        Permission.MEMORY_UPDATE,
        Permission.MEMORY_EXPORT,
        Permission.QUERY_EXECUTE,
        Permission.QUERY_BATCH,
        Permission.ACTION_SUBMIT,
        Permission.SYSTEM_HEALTH,
    },
    Role.VIEWER: {
        Permission.MEMORY_READ,
        Permission.QUERY_EXECUTE,
        Permission.SYSTEM_HEALTH,
    },
    Role.AGENT: {
        Permission.MEMORY_CREATE,
        Permission.MEMORY_READ,
        Permission.MEMORY_UPDATE,
        Permission.QUERY_EXECUTE,
        Permission.ACTION_SUBMIT,
        Permission.ACTION_EXECUTE,
    },
    Role.SYSTEM: {
        # System has all permissions
        Permission.MEMORY_CREATE,
        Permission.MEMORY_READ,
        Permission.MEMORY_UPDATE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_EXPORT,
        Permission.MEMORY_IMPORT,
        Permission.QUERY_EXECUTE,
        Permission.QUERY_BATCH,
        Permission.ACTION_SUBMIT,
        Permission.ACTION_APPROVE,
        Permission.ACTION_EXECUTE,
        Permission.ADMIN_USERS,
        Permission.ADMIN_TENANTS,
        Permission.ADMIN_POLICIES,
        Permission.ADMIN_ANALYTICS,
        Permission.ADMIN_WEBHOOKS,
        Permission.SYSTEM_HEALTH,
        Permission.SYSTEM_METRICS,
    },
}


# Endpoint to permission mapping
ENDPOINT_PERMISSIONS: Dict[str, List[Permission]] = {
    # Memory endpoints
    "POST /v1/memories": [Permission.MEMORY_CREATE],
    "GET /v1/memories": [Permission.MEMORY_READ],
    "GET /v1/memories/{id}": [Permission.MEMORY_READ],
    "PUT /v1/memories/{id}": [Permission.MEMORY_UPDATE],
    "DELETE /v1/memories/{id}": [Permission.MEMORY_DELETE],
    "POST /v1/memories/batch": [Permission.MEMORY_CREATE],
    
    # Query endpoints
    "POST /v1/query": [Permission.QUERY_EXECUTE],
    "POST /v1/query/batch": [Permission.QUERY_BATCH],
    
    # Action endpoints
    "POST /v1/actions/submit": [Permission.ACTION_SUBMIT],
    "POST /v1/actions/approve": [Permission.ACTION_APPROVE],
    
    # Export/Import
    "POST /v1/export": [Permission.MEMORY_EXPORT],
    "POST /v1/import": [Permission.MEMORY_IMPORT],
    
    # Analytics
    "GET /v1/analytics": [Permission.ADMIN_ANALYTICS],
    "GET /v1/analytics/overview": [Permission.MEMORY_READ],
    "GET /v1/analytics/usage": [Permission.ADMIN_ANALYTICS],
    
    # Monitoring
    "GET /v1/monitoring/health": [Permission.SYSTEM_HEALTH],
    "GET /v1/monitoring/metrics": [Permission.SYSTEM_METRICS],
    
    # Admin
    "GET /v1/admin/users": [Permission.ADMIN_USERS],
    "POST /v1/admin/policies": [Permission.ADMIN_POLICIES],
    "GET /v1/webhooks": [Permission.ADMIN_WEBHOOKS],
    "POST /v1/webhooks": [Permission.ADMIN_WEBHOOKS],
}


@dataclass
class UserContext:
    """User context with role and permissions."""
    user_id: str
    tenant_id: str
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    custom_permissions: Set[Permission] = field(default_factory=set)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or permission in self.custom_permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the given permissions."""
        return any(self.has_permission(p) for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the given permissions."""
        return all(self.has_permission(p) for p in permissions)


class RBACManager:
    """Manages RBAC policies and user contexts."""
    
    def __init__(self):
        """Initialize RBAC manager."""
        self.role_permissions = ROLE_PERMISSIONS.copy()
        self.user_roles: Dict[str, Dict[str, Role]] = {}  # tenant_id -> user_id -> role
        self.custom_permissions: Dict[str, Dict[str, Set[Permission]]] = {}  # tenant_id -> user_id -> permissions
        
    def set_user_role(self, tenant_id: str, user_id: str, role: Role):
        """Set the role for a user in a tenant."""
        if tenant_id not in self.user_roles:
            self.user_roles[tenant_id] = {}
        self.user_roles[tenant_id][user_id] = role
        
        logger.info(
            "User role set",
            tenant_id=tenant_id,
            user_id=user_id,
            role=role.value,
        )
        
    def get_user_role(self, tenant_id: str, user_id: str) -> Role:
        """Get the role for a user, defaulting to viewer."""
        return self.user_roles.get(tenant_id, {}).get(user_id, Role.VIEWER)
        
    def add_custom_permission(self, tenant_id: str, user_id: str, permission: Permission):
        """Add a custom permission to a user."""
        if tenant_id not in self.custom_permissions:
            self.custom_permissions[tenant_id] = {}
        if user_id not in self.custom_permissions[tenant_id]:
            self.custom_permissions[tenant_id][user_id] = set()
        self.custom_permissions[tenant_id][user_id].add(permission)
        
    def remove_custom_permission(self, tenant_id: str, user_id: str, permission: Permission):
        """Remove a custom permission from a user."""
        if tenant_id in self.custom_permissions and user_id in self.custom_permissions[tenant_id]:
            self.custom_permissions[tenant_id][user_id].discard(permission)
            
    def get_user_context(self, tenant_id: str, user_id: str) -> UserContext:
        """Build a user context with all permissions."""
        role = self.get_user_role(tenant_id, user_id)
        base_permissions = self.role_permissions.get(role, set())
        custom = self.custom_permissions.get(tenant_id, {}).get(user_id, set())
        
        return UserContext(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
            permissions=base_permissions,
            custom_permissions=custom,
        )
        
    def check_endpoint_permission(
        self,
        context: UserContext,
        method: str,
        path: str,
    ) -> bool:
        """Check if user can access an endpoint."""
        # Normalize path (remove IDs)
        normalized_path = self._normalize_path(path)
        endpoint_key = f"{method} {normalized_path}"
        
        required_permissions = ENDPOINT_PERMISSIONS.get(endpoint_key)
        
        if required_permissions is None:
            # No specific permissions required, allow access
            return True
            
        return context.has_any_permission(required_permissions)
        
    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing UUIDs and IDs with placeholders."""
        import re
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path,
            flags=re.IGNORECASE,
        )
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        return path


# Global RBAC manager instance
_rbac_manager = RBACManager()


def get_rbac_manager() -> RBACManager:
    """Get the global RBAC manager."""
    return _rbac_manager


class RBACMiddleware:
    """FastAPI middleware for RBAC enforcement."""
    
    def __init__(self, app, skip_paths: Optional[List[str]] = None):
        """Initialize RBAC middleware.
        
        Args:
            app: FastAPI application
            skip_paths: Paths to skip RBAC checking (e.g., health endpoints)
        """
        self.app = app
        self.skip_paths = skip_paths or ["/health", "/docs", "/openapi.json", "/v1/auth"]
        self.rbac = get_rbac_manager()
        
    async def __call__(self, scope, receive, send):
        """Process request through RBAC."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        path = scope.get("path", "")
        
        # Skip RBAC for certain paths
        if any(path.startswith(skip) for skip in self.skip_paths):
            await self.app(scope, receive, send)
            return
            
        # RBAC checking happens in the dependency
        await self.app(scope, receive, send)


def check_permission(required_permissions: List[Permission]):
    """Dependency factory to check permissions.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            ...,
            _: None = Depends(check_permission([Permission.MEMORY_CREATE])),
        ):
    """
    async def permission_checker(request: Request) -> None:
        # Get user from request state (set by auth dependency)
        user = getattr(request.state, "user", None)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
            
        tenant_id = user.get("tenant_id", "default")
        user_id = user.get("user_id", "unknown")
        
        rbac = get_rbac_manager()
        context = rbac.get_user_context(tenant_id, user_id)
        
        if not context.has_any_permission(required_permissions):
            logger.warning(
                "Permission denied",
                user_id=user_id,
                tenant_id=tenant_id,
                required=required_permissions,
                role=context.role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {[p.value for p in required_permissions]}",
            )
            
    return Depends(permission_checker)


def require_role(required_role: Role):
    """Dependency factory to require a specific role.
    
    Usage:
        @router.post("/admin/endpoint")
        async def admin_endpoint(
            ...,
            _: None = Depends(require_role(Role.ADMIN)),
        ):
    """
    async def role_checker(request: Request) -> None:
        user = getattr(request.state, "user", None)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
            
        tenant_id = user.get("tenant_id", "default")
        user_id = user.get("user_id", "unknown")
        
        rbac = get_rbac_manager()
        user_role = rbac.get_user_role(tenant_id, user_id)
        
        # Role hierarchy: ADMIN > EDITOR > VIEWER
        role_hierarchy = {
            Role.ADMIN: 4,
            Role.SYSTEM: 4,
            Role.EDITOR: 3,
            Role.AGENT: 2,
            Role.VIEWER: 1,
        }
        
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
            logger.warning(
                "Role check failed",
                user_id=user_id,
                tenant_id=tenant_id,
                required_role=required_role.value,
                user_role=user_role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role.value}",
            )
            
    return Depends(role_checker)

