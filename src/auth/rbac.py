"""
Role-Based Access Control (RBAC) — Permission Matrix
=============================================================================
CONCEPT: What is RBAC?

RBAC (Role-Based Access Control) is an authorization model where access
permissions are assigned to ROLES, not individual users. Users are then
assigned one or more roles, and they inherit all the permissions of their
assigned roles.

WITHOUT RBAC (per-user permissions):
    User "Alice" -> can_view_payroll, can_run_payroll, can_approve_payroll, ...
    User "Bob"   -> can_view_own_data, can_view_own_payslip, ...
    User "Carol" -> can_view_payroll, can_run_payroll, ...

    Problem: With 100 employees and 50 permissions, you need to manage
    5,000 permission assignments. Adding a new permission requires
    updating every user individually.

WITH RBAC (role-based permissions):
    Role "admin"    -> all permissions
    Role "manager"  -> payroll + approvals + view all employees
    Role "employee" -> view own data only

    User "Alice" -> role: admin      (inherits all admin permissions)
    User "Bob"   -> role: employee   (inherits all employee permissions)
    User "Carol" -> role: manager    (inherits all manager permissions)

    Benefit: Adding a new permission to the "manager" role automatically
    applies to all managers. Adding a new manager just requires assigning
    the role.

WHY IS RBAC IMPORTANT FOR HR/PAYROLL SYSTEMS?

HR and payroll systems handle some of the most sensitive data in any
organization:
  - Salary information (competitive advantage, employee privacy)
  - Tax details (PII — Personally Identifiable Information)
  - Performance reviews (confidential)
  - Medical/benefits data (HIPAA in the US, GDPR in the EU)

Without proper RBAC:
  - An employee could view other employees' salaries
  - A manager could approve their own raise
  - An agent could run unauthorized payroll
  - Audit logs could be tampered with

This module defines a PERMISSION MATRIX that maps:
    (role, resource, action) -> allowed: bool

RBAC HIERARCHY IN THIS SYSTEM:
  admin (highest privilege)
    |
    +-- Can do EVERYTHING a manager can do, PLUS:
    |   - Manage users (create, deactivate)
    |   - View audit logs
    |   - Configure system settings
    |   - Delete records
    |
  manager
    |
    +-- Can do EVERYTHING an employee can do, PLUS:
    |   - Run payroll calculations
    |   - Approve/reject operations
    |   - View all employee data
    |   - Generate reports
    |
  employee (lowest privilege)
    |
    +-- Can ONLY:
        - View their own profile
        - View their own payslips
        - Use the employee chatbot agent (for their own queries)

THE PERMISSION MATRIX:
  Each entry is: {resource: {action: True/False}}

  Resources are the "things" users interact with:
    - payroll: Payroll runs and calculations
    - employees: Employee records
    - approvals: Approval workflow
    - agents: AI agent execution
    - reports: Generated reports
    - users: User management
    - audit_logs: System audit trail

  Actions are what users can do with each resource:
    - view, view_all, view_own
    - create, update, delete
    - run, approve, reject
    - execute
=============================================================================
"""

from typing import Any


# =============================================================================
# Permission Matrix
# =============================================================================
# This dictionary defines EVERY permission in the system.
# Structure: ROLE -> RESOURCE -> ACTION -> bool
#
# To add a new permission:
#   1. Add the resource/action to the appropriate role(s)
#   2. Use check_permission() in your route or business logic
#
# DESIGN DECISION: Why a dictionary instead of a database table?
#   - Permissions change rarely (they're part of the application design)
#   - A dictionary is simpler, faster, and version-controlled with code
#   - No database query needed for every permission check (performance)
#   - Changes are reviewed in code review (safer than DB changes)
#
#   For more complex scenarios (dynamic roles, per-org customization),
#   you would move permissions to the database. But for this system,
#   three fixed roles are sufficient.
# =============================================================================

PERMISSIONS: dict[str, dict[str, dict[str, bool]]] = {
    # -----------------------------------------------------------------
    # ADMIN — Full access to everything
    # -----------------------------------------------------------------
    # Admins are system operators. They can manage users, view audit logs,
    # and perform any operation. In an HR context, this is typically the
    # IT department or HR director.
    "admin": {
        "payroll": {
            "view": True,
            "create": True,
            "run": True,
            "approve": True,
            "reject": True,
            "delete": True,
        },
        "employees": {
            "view_all": True,
            "view_own": True,
            "create": True,
            "update": True,
            "delete": True,
        },
        "approvals": {
            "view": True,
            "approve": True,
            "reject": True,
        },
        "agents": {
            "execute": True,      # Can run any agent
            "view_logs": True,    # Can view all agent execution logs
        },
        "reports": {
            "view": True,
            "create": True,
            "export": True,
        },
        "users": {
            "view": True,
            "create": True,
            "update": True,
            "deactivate": True,
        },
        "audit_logs": {
            "view": True,
        },
    },

    # -----------------------------------------------------------------
    # MANAGER — Payroll operations + team oversight
    # -----------------------------------------------------------------
    # Managers can run payroll, approve operations, and view all employee
    # data. They cannot manage users or view raw audit logs.
    # In an HR context, this is a payroll manager or HR manager.
    "manager": {
        "payroll": {
            "view": True,
            "create": True,
            "run": True,
            "approve": True,
            "reject": True,
            "delete": False,       # Managers cannot delete payroll runs
        },
        "employees": {
            "view_all": True,      # Can view all employees in their scope
            "view_own": True,
            "create": False,       # Only admins create employee records
            "update": True,        # Can update employee details
            "delete": False,       # Cannot delete employee records
        },
        "approvals": {
            "view": True,
            "approve": True,
            "reject": True,
        },
        "agents": {
            "execute": True,       # Can run payroll and employee agents
            "view_logs": True,     # Can view agent logs for their operations
        },
        "reports": {
            "view": True,
            "create": True,
            "export": True,
        },
        "users": {
            "view": False,
            "create": False,
            "update": False,
            "deactivate": False,
        },
        "audit_logs": {
            "view": False,         # Managers cannot access audit logs
        },
    },

    # -----------------------------------------------------------------
    # EMPLOYEE — View own data only
    # -----------------------------------------------------------------
    # Employees can only see their own information: profile, payslips,
    # and use the chatbot to ask questions about their data.
    # They cannot see other employees' data or run payroll.
    "employee": {
        "payroll": {
            "view": False,         # Cannot view payroll runs
            "create": False,
            "run": False,
            "approve": False,
            "reject": False,
            "delete": False,
        },
        "employees": {
            "view_all": False,     # Cannot view other employees
            "view_own": True,      # CAN view their own data
            "create": False,
            "update": False,       # Cannot modify records (even their own)
            "delete": False,
        },
        "approvals": {
            "view": False,
            "approve": False,
            "reject": False,
        },
        "agents": {
            "execute": True,       # Can use the employee chatbot agent
            "view_logs": False,    # Cannot view agent execution logs
        },
        "reports": {
            "view": False,
            "create": False,
            "export": False,
        },
        "users": {
            "view": False,
            "create": False,
            "update": False,
            "deactivate": False,
        },
        "audit_logs": {
            "view": False,
        },
    },
}


def check_permission(role: str, resource: str, action: str) -> bool:
    """
    Check whether a role has permission to perform an action on a resource.

    This is the central authorization function. Call it whenever you need
    to verify whether the current user is allowed to do something.

    PARAMETERS:
      role: The user's role ("admin", "manager", "employee").
      resource: The resource being accessed ("payroll", "employees", etc.).
      action: The action being performed ("view", "create", "run", etc.).

    RETURNS:
      True if the role has permission, False otherwise.
      Returns False for unknown roles, resources, or actions (deny by default).

    SECURITY PRINCIPLE: Default Deny
      If a role, resource, or action is not explicitly listed in the
      PERMISSIONS matrix, access is DENIED. This is the "principle of
      least privilege" — users can only do what is explicitly allowed.

      This is critical for security: if you add a new resource but forget
      to update the permissions, it's automatically locked down (instead
      of accidentally being open to everyone).

    USAGE:
        # In a route handler:
        if not check_permission(user.role, "payroll", "run"):
            raise HTTPException(status_code=403, detail="Not authorized")

        # In agent tool logic:
        if not check_permission(user.role, "employees", "view_all"):
            return "You don't have permission to view other employees' data."

        # Combined with the dependency system:
        # For simple role checks, use require_role() from dependencies.py.
        # For granular resource/action checks, use check_permission().

    EXAMPLE LOOKUPS:
        check_permission("admin", "payroll", "delete")     -> True
        check_permission("manager", "payroll", "delete")   -> False
        check_permission("employee", "employees", "view_own") -> True
        check_permission("employee", "employees", "view_all") -> False
        check_permission("unknown_role", "payroll", "view")   -> False  (default deny)
        check_permission("admin", "unknown_resource", "view") -> False  (default deny)
    """
    # Look up the role's permissions. If the role is unknown, return
    # an empty dict (which means no permissions -> deny all).
    role_permissions: dict[str, dict[str, bool]] = PERMISSIONS.get(role, {})

    # Look up the resource permissions within this role. If the resource
    # is unknown, return an empty dict (deny all actions on it).
    resource_permissions: dict[str, bool] = role_permissions.get(resource, {})

    # Look up the specific action. If the action is unknown, return False
    # (deny by default). This is the "default deny" principle.
    return resource_permissions.get(action, False)


def get_role_permissions(role: str) -> dict[str, dict[str, bool]]:
    """
    Get the complete permission set for a role.

    Useful for:
      - Displaying a user's permissions in a UI
      - Auditing what each role can do
      - API endpoints that return the current user's capabilities

    PARAMETERS:
      role: The role to look up ("admin", "manager", "employee").

    RETURNS:
      A dictionary of resource -> action -> bool for the given role.
      Returns an empty dictionary for unknown roles.

    USAGE:
        permissions = get_role_permissions("manager")
        # Returns:
        # {
        #     "payroll": {"view": True, "create": True, "run": True, ...},
        #     "employees": {"view_all": True, "view_own": True, ...},
        #     ...
        # }

        # In an API endpoint:
        @app.get("/auth/me/permissions")
        async def my_permissions(user: User = Depends(get_current_user)):
            return get_role_permissions(user.role)
    """
    return PERMISSIONS.get(role, {})


def get_allowed_actions(role: str, resource: str) -> list[str]:
    """
    Get a list of actions a role can perform on a specific resource.

    This is a convenience function for building UI elements (e.g.,
    showing/hiding buttons based on what the user can do).

    PARAMETERS:
      role: The user's role.
      resource: The resource to check permissions for.

    RETURNS:
      A list of action names that are allowed.
      Returns an empty list for unknown roles or resources.

    USAGE:
        actions = get_allowed_actions("manager", "payroll")
        # Returns: ["view", "create", "run", "approve", "reject"]
        # Note: "delete" is excluded because it's False for managers.

        # In a UI context:
        if "approve" in get_allowed_actions(user.role, "approvals"):
            show_approve_button()
    """
    role_permissions = PERMISSIONS.get(role, {})
    resource_permissions = role_permissions.get(resource, {})
    return [action for action, allowed in resource_permissions.items() if allowed]
