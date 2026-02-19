"""
FastAPI Authentication Dependencies
=============================================================================
CONCEPT: What is Dependency Injection in FastAPI?

Dependency Injection (DI) is a pattern where a function declares what it
NEEDS (its dependencies), and the framework PROVIDES them automatically.
In FastAPI, this is done via the `Depends()` function.

WITHOUT dependency injection:
    @app.get("/employees")
    async def list_employees(request: Request):
        # Manually extract and validate the token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401)
        token = auth_header.split(" ")[1]
        payload = verify_token(token)
        user = await get_user_from_db(payload["sub"])
        if user is None or not user.is_active:
            raise HTTPException(status_code=401)
        # NOW you can finally use the user
        ...

WITH dependency injection:
    @app.get("/employees")
    async def list_employees(user: User = Depends(get_current_user)):
        # `user` is already authenticated and validated!
        ...

Benefits:
  1. DRY (Don't Repeat Yourself) — Auth logic is written once, used everywhere
  2. Testability — In tests, you can override the dependency:
       app.dependency_overrides[get_current_user] = lambda: mock_user
  3. Composability — Dependencies can depend on other dependencies:
       get_current_user depends on HTTPBearer (which extracts the token)
       require_role depends on get_current_user
  4. Automatic documentation — FastAPI adds auth requirements to OpenAPI docs

CONCEPT: How Does RBAC (Role-Based Access Control) Work with Dependencies?

RBAC restricts access based on the user's role. In this system:
  - admin: Can access everything
  - manager: Can access payroll operations and view all employees
  - employee: Can only view their own data

We implement RBAC as a dependency factory:
    require_role("admin", "manager") returns a dependency function that:
      1. Gets the current user (via get_current_user)
      2. Checks if the user's role is in the allowed list
      3. Raises 403 Forbidden if not authorized

Usage in routes:
    @app.post("/payroll/run")
    async def run_payroll(user: User = Depends(require_role("admin", "manager"))):
        # Only admins and managers can reach this code
        ...

    @app.get("/employees/me")
    async def get_my_data(user: User = Depends(require_role("employee", "manager", "admin"))):
        # All authenticated users can access this
        ...

The dependency chain looks like:
    HTTP Request
      -> HTTPBearer (extracts token from Authorization header)
        -> get_current_user (validates token, loads user from DB)
          -> require_role (checks if user's role is allowed)
            -> Your route handler (receives the validated user)
=============================================================================
"""

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.jwt import verify_token
from src.db.engine import get_db_session
from src.db.models import User
from src.db.repositories import get_user_by_username


# ---------------------------------------------------------------------------
# HTTPBearer Security Scheme
# ---------------------------------------------------------------------------
# HTTPBearer is a FastAPI security utility that:
#   1. Looks for the "Authorization" header in the request
#   2. Validates it starts with "Bearer "
#   3. Extracts the token (everything after "Bearer ")
#   4. Returns an HTTPAuthorizationCredentials object with .credentials
#
# If auto_error=True (default), it automatically returns a 401 response
# if the header is missing or malformed. This saves us from writing that
# validation logic ourselves.
#
# In the OpenAPI docs (Swagger UI), this adds a "lock" icon to protected
# routes, and provides a dialog for entering the Bearer token.
# ---------------------------------------------------------------------------
bearer_scheme = HTTPBearer(
    auto_error=True,
    description="Enter your JWT access token (obtained from POST /auth/token)",
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """
    FastAPI dependency that extracts and validates the current user from
    the JWT Bearer token in the Authorization header.

    THE AUTHENTICATION FLOW:
      1. HTTPBearer extracts the token from "Authorization: Bearer <token>"
      2. verify_token() validates the JWT signature and expiration
      3. We extract the username from the "sub" claim
      4. We load the full User object from the database
      5. We verify the user still exists and is active

    WHY LOAD FROM THE DATABASE?
      The JWT payload contains a snapshot of the user at login time. But
      the user's state can change between login and the current request:
        - Admin could deactivate the user (is_active = False)
        - User's role could change (promoted from employee to manager)
        - User could be deleted entirely
      Loading from the database ensures we have the CURRENT state.

    PARAMETERS:
      credentials: Automatically injected by FastAPI via HTTPBearer.
          Contains .credentials (the raw JWT token string).
      db: Automatically injected by FastAPI via get_db_session.
          An async database session for querying the users table.

    RETURNS:
      The authenticated User ORM object with current data from the database.

    RAISES:
      HTTPException 401: If the token is invalid, expired, or the user
          doesn't exist / is inactive.

    USAGE:
        @app.get("/protected")
        async def protected_route(
            current_user: User = Depends(get_current_user),
        ):
            return {"message": f"Hello, {current_user.full_name}"}
    """
    # Step 1: Validate the JWT token (signature + expiration)
    try:
        payload = verify_token(credentials.credentials)
    except ValueError as e:
        # ValueError is raised by verify_token() for invalid/expired tokens.
        # We convert it to a 401 HTTP response with a descriptive message.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            # WWW-Authenticate header tells the client which auth scheme
            # to use. This is required by the HTTP spec for 401 responses.
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Step 2: Extract the username from the token's "sub" claim
    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3: Load the user from the database
    # This ensures we have the CURRENT user state, not a stale snapshot.
    user = await get_user_by_username(db, username)

    if user is None:
        # The token references a user that no longer exists.
        # This can happen if the user was deleted after they logged in.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 4: Verify the user account is active
    if not user.is_active:
        # The user exists but has been deactivated by an admin.
        # Their tokens should no longer grant access.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(*allowed_roles: str) -> Callable:
    """
    Dependency factory that creates a role-checking dependency.

    CONCEPT: Dependency Factory Pattern
    A "dependency factory" is a function that RETURNS a dependency function.
    This lets you parameterize dependencies:
      - require_role("admin") -> returns a function that only allows admins
      - require_role("admin", "manager") -> allows admins AND managers

    This is more flexible than creating separate dependencies for each
    combination of roles:
      - get_admin_user()
      - get_admin_or_manager_user()
      - get_any_authenticated_user()
    With the factory, you just pass the roles you need.

    HOW IT WORKS:
      1. require_role("admin", "manager") is called at import time
         (when the route decorator is evaluated)
      2. It returns the inner `role_checker` function
      3. FastAPI calls `role_checker` for each request to the route
      4. `role_checker` depends on `get_current_user`, so the full
         auth chain runs before the role check

    PARAMETERS:
      *allowed_roles: One or more role strings that are permitted to
          access the route. Common values: "admin", "manager", "employee"

    RETURNS:
      A FastAPI dependency function that:
        - Returns the User if their role is in allowed_roles
        - Raises 403 Forbidden if the role is not allowed

    USAGE:
        # Only admins can access this route
        @app.delete("/employees/{id}")
        async def delete_employee(
            employee_id: UUID,
            user: User = Depends(require_role("admin")),
        ):
            ...

        # Admins and managers can access this route
        @app.post("/payroll/run")
        async def run_payroll(
            user: User = Depends(require_role("admin", "manager")),
        ):
            ...

        # All authenticated users (any role) can access this route
        @app.get("/me")
        async def get_my_profile(
            user: User = Depends(require_role("admin", "manager", "employee")),
        ):
            ...
    """

    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        """
        Inner dependency that checks the user's role against allowed_roles.

        This function is called by FastAPI for each request. It receives
        the authenticated user from get_current_user (which already
        validated the JWT and loaded the user from the database).

        The role check is straightforward:
          - If the user's role is in the allowed set -> return the user
          - If not -> raise 403 Forbidden

        WHY 403 (Forbidden) NOT 401 (Unauthorized)?
          - 401 means "you are not authenticated" (identity unknown)
          - 403 means "you are authenticated but not authorized" (identity
            known, but you don't have permission)
          The user successfully proved who they are (valid JWT), but their
          role doesn't grant access to this specific resource.
        """
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not authorized for this "
                    f"operation. Required roles: {', '.join(allowed_roles)}"
                ),
            )
        return current_user

    return role_checker
