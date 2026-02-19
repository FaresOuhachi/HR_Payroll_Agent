"""
Authentication API Endpoint
=============================================================================
CONCEPT: The Login Flow

This module provides the POST /auth/token endpoint — the entry point for
user authentication. Here's the complete flow:

  1. CLIENT sends: POST /auth/token
     Body: {"username": "admin_user", "password": "secretpass123"}

  2. SERVER looks up the user by username in the database

  3. SERVER verifies the password:
     - The database stores a HASHED password (e.g., "$2b$12$LJ3m4...")
     - We hash the provided password and compare it to the stored hash
     - If they match -> authentication successful
     - If they don't match -> return 401 Unauthorized

  4. SERVER creates a JWT token containing:
     - sub: the username (subject)
     - user_id: the user's UUID
     - role: the user's role (admin/manager/employee)

  5. SERVER returns:
     {
       "access_token": "eyJhbGciOiJIUzI1NiIs...",
       "token_type": "bearer",
       "role": "admin"
     }

  6. CLIENT stores the token and includes it in subsequent requests:
     Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

CONCEPT: Password Hashing with bcrypt

NEVER store passwords in plain text. If the database is compromised,
attackers would have everyone's passwords. Instead, we store a ONE-WAY
HASH of the password:

  plaintext: "secretpass123"
  bcrypt hash: "$2b$12$LJ3m4YFhKfG2kDxQ8ZXYZ.7tR9wN5FpPqV2Bc8mK3jN1pL0sH6vKi"

  Properties of bcrypt:
    - ONE-WAY: You cannot reverse the hash to get the password
    - SALTED: Each hash includes a random salt, so two users with the
      same password get different hashes (prevents rainbow table attacks)
    - SLOW BY DESIGN: bcrypt intentionally takes ~100ms to compute.
      This makes brute-force attacks impractical (trying 1M passwords
      would take ~28 hours instead of seconds)
    - ADAPTIVE: The "cost factor" (12 in "$2b$12$...") can be increased
      as hardware gets faster, keeping the hash function slow enough

  passlib is a Python library that provides a clean API for password
  hashing. CryptContext manages the hashing scheme and handles:
    - Hashing new passwords
    - Verifying passwords against stored hashes
    - Migrating to new hashing schemes (if you upgrade from bcrypt to argon2)
=============================================================================
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.jwt import create_access_token
from src.db.engine import get_db_session
from src.db.repositories import get_user_by_username


# =============================================================================
# Password Hashing Configuration
# =============================================================================
# CryptContext is passlib's main interface for password hashing.
#
# PARAMETERS:
#   schemes=["bcrypt"]: Use bcrypt as the hashing algorithm.
#     bcrypt is the gold standard for password hashing because:
#       - It's slow (intentionally) — takes ~100ms per hash
#       - It's salted — random salt is embedded in each hash
#       - It's adaptive — cost factor controls computational difficulty
#     Alternatives: argon2 (newer, memory-hard), scrypt (also memory-hard)
#
#   deprecated="auto": If we add a new scheme later (e.g., argon2),
#     passlib will automatically re-hash old bcrypt passwords to the
#     new scheme on the next successful login. This allows transparent
#     migration without forcing password resets.
# =============================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# Request/Response Schemas
# =============================================================================
# Pydantic models define the shape of request bodies and responses.
# They provide automatic validation, serialization, and OpenAPI
# documentation generation.

class LoginRequest(BaseModel):
    """
    Login request body schema.

    The client sends this JSON body when calling POST /auth/token.

    FIELDS:
      username: The user's login name (matches the users.username column).
      password: The user's plain-text password (will be verified against
                the stored bcrypt hash — NEVER logged or stored).

    EXAMPLE REQUEST:
      POST /auth/token
      Content-Type: application/json
      {
        "username": "admin_user",
        "password": "admin_password_123"
      }
    """
    username: str
    password: str


class TokenResponse(BaseModel):
    """
    Login response schema — returned on successful authentication.

    FIELDS:
      access_token: The signed JWT string. The client should store this
          securely (in memory, NOT localStorage for SPAs — localStorage
          is vulnerable to XSS attacks).
      token_type: Always "bearer". Tells the client how to send the token
          in subsequent requests (Authorization: Bearer <token>).
      role: The user's role. Included as a convenience so the frontend
          can adjust the UI without decoding the JWT. For example:
            - admin -> show admin dashboard
            - employee -> show personal dashboard only

    EXAMPLE RESPONSE:
      {
        "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "token_type": "bearer",
        "role": "admin"
      }
    """
    access_token: str
    token_type: str = "bearer"
    role: str


# =============================================================================
# Router Definition
# =============================================================================
# APIRouter groups related endpoints under a common prefix and tag.
# The tag appears in Swagger UI as a collapsible section.
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Login and obtain JWT access token",
    responses={
        200: {"description": "Successful authentication — returns JWT token"},
        401: {"description": "Invalid credentials — wrong username or password"},
    },
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """
    Authenticate a user and return a JWT access token.

    THE AUTHENTICATION ALGORITHM:
      1. Look up the user by username in the database
      2. If user not found -> return 401 (don't reveal which field was wrong)
      3. If user is deactivated -> return 401
      4. Verify the password against the stored bcrypt hash
      5. If password wrong -> return 401
      6. Create a JWT token with the user's claims
      7. Return the token

    SECURITY BEST PRACTICES IMPLEMENTED:
      1. GENERIC ERROR MESSAGE: We return "Incorrect username or password"
         for both wrong username AND wrong password. If we returned "User
         not found" vs "Wrong password", an attacker could enumerate valid
         usernames by checking which error they get.

      2. CONSTANT-TIME COMPARISON: bcrypt.verify() uses constant-time
         comparison internally, preventing timing attacks. (A timing attack
         measures how long verification takes — if comparison stops at the
         first wrong character, shorter times indicate "closer" passwords.)

      3. NO PASSWORD LOGGING: The request.password is never logged anywhere.
         Even in debug mode, we only log the username, not the password.

      4. INACTIVE USER CHECK: Deactivated users cannot authenticate, even
         if they know their password. This allows immediate access revocation.

    PARAMETERS:
      request: LoginRequest body containing username and password.
      db: Database session (injected by FastAPI via Depends).

    RETURNS:
      TokenResponse with access_token, token_type, and role.

    RAISES:
      HTTPException 401: If authentication fails for any reason.
    """
    # Step 1: Look up the user by username
    user = await get_user_by_username(db, request.username)

    # Step 2: Check if user exists
    # SECURITY: Don't reveal whether the username or password was wrong.
    # Use a generic error message for all failure cases.
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3: Check if the user account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 4: Verify the password against the stored hash
    # pwd_context.verify() hashes the plain-text password with the same
    # salt stored in user.hashed_password, then compares the results.
    # This is a CPU-intensive operation (~100ms) due to bcrypt's design.
    if not pwd_context.verify(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 5: Create the JWT access token
    # The token payload contains claims that identify the user and their
    # permissions. These claims are available to any downstream service
    # that validates the token (without a database lookup).
    access_token = create_access_token(
        data={
            "sub": user.username,           # Subject — who the token is for
            "user_id": str(user.id),        # UUID as string (JSON-serializable)
            "role": user.role,              # Role for client-side UI decisions
        }
    )

    # Step 6: Return the token to the client
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
    )
