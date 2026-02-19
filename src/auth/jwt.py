"""
JWT (JSON Web Token) Utilities
=============================================================================
CONCEPT: What is JWT and How Does Stateless Authentication Work?

Traditional session-based authentication:
  1. User logs in with username/password
  2. Server creates a session (stored in memory or database)
  3. Server sends a session_id cookie to the client
  4. Client sends the cookie with every request
  5. Server looks up the session_id in its storage to identify the user

  Problem: The server must store and look up sessions for EVERY request.
  This doesn't scale well — if you have 10 servers, they all need access
  to the session store (shared Redis/database). If the session store goes
  down, nobody can authenticate.

JWT-based (stateless) authentication:
  1. User logs in with username/password
  2. Server creates a JWT token containing user info (no server-side storage)
  3. Server sends the JWT to the client
  4. Client sends the JWT in the Authorization header with every request
  5. Server VALIDATES the JWT's signature (no database lookup needed!)

  Advantage: The server doesn't need to store anything. Each request is
  self-contained — the token carries all the information needed to
  authenticate the user. Any server can validate the token independently.

WHAT'S INSIDE A JWT?
  A JWT has three parts, separated by dots: HEADER.PAYLOAD.SIGNATURE

  1. HEADER (base64-encoded JSON):
     {"alg": "HS256", "typ": "JWT"}
     - alg: The signing algorithm (HS256 = HMAC-SHA256)
     - typ: Token type (always "JWT")

  2. PAYLOAD (base64-encoded JSON) — the "claims":
     {
       "sub": "admin_user",         <- Subject (who the token is for)
       "user_id": "uuid-abc-123",   <- Custom claim (user's database ID)
       "role": "admin",             <- Custom claim (user's role for RBAC)
       "exp": 1705312245,           <- Expiration time (Unix timestamp)
       "iat": 1705308645            <- Issued at (Unix timestamp)
     }
     IMPORTANT: The payload is base64-encoded, NOT encrypted. Anyone can
     decode it and read the contents. Never put sensitive data (passwords,
     SSNs) in the payload. The signature only ensures the payload hasn't
     been TAMPERED with, not that it's SECRET.

  3. SIGNATURE:
     HMAC-SHA256(
       base64(header) + "." + base64(payload),
       secret_key
     )
     The signature is created using a secret key that only the server
     knows. If anyone modifies the payload (e.g., changes role from
     "employee" to "admin"), the signature won't match, and the server
     will reject the token.

SECURITY CONSIDERATIONS:
  - The secret key (jwt_secret_key) must be truly random and kept secret.
    If compromised, attackers can forge valid tokens for any user.
  - Token expiration (exp) limits the damage if a token is stolen.
    Our default is 60 minutes (from settings.jwt_access_token_expire_minutes).
  - HS256 (HMAC) is symmetric — the same key signs and verifies.
    For microservices, RS256 (asymmetric) is better: one private key signs,
    many public keys can verify (without exposing the signing key).
=============================================================================
"""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from src.config import settings


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.

    This function takes a dictionary of claims (user information), adds
    standard JWT fields (expiration, issued-at), and signs the result
    with the server's secret key.

    PARAMETERS:
      data: A dictionary of claims to include in the token payload.
          Typically contains:
            - "sub" (subject): The username or user identifier
            - "user_id": The user's UUID (as a string)
            - "role": The user's role ("admin", "manager", "employee")
          You can add any JSON-serializable data, but keep tokens small
          (they're sent with every HTTP request).

      expires_delta: How long the token should be valid. If None, uses the
          default from settings (jwt_access_token_expire_minutes).
          Short-lived tokens (15-60 min) are more secure — if stolen, the
          attacker has a limited window. For longer sessions, use refresh
          tokens (not implemented here for simplicity).

    RETURNS:
      A signed JWT string like:
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbl..."

    USAGE:
        token = create_access_token(
            data={
                "sub": "admin_user",
                "user_id": str(user.id),
                "role": user.role,
            },
            expires_delta=timedelta(minutes=60),
        )
        # token is now a string that can be sent to the client

    HOW SIGNING WORKS (HS256):
      1. The header and payload are base64-encoded
      2. They're concatenated with a dot: base64(header).base64(payload)
      3. HMAC-SHA256 hashes this string with the secret key
      4. The hash becomes the signature
      5. Final token: base64(header).base64(payload).base64(signature)

      When verifying, the server re-computes the signature using the same
      secret key. If the computed signature matches the token's signature,
      the token is valid (hasn't been tampered with).
    """
    # Make a copy so we don't modify the caller's dictionary.
    # This is a defensive programming practice — the caller might reuse
    # the dict, and we don't want our additions (exp, iat) to leak.
    to_encode = data.copy()

    # Calculate expiration time
    if expires_delta is not None:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    # Add standard JWT claims
    # "exp" (expiration): After this time, the token is invalid.
    #   python-jose automatically checks this during decoding.
    # "iat" (issued at): When the token was created.
    #   Useful for auditing ("when did this user authenticate?")
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }
    )

    # Sign and encode the token
    # jwt.encode() does: base64(header) + "." + base64(payload) + "." + signature
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.

    This function verifies the token's signature and expiration, then
    returns the payload (claims). If the token is invalid for any reason,
    it raises a ValueError with a descriptive message.

    WHAT VALIDATION DOES:
      1. Decodes the base64-encoded header and payload
      2. Re-computes the signature using the server's secret key
      3. Compares the computed signature with the token's signature
         -> If they don't match: the token was tampered with (REJECT)
      4. Checks the "exp" claim against the current time
         -> If expired: the token is no longer valid (REJECT)
      5. Returns the payload dictionary if everything checks out

    PARAMETERS:
      token: The JWT string to validate.

    RETURNS:
      The decoded payload dictionary containing all claims:
        {
            "sub": "admin_user",
            "user_id": "uuid-abc-123",
            "role": "admin",
            "exp": 1705312245,
            "iat": 1705308645
        }

    RAISES:
      ValueError: If the token is invalid, expired, or malformed.
        The error message indicates the specific failure reason.

    USAGE:
        try:
            payload = verify_token(token_string)
            username = payload["sub"]
            role = payload["role"]
        except ValueError as e:
            # Token is invalid — reject the request
            raise HTTPException(status_code=401, detail=str(e))

    COMMON FAILURE SCENARIOS:
      1. Expired token: "Token has expired"
         The "exp" claim is in the past. User needs to log in again.

      2. Invalid signature: "Could not validate token"
         The token was tampered with (payload modified) or signed with
         a different secret key (e.g., from a different server).

      3. Malformed token: "Could not validate token"
         The token string isn't valid JWT format.

      4. Missing "sub" claim: "Token payload missing 'sub' claim"
         The token was created without a subject — this indicates a
         programming error in create_access_token().
    """
    try:
        # jwt.decode() performs signature verification AND expiration check.
        # If the signature doesn't match or the token is expired, it raises
        # JWTError or ExpiredSignatureError.
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as e:
        # JWTError is the base exception for all JWT-related errors:
        #   - ExpiredSignatureError (token past its "exp" time)
        #   - JWTClaimsError (invalid claims)
        #   - JWSError (invalid signature)
        # We convert to ValueError for a cleaner API surface.
        raise ValueError(f"Could not validate token: {e}") from e

    # Verify the token contains the required "sub" (subject) claim.
    # A token without "sub" is technically valid JWT but useless for auth.
    if "sub" not in payload:
        raise ValueError("Token payload missing 'sub' claim")

    return payload
