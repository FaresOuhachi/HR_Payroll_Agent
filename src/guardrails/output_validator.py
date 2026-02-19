"""
Output Guardrails — Validate Agent Responses Before They Reach the User
=============================================================================
CONCEPT: Output Guardrails

Input guardrails protect the agent FROM the user.
Output guardrails protect the user FROM the agent.

WHY VALIDATE OUTPUTS TOO? WHAT COULD GO WRONG IF WE ONLY CHECK INPUTS?

  Even with perfect input validation, the agent can produce dangerous outputs:

  1. PII Leakage from Database
     The user asks "Tell me about the Engineering department." The agent
     queries the database and includes SSNs, bank accounts, or home addresses
     in its response — even though the user never provided any PII.

     Input guardrails CANNOT catch this because the PII comes from the
     database, not the user's input.

  2. Hallucinated Sensitive Data
     LLMs can hallucinate realistic-looking SSNs, credit card numbers, or
     other PII. Even if the data is fake, including it in responses:
       - Trains users to expect PII in responses (bad security culture)
       - Could accidentally match a real person's data
       - Violates data handling policies

  3. Excessive Information Disclosure
     The agent might include internal system details, database schema names,
     tool execution logs, or error stack traces in user-facing responses.
     This information helps attackers understand the system's internals.

  4. Prompt Injection Echo
     If a prompt injection partially succeeds, the agent might repeat the
     injected instructions in its response, revealing that the attack
     was processed (useful information for an attacker iterating on attacks).

  5. Response Length Exploitation
     Without output limits, a compromised agent could return gigabytes of
     data, causing client-side crashes or excessive bandwidth usage.

  THE DEFENSE-IN-DEPTH PRINCIPLE:
    Security works in LAYERS. Each layer catches what the previous layer missed.

    Layer 1: Input Guardrails    → Catch malicious/accidental user input
    Layer 2: Tool Governance     → Restrict what the agent can do
    Layer 3: Output Guardrails   → Catch dangerous agent responses  <-- THIS FILE
    Layer 4: Human-in-the-Loop   → Human reviews high-risk operations

    If any single layer fails, the others still provide protection.

=============================================================================
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Reuse the same ValidationResult structure from input_validator
# =============================================================================
# CONCEPT: DRY (Don't Repeat Yourself)
# We define ValidationResult again here rather than importing from
# input_validator.py. Why? To keep each guardrail module self-contained
# and independently deployable. In a larger system, these might live in
# separate microservices.
#
# In practice, you could also extract ValidationResult into a shared
# module like src/guardrails/types.py and import from there.
# =============================================================================
@dataclass
class ValidationResult:
    """
    The outcome of running an agent response through output guardrail checks.

    Attributes:
        is_valid: Whether the response passed all checks. If False, the system
                  should either sanitize the response or return a generic error
                  message to the user instead.
        violations: List of human-readable descriptions of what failed.
        sanitized_input: The cleaned version of the response. PII patterns
                         are replaced with safe placeholders.
                         Note: This field is named "sanitized_input" for
                         consistency with InputGuardrail's ValidationResult,
                         even though it holds the sanitized OUTPUT.
    """
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    sanitized_input: str = ""


# =============================================================================
# PII Patterns for Output Scanning
# =============================================================================
# CONCEPT: Same PII Detection in Outputs
#
# We use the same PII patterns as input validation. The patterns detect PII
# regardless of whether it came from user input, database queries, or LLM
# hallucination. The source doesn't matter — PII in the output is always
# a problem.
#
# ADDITIONAL OUTPUT-SPECIFIC PATTERNS:
# In outputs, we also watch for patterns that wouldn't appear in user input
# but could come from database queries:
#   - Salary amounts with identifiers (e.g., "John Doe earns $150,000")
#   - Internal IDs that shouldn't be exposed
#   - Database error messages containing table/column names
# =============================================================================
OUTPUT_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "SSN_NO_DASHES",
        re.compile(r"\b\d{9}\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    ),
    (
        "BANK_ACCOUNT",
        re.compile(r"\b[0-3]\d{8}\b"),
    ),
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "PHONE_NUMBER",
        re.compile(r"\b(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-.]?\d{4}\b"),
    ),
]

# Map PII type to its redaction placeholder (same as input)
OUTPUT_PII_REDACTION_MAP: dict[str, str] = {
    "SSN": "[SSN_REDACTED]",
    "SSN_NO_DASHES": "[SSN_REDACTED]",
    "CREDIT_CARD": "[CARD_REDACTED]",
    "BANK_ACCOUNT": "[ACCOUNT_REDACTED]",
    "EMAIL_ADDRESS": "[EMAIL_REDACTED]",
    "PHONE_NUMBER": "[PHONE_REDACTED]",
}


# =============================================================================
# Sensitive Internal Patterns — Things that should NEVER appear in user-facing output
# =============================================================================
# CONCEPT: Information Disclosure Prevention
#
# These patterns detect internal system information that could leak
# through agent responses. Even partial exposure helps attackers:
#   - Database errors reveal table names and column names
#   - Stack traces reveal file paths and framework versions
#   - Internal UUIDs reveal record identifiers that could be used in API attacks
# =============================================================================
INTERNAL_LEAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "SQL_ERROR",
        # Catches common SQLAlchemy / PostgreSQL error fragments
        re.compile(
            r"(?:sqlalchemy\.exc\.|psycopg2\.|asyncpg\.|"
            r"DETAIL:|HINT:|ProgrammingError|OperationalError)",
            re.IGNORECASE,
        ),
    ),
    (
        "STACK_TRACE",
        # Python traceback header
        re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    ),
    (
        "FILE_PATH",
        # Absolute file paths that could reveal server structure
        re.compile(r"(?:/home/|/usr/|/var/|C:\\\\Users\\\\)\S+\.py"),
    ),
]


# =============================================================================
# Response length limits
# =============================================================================
# CONCEPT: Output Length Limits
#
# Why limit output length?
#   1. Bandwidth — Large responses consume network bandwidth and cost money
#   2. Client safety — Some clients may crash on extremely large responses
#   3. Cost — If the output is fed back into another LLM call, long outputs
#      consume expensive tokens
#   4. Usability — Humans can't read a 50,000-character response anyway
#
# 10,000 characters is generous for an HR chatbot response. Typical responses
# are 200-500 characters. Payroll breakdowns with multiple employees might
# reach 2,000-3,000 characters.
# =============================================================================
MAX_OUTPUT_LENGTH = 10_000


class OutputGuardrail:
    """
    Output guardrail that validates and sanitizes agent responses before
    they are returned to the user.

    ARCHITECTURE:
    This class runs a pipeline of checks on every agent output:

        Agent Response → [Length Check] → [PII Detection] → [Internal Leak Check] → Result

    The checks mirror the input guardrail pattern, but focus on different
    threats (database PII leakage, internal info disclosure, etc.).

    USAGE:
        guardrail = OutputGuardrail()
        result = guardrail.validate(agent_response_text)
        if not result.is_valid:
            # Option 1: Return sanitized version (PII redacted)
            return result.sanitized_input
            # Option 2: Return generic error (more conservative)
            return "I encountered an issue generating your response."

    INTEGRATION WITH LANGGRAPH:
    In the agent's final output node:

        output_guard = OutputGuardrail()
        result = output_guard.validate(state["final_response"])
        if not result.is_valid:
            # Log the violations for the security team
            logger.error("Output violations: %s", result.violations)
            # Return the sanitized version to the user
            state["final_response"] = result.sanitized_input
    """

    def __init__(
        self,
        max_length: int = MAX_OUTPUT_LENGTH,
        pii_patterns: list[tuple[str, re.Pattern]] | None = None,
        internal_patterns: list[tuple[str, re.Pattern]] | None = None,
    ):
        """
        Initialize the output guardrail.

        Args:
            max_length: Maximum allowed response length in characters.
            pii_patterns: Custom PII regex patterns for output scanning.
                          Defaults to the module-level OUTPUT_PII_PATTERNS.
            internal_patterns: Patterns for detecting internal info leaks.
                               Defaults to INTERNAL_LEAK_PATTERNS.
        """
        self.max_length = max_length
        self.pii_patterns = pii_patterns or OUTPUT_PII_PATTERNS
        self.internal_patterns = internal_patterns or INTERNAL_LEAK_PATTERNS

    def validate(self, response: str) -> ValidationResult:
        """
        Run ALL guardrail checks on the agent's response and return the result.

        This is the main entry point. It orchestrates three sub-checks:
          1. Response length validation
          2. PII detection in the output (with redaction)
          3. Internal information leak detection

        Args:
            response: The agent's raw response text.

        Returns:
            ValidationResult with:
              - is_valid=True if all checks pass
              - violations listing each failed check
              - sanitized_input with PII and internal info redacted
        """
        violations: list[str] = []
        sanitized = response

        # ------------------------------------------------------------------
        # Check 1: Response Length
        # ------------------------------------------------------------------
        length_violations = self._check_length(response)
        violations.extend(length_violations)

        # If too long, truncate the sanitized version (don't just flag it)
        if length_violations:
            sanitized = sanitized[: self.max_length] + "... [TRUNCATED]"

        # ------------------------------------------------------------------
        # Check 2: PII Detection in Output
        # ------------------------------------------------------------------
        # This catches PII that came from database queries or LLM hallucination.
        # The user never sent this PII — it was generated during processing.
        # ------------------------------------------------------------------
        pii_violations, sanitized = self._detect_and_redact_pii(sanitized)
        violations.extend(pii_violations)

        # ------------------------------------------------------------------
        # Check 3: Internal Information Leak Detection
        # ------------------------------------------------------------------
        # Catches database errors, stack traces, file paths, etc.
        # ------------------------------------------------------------------
        leak_violations = self._detect_internal_leaks(sanitized)
        violations.extend(leak_violations)

        if violations:
            logger.warning(
                "Output guardrail violations detected: %s",
                violations,
            )

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            sanitized_input=sanitized,
        )

    def _check_length(self, response: str) -> list[str]:
        """
        Validate that the response does not exceed the maximum allowed length.

        Args:
            response: The agent's raw response text.

        Returns:
            A list with one violation message if the response is too long,
            or an empty list if the length is acceptable.
        """
        if len(response) > self.max_length:
            return [
                f"Response too long: {len(response)} characters "
                f"(maximum allowed: {self.max_length})"
            ]
        return []

    def _detect_and_redact_pii(
        self, response: str
    ) -> tuple[list[str], str]:
        """
        Scan the agent's response for PII patterns and replace matches
        with safe placeholders.

        This is the OUTPUT-side equivalent of input PII detection. The key
        difference is that PII in outputs typically comes from:
          - Database queries (employee records contain SSNs, emails, etc.)
          - LLM hallucination (the model generates realistic-looking PII)

        Unlike input PII (which the user controls), output PII indicates
        a DATA HANDLING problem in the system — the agent is exposing
        information it should be filtering.

        Args:
            response: The agent's response text (possibly already truncated).

        Returns:
            Tuple of (violations_list, sanitized_response).
        """
        violations: list[str] = []
        sanitized = response

        for pii_type, pattern in self.pii_patterns:
            matches = pattern.findall(sanitized)
            if matches:
                violations.append(
                    f"PII detected in output ({pii_type}): "
                    f"{len(matches)} instance(s) found and redacted."
                )
                placeholder = OUTPUT_PII_REDACTION_MAP.get(
                    pii_type, "[REDACTED]"
                )
                sanitized = pattern.sub(placeholder, sanitized)

        return violations, sanitized

    def _detect_internal_leaks(self, response: str) -> list[str]:
        """
        Check for internal system information in the agent's response.

        Internal information includes:
          - Database error messages (reveal table/column names)
          - Python stack traces (reveal file paths and frameworks)
          - Internal file paths (reveal server directory structure)

        These should NEVER appear in user-facing responses. If detected,
        we flag them but do NOT try to redact them (the entire response
        should be replaced with a generic error message instead).

        WHY NOT REDACT?
        Unlike PII (which has clear boundaries — an SSN is 9 digits),
        internal errors can span multiple lines and are difficult to
        cleanly extract. It's safer to flag and replace the entire response.

        Args:
            response: The agent's response text.

        Returns:
            A list of violation messages for each detected leak pattern.
        """
        violations: list[str] = []

        for leak_type, pattern in self.internal_patterns:
            if pattern.search(response):
                violations.append(
                    f"Internal information leak detected ({leak_type}): "
                    f"Response contains system internals that should not "
                    f"be exposed to users."
                )

        return violations
