"""
Tool Governor — Access Control & Risk Classification for Agent Tool Usage
=============================================================================
CONCEPT: Tool Governance

In an agentic AI system, the LLM decides which tools to call. Without
governance, a single compromised or confused agent could:
  - Call ANY tool in the system (payroll agent accessing compliance tools)
  - Pass extreme parameters (adjust salary by $10 million)
  - Perform operations far beyond its intended scope

Tool governance is the practice of restricting WHICH tools each agent type
can call and validating the PARAMETERS before execution.

WHY RESTRICT TOOL ACCESS PER AGENT?

  Think of it like role-based access control (RBAC), but for AI agents:

  1. Principle of Least Privilege
     Each agent should only have access to the tools it needs. The "employee
     info" agent has no business calling "process_payroll" — even if a prompt
     injection tricks it into trying.

  2. Blast Radius Reduction
     If the payroll agent is compromised (via prompt injection or a bug),
     the damage is limited to payroll operations. It cannot access compliance
     reports, modify employee records, or approve its own operations.

  3. Auditability
     When each agent has a defined toolset, it's easy to audit: "Did this
     agent call a tool it shouldn't have?" This is a clear security signal.

  4. Predictability
     Restricting tools makes agent behavior more predictable. A payroll
     agent that can ONLY call payroll tools will always produce payroll-
     related outputs, regardless of what the user asks.

WHAT IS RISK CLASSIFICATION?

  Not all operations are equal. Querying an employee's name is low-risk.
  Processing a $200,000 payroll run is critical-risk. Risk classification
  determines whether an operation can proceed automatically or needs
  human approval.

  Financial Thresholds (configurable):
    - Low:      amount <= $10,000  → Auto-approve
    - Medium:   $10,001 - $50,000  → Log and flag
    - High:     $50,001 - $100,000 → Require manager approval
    - Critical: > $100,000         → Require admin approval

  These thresholds are domain-specific. In HR/payroll:
    - Individual salary queries are always low-risk
    - Single employee payroll is usually medium-risk
    - Department-wide payroll is often high-risk
    - Company-wide payroll runs are always critical-risk

=============================================================================
"""

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Allowlist — Which agent types can use which tools
# =============================================================================
# CONCEPT: Allowlist (Whitelist) vs. Blocklist (Blacklist)
#
# We use an ALLOWLIST approach: agents can ONLY call tools explicitly listed
# for their type. Anything not listed is denied by default.
#
# This is safer than a blocklist because:
#   - New tools are automatically BLOCKED until explicitly allowed
#   - You can't forget to block a dangerous tool (it's blocked by default)
#   - The security posture is "deny by default, allow by exception"
#
# Each key is an agent type (matching AgentState.target_agent values).
# Each value is a set of tool names the agent is allowed to call.
# =============================================================================
TOOL_ALLOWLIST: dict[str, set[str]] = {
    # -------------------------------------------------------------------------
    # Payroll Agent — Full access to payroll calculation and query tools.
    # This agent handles salary calculations, deductions, and payroll runs.
    # -------------------------------------------------------------------------
    "payroll": {
        "get_employee_info",
        "calculate_gross_pay",
        "calculate_deductions",
        "calculate_net_pay",
        "calculate_department_payroll",
        "get_leave_balance",
        "search_employees_by_department",
    },

    # -------------------------------------------------------------------------
    # Employee Agent — Read-only access to employee information.
    # This agent helps employees query their own data. It should NEVER be
    # able to modify payroll, approve transactions, or access other employees'
    # data (access control for "whose data" is handled separately).
    # -------------------------------------------------------------------------
    "employee": {
        "get_employee_info",
        "get_leave_balance",
        "calculate_net_pay",
    },

    # -------------------------------------------------------------------------
    # Compliance Agent — Access to reporting and audit tools.
    # This agent answers questions about policies and compliance.
    # It can read data but never modify it.
    # -------------------------------------------------------------------------
    "compliance": {
        "get_employee_info",
        "search_employees_by_department",
        "calculate_department_payroll",
    },

    # -------------------------------------------------------------------------
    # General/Router Agent — Minimal tools.
    # The router agent classifies intent and delegates to specialists.
    # It should rarely call tools directly.
    # -------------------------------------------------------------------------
    "general": {
        "get_employee_info",
    },
}


# =============================================================================
# Financial Risk Thresholds
# =============================================================================
# CONCEPT: Financial Control Thresholds
#
# These thresholds determine the risk level of financial operations.
# They should be configurable per organization (some companies may set
# the critical threshold at $50k, others at $500k).
#
# In a production system, these would come from the Settings/config,
# not hardcoded constants. We keep them here for clarity.
# =============================================================================
FINANCIAL_THRESHOLDS = {
    "medium": 10_000.0,    # Operations above $10k require logging
    "high": 50_000.0,      # Operations above $50k require manager approval
    "critical": 100_000.0, # Operations above $100k require admin approval
}


# =============================================================================
# Parameter Bounds — Limits on tool arguments
# =============================================================================
# CONCEPT: Parameter Validation
#
# Even if a tool is allowed, its parameters must be within safe bounds.
# For example, a salary adjustment tool should not accept negative amounts
# or amounts above a reasonable maximum.
#
# Each entry maps a tool name to a dict of parameter constraints.
# Each constraint is a dict with "min" and/or "max" values.
# =============================================================================
PARAMETER_BOUNDS: dict[str, dict[str, dict[str, float]]] = {
    "calculate_deductions": {
        "gross_pay": {
            "min": 0.0,           # Gross pay cannot be negative
            "max": 1_000_000.0,   # Sanity check: $1M monthly gross is extreme
        },
    },
    "calculate_department_payroll": {
        # No numeric parameters to bound, but we keep the entry
        # as documentation and for future extension
    },
}


# =============================================================================
# Operation Risk Mapping — Non-financial risk classification
# =============================================================================
# CONCEPT: Operation-Based Risk
#
# Some operations are risky regardless of dollar amount. Deleting an
# employee record, for example, is always high-risk. This mapping lets
# us classify operations by their type, not just their financial impact.
# =============================================================================
OPERATION_RISK_MAP: dict[str, str] = {
    # Read operations — generally low risk
    "query_employee": "low",
    "view_payslip": "low",
    "check_leave_balance": "low",
    "search_employees": "low",

    # Calculation operations — medium risk (they produce financial data)
    "calculate_pay": "medium",
    "calculate_deductions": "medium",

    # Batch operations — high risk (affect multiple employees)
    "run_department_payroll": "high",
    "bulk_salary_update": "high",

    # Modification operations — critical risk (change state)
    "process_payroll": "critical",
    "approve_payroll": "critical",
    "modify_salary": "critical",
    "terminate_employee": "critical",
}


class ToolGovernor:
    """
    Governs which tools each agent type can access and validates parameters
    before tool execution.

    ARCHITECTURE:
    The ToolGovernor sits between the agent's "decide which tool to call"
    step and the actual tool execution:

        Agent Decision → [ToolGovernor] → Tool Execution
                              |
                         Check Access ──→ BLOCKED (if not in allowlist)
                         Check Params ──→ VIOLATION (if out of bounds)
                         Classify Risk ──→ APPROVAL NEEDED (if high/critical)

    USAGE IN LANGGRAPH:
    In the agent's tool execution node, BEFORE calling the tool:

        governor = ToolGovernor()

        # Step 1: Is this tool allowed for this agent?
        if not governor.check_tool_access(agent_type="employee", tool_name="process_payroll"):
            return {"error": "Tool access denied"}

        # Step 2: Are the parameters within bounds?
        violations = governor.check_parameter_bounds("calculate_deductions", {"gross_pay": 5000})
        if violations:
            return {"error": violations}

        # Step 3: What risk level is this operation?
        risk = governor.classify_risk("calculate_pay", amount=75000)
        if risk in ("high", "critical"):
            # Trigger human-in-the-loop approval
            ...
    """

    def __init__(
        self,
        tool_allowlist: dict[str, set[str]] | None = None,
        parameter_bounds: dict[str, dict[str, dict[str, float]]] | None = None,
        financial_thresholds: dict[str, float] | None = None,
        operation_risk_map: dict[str, str] | None = None,
    ):
        """
        Initialize the ToolGovernor with configurable policies.

        Args:
            tool_allowlist: Map of agent_type -> set of allowed tool names.
                            Defaults to the module-level TOOL_ALLOWLIST.
            parameter_bounds: Map of tool_name -> parameter constraints.
                              Defaults to the module-level PARAMETER_BOUNDS.
            financial_thresholds: Risk level dollar thresholds.
                                 Defaults to the module-level FINANCIAL_THRESHOLDS.
            operation_risk_map: Map of operation_name -> risk_level.
                                Defaults to the module-level OPERATION_RISK_MAP.
        """
        self.tool_allowlist = tool_allowlist or TOOL_ALLOWLIST
        self.parameter_bounds = parameter_bounds or PARAMETER_BOUNDS
        self.financial_thresholds = financial_thresholds or FINANCIAL_THRESHOLDS
        self.operation_risk_map = operation_risk_map or OPERATION_RISK_MAP

    def check_tool_access(self, agent_type: str, tool_name: str) -> bool:
        """
        Check if a specific agent type is allowed to call a specific tool.

        HOW IT WORKS:
          1. Look up the agent type in the allowlist
          2. If the agent type is unknown, DENY (fail-closed)
          3. If the tool is in the allowed set, ALLOW
          4. Otherwise, DENY

        SECURITY PRINCIPLE: Fail-Closed
        If we don't recognize the agent type, we deny access.
        This is safer than fail-open (allowing unknown agents).

        Args:
            agent_type: The type of agent making the request
                        (e.g., "payroll", "employee", "compliance").
            tool_name: The name of the tool the agent wants to call
                       (e.g., "calculate_net_pay", "get_employee_info").

        Returns:
            True if the agent is allowed to use this tool, False otherwise.
        """
        allowed_tools = self.tool_allowlist.get(agent_type)

        # Unknown agent type — deny by default (fail-closed)
        if allowed_tools is None:
            logger.warning(
                "Tool access DENIED: unknown agent type '%s' attempted to call '%s'",
                agent_type,
                tool_name,
            )
            return False

        is_allowed = tool_name in allowed_tools

        if not is_allowed:
            logger.warning(
                "Tool access DENIED: agent type '%s' is not allowed to call '%s'. "
                "Allowed tools: %s",
                agent_type,
                tool_name,
                allowed_tools,
            )

        return is_allowed

    def check_parameter_bounds(
        self, tool_name: str, params: dict
    ) -> list[str]:
        """
        Validate that tool parameters are within acceptable bounds.

        HOW IT WORKS:
          1. Look up the tool's parameter constraints
          2. For each constrained parameter, check if the value is within [min, max]
          3. Return a list of violations (empty = all good)

        WHY VALIDATE PARAMETERS?
        The LLM generates tool arguments. It could hallucinate extreme values
        like a $999,999,999 salary or a -$50,000 deduction. Parameter bounds
        catch these before the tool executes with nonsensical inputs.

        Args:
            tool_name: The name of the tool being called.
            params: The parameters the agent wants to pass to the tool.

        Returns:
            A list of violation messages. Empty list means all parameters
            are within bounds.
        """
        violations: list[str] = []
        bounds = self.parameter_bounds.get(tool_name)

        # No bounds defined for this tool — allow all parameters
        if not bounds:
            return violations

        for param_name, constraints in bounds.items():
            if param_name not in params:
                # Parameter not provided — skip (the tool itself will handle
                # missing required params)
                continue

            value = params[param_name]

            # Only validate numeric parameters
            if not isinstance(value, (int, float)):
                continue

            min_val = constraints.get("min")
            max_val = constraints.get("max")

            if min_val is not None and value < min_val:
                violations.append(
                    f"Parameter '{param_name}' value {value} is below "
                    f"minimum allowed value {min_val} for tool '{tool_name}'"
                )

            if max_val is not None and value > max_val:
                violations.append(
                    f"Parameter '{param_name}' value {value} exceeds "
                    f"maximum allowed value {max_val} for tool '{tool_name}'"
                )

        if violations:
            logger.warning(
                "Parameter bounds violations for tool '%s': %s",
                tool_name,
                violations,
            )

        return violations

    def classify_risk(self, operation: str, amount: float = 0.0) -> str:
        """
        Classify the risk level of an operation based on its type and
        financial amount.

        HOW IT WORKS:
          The risk level is the HIGHER of:
            1. Operation-based risk (from OPERATION_RISK_MAP)
            2. Financial-based risk (from dollar amount thresholds)

          For example:
            - "query_employee" with $0 → low (operation=low, financial=low)
            - "calculate_pay" with $75,000 → high (operation=medium, financial=high)
            - "process_payroll" with $5,000 → critical (operation=critical, financial=low)

          We take the MAX because risk should never be LOWERED by one factor
          when the other factor indicates danger.

        RISK LEVELS AND THEIR IMPLICATIONS:
          - "low"      → Proceed automatically, log for audit trail
          - "medium"   → Proceed automatically, flag in dashboard
          - "high"     → Pause and request manager approval
          - "critical" → Pause and request admin approval

        Args:
            operation: The type of operation being performed (e.g., "calculate_pay",
                       "process_payroll"). Must match keys in OPERATION_RISK_MAP.
            amount: The financial amount involved. Use 0 for non-financial operations.

        Returns:
            Risk level string: "low", "medium", "high", or "critical".
        """
        # Define risk level ordering for comparison
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        # --- Operation-based risk ---
        operation_risk = self.operation_risk_map.get(operation, "low")

        # --- Financial-based risk ---
        if amount > self.financial_thresholds["critical"]:
            financial_risk = "critical"
        elif amount > self.financial_thresholds["high"]:
            financial_risk = "high"
        elif amount > self.financial_thresholds["medium"]:
            financial_risk = "medium"
        else:
            financial_risk = "low"

        # Take the higher of the two risk levels
        if risk_order.get(financial_risk, 0) > risk_order.get(operation_risk, 0):
            final_risk = financial_risk
        else:
            final_risk = operation_risk

        logger.info(
            "Risk classification for '%s' (amount=$%.2f): "
            "operation_risk=%s, financial_risk=%s, final=%s",
            operation,
            amount,
            operation_risk,
            financial_risk,
            final_risk,
        )

        return final_risk

    def get_allowed_tools(self, agent_type: str) -> set[str]:
        """
        Return the set of tools an agent type is allowed to use.

        Useful for passing the tool list to the LLM — only bind tools
        the agent is actually allowed to call.

        Args:
            agent_type: The type of agent (e.g., "payroll", "employee").

        Returns:
            A set of tool name strings. Empty set if the agent type is unknown.
        """
        return self.tool_allowlist.get(agent_type, set())
