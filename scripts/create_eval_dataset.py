"""
Create LangSmith Evaluation Dataset
=============================================================================
Creates a dataset of HR payroll test cases in LangSmith for evaluating
the agent's routing accuracy, tool selection, and response quality.

Run: python -m scripts.create_eval_dataset
=============================================================================
"""

import src.config  # noqa: F401 — triggers os.environ export for LangSmith
from langsmith import Client

DATASET_NAME = "hr-payroll-eval"
DATASET_DESCRIPTION = "Evaluation dataset for the HR Payroll Agent — covers routing, tool selection, and response quality."

# Each example has:
#   inputs:  what the user sends
#   outputs: what we expect (used by evaluators to score the agent)
EXAMPLES = [
    # --- Payroll ---
    {
        "inputs": {"input": "Calculate net pay for EMP001"},
        "outputs": {
            "expected_agent": "payroll",
            "expected_tools": ["calculate_net_pay"],
        },
    },
    {
        "inputs": {"input": "What is the gross pay for EMP002?"},
        "outputs": {
            "expected_agent": "payroll",
            "expected_tools": ["calculate_gross_pay"],
        },
    },
    {
        "inputs": {"input": "Show me the total payroll cost for the Engineering department"},
        "outputs": {
            "expected_agent": "payroll",
            "expected_tools": ["calculate_department_payroll"],
        },
    },
    {
        "inputs": {"input": "What are the deductions for EMP004?"},
        "outputs": {
            "expected_agent": "payroll",
            "expected_tools": ["calculate_deductions"],
        },
    },

    # --- Employee ---
    {
        "inputs": {"input": "How many PTO days does Amina have left?"},
        "outputs": {
            "expected_agent": "employee",
            "expected_tools": ["get_leave_balance"],
        },
    },
    {
        "inputs": {"input": "Get employee info for EMP003"},
        "outputs": {
            "expected_agent": "employee",
            "expected_tools": ["get_employee_info"],
        },
    },
    {
        "inputs": {"input": "Who works in the Finance department?"},
        "outputs": {
            "expected_agent": "employee",
            "expected_tools": ["search_employees_by_department"],
        },
    },

    # --- Compliance ---
    {
        "inputs": {"input": "What is the maximum overtime allowed per month?"},
        "outputs": {
            "expected_agent": "compliance",
            "expected_tools": [],
        },
    },
    {
        "inputs": {"input": "What are the minimum wage requirements?"},
        "outputs": {
            "expected_agent": "compliance",
            "expected_tools": [],
        },
    },

    # --- General ---
    {
        "inputs": {"input": "Hello, what can you do?"},
        "outputs": {
            "expected_agent": "general",
            "expected_tools": [],
        },
    },
    {
        "inputs": {"input": "Thanks for your help!"},
        "outputs": {
            "expected_agent": "general",
            "expected_tools": [],
        },
    },
]


def main():
    client = Client()

    # Delete existing dataset if it exists (for idempotent re-runs)
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset '{DATASET_NAME}'")
    except Exception:
        pass

    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"Created dataset '{DATASET_NAME}' (id: {dataset.id})")

    # Add examples
    client.create_examples(
        inputs=[ex["inputs"] for ex in EXAMPLES],
        outputs=[ex["outputs"] for ex in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Added {len(EXAMPLES)} examples to dataset")


if __name__ == "__main__":
    main()
