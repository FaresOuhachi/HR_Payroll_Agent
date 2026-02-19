"""
Run LangSmith Evaluation
=============================================================================
Runs all evaluators against the hr-payroll-eval dataset and uploads
results to LangSmith Experiments.

Prerequisites:
  1. Create the dataset first: python -m scripts.create_eval_dataset
  2. Ensure Docker services are running: docker-compose up -d
  3. Ensure LANGSMITH_* env vars are set in .env

Run: python -m scripts.run_evaluation
=============================================================================
"""

import asyncio

import src.config  # noqa: F401 — triggers os.environ export for LangSmith
from langsmith.evaluation import evaluate

from src.evaluation.evaluators import correct_routing, correct_tools, response_quality
from src.agents.router_agent import route_and_execute


def agent_target(inputs: dict) -> dict:
    """
    Wraps route_and_execute() for use with langsmith.evaluate().

    evaluate() calls this function for each dataset example.
    It passes the example inputs and expects a dict of outputs.
    """
    result = asyncio.run(route_and_execute(user_input=inputs["input"]))
    return {
        "response": result["response"],
        "agent_type": result["agent_type"],
        "tools_used": result["tools_used"],
    }


def main():
    print("Running evaluation against 'hr-payroll-eval' dataset...")
    print("Results will appear in LangSmith Experiments tab.\n")

    results = evaluate(
        agent_target,
        data="hr-payroll-eval",
        evaluators=[correct_routing, correct_tools, response_quality],
        experiment_prefix="hr-payroll",
        metadata={"version": "1.0", "model": "llama-3.1-8b-instant"},
        max_concurrency=2,
    )

    print("\n=== Evaluation Complete ===")
    print("View results at: https://eu.smith.langchain.com")


if __name__ == "__main__":
    main()
