"""
LangSmith Evaluators for the HR Payroll Agent
=============================================================================
Three evaluators that score agent runs:

1. correct_routing — Did the router pick the right specialist agent?
2. correct_tools  — Did the agent call the expected tools?
3. response_quality — Is the response helpful and well-formatted? (LLM-as-judge)

These are used with `langsmith.evaluation.evaluate()` in scripts/run_evaluation.py.
=============================================================================
"""

from groq import Groq
from src.config import settings


def correct_routing(outputs: dict, reference_outputs: dict) -> dict:
    """
    Check if the router classified the user's intent correctly.

    Returns score 1.0 if agent_type matches expected_agent, else 0.0.
    """
    expected = reference_outputs.get("expected_agent", "")
    actual = outputs.get("agent_type", "")
    match = expected == actual

    return {
        "key": "correct_routing",
        "score": 1.0 if match else 0.0,
        "comment": f"Expected '{expected}', got '{actual}'",
    }


def correct_tools(outputs: dict, reference_outputs: dict) -> dict:
    """
    Check if the agent used the expected tools.

    Scores based on set overlap between expected and actual tools:
    - 1.0 if exact match (or both empty)
    - Partial credit for partial overlap
    - 0.0 if no overlap
    """
    expected = set(reference_outputs.get("expected_tools", []))
    actual = set(outputs.get("tools_used", []))

    if not expected and not actual:
        return {"key": "correct_tools", "score": 1.0, "comment": "No tools expected or used"}

    if not expected or not actual:
        return {
            "key": "correct_tools",
            "score": 0.0,
            "comment": f"Expected {expected or 'none'}, got {actual or 'none'}",
        }

    intersection = expected & actual
    # F1-style score: harmonic mean of precision and recall
    precision = len(intersection) / len(actual) if actual else 0
    recall = len(intersection) / len(expected) if expected else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "key": "correct_tools",
        "score": f1,
        "comment": f"Expected {expected}, got {actual} (F1={f1:.2f})",
    }


def response_quality(inputs: dict, outputs: dict) -> dict:
    """
    LLM-as-judge evaluator: scores the response on helpfulness and accuracy.

    Uses Groq (fast inference) to rate the agent's response on a 1-5 scale,
    then normalizes to 0.0-1.0.
    """
    user_input = inputs.get("input", "")
    agent_response = outputs.get("response", "")

    judge_prompt = f"""Rate the following AI assistant response on a scale of 1-5.

USER QUESTION: {user_input}

ASSISTANT RESPONSE: {agent_response}

CRITERIA:
- Helpfulness: Does it answer the question?
- Accuracy: Are numbers and facts correct?
- Clarity: Is the response clear and well-formatted?
- Completeness: Does it cover all aspects of the question?

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<brief explanation>"}}"""

    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
        )

        import json
        result = json.loads(response.choices[0].message.content)
        raw_score = float(result.get("score", 3))
        normalized = (raw_score - 1) / 4  # Map 1-5 to 0.0-1.0

        return {
            "key": "response_quality",
            "score": normalized,
            "comment": result.get("reason", ""),
        }
    except Exception as e:
        return {
            "key": "response_quality",
            "score": 0.5,
            "comment": f"Judge failed: {e}",
        }
