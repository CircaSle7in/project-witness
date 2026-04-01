"""LLM-as-judge scoring and local fallback scorers.

Provides multiple scoring strategies: LLM judge, exact match, fuzzy
embedding similarity, and multi-match for checking multiple expected items.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BaseModel
    from src.pipeline.schemas import EvalTask


async def llm_judge(
    model: BaseModel,
    prompt: str,
    response: str,
    expected: str,
) -> tuple[float, str]:
    """Use a model's judge() method to score a response.

    Args:
        model: A model wrapper that implements the judge interface.
        prompt: The original task prompt.
        response: The model's response to evaluate.
        expected: The expected/reference answer.

    Returns:
        A tuple of (score 0-1, explanation string).
    """
    return await model.judge(prompt, response, expected)


def exact_match(response: str, expected: str) -> float:
    """Case-insensitive exact string match.

    Args:
        response: The model's response.
        expected: The expected answer.

    Returns:
        1.0 if the normalized strings match, 0.0 otherwise.
    """
    return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0


def fuzzy_match(
    response: str,
    expected: str,
    threshold: float = 0.8,
) -> float:
    """Compute cosine similarity between response and expected using sentence embeddings.

    Uses sentence-transformers for embedding computation. Returns the raw
    cosine similarity as the score if it meets the threshold, 0.0 otherwise.

    Args:
        response: The model's response.
        expected: The expected answer.
        threshold: Minimum cosine similarity to count as a match.

    Returns:
        The cosine similarity if above threshold, otherwise 0.0.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([response, expected])
        # Cosine similarity
        import numpy as np

        a = embeddings[0]
        b = embeddings[1]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        return similarity if similarity >= threshold else 0.0
    except ImportError:
        # Fallback: simple token overlap if sentence-transformers unavailable
        resp_tokens = set(response.lower().split())
        exp_tokens = set(expected.lower().split())
        if not exp_tokens:
            return 0.0
        overlap = len(resp_tokens & exp_tokens) / len(exp_tokens)
        return overlap if overlap >= threshold else 0.0


def multi_match(response: str, expected_items: list[str]) -> float:
    """Check what fraction of expected items appear in the response.

    Performs case-insensitive substring matching for each expected item.

    Args:
        response: The model's response.
        expected_items: List of expected items that should appear.

    Returns:
        The fraction of expected items found in the response (0.0 to 1.0).
    """
    if not expected_items:
        return 1.0

    response_lower = response.lower()
    found = sum(1 for item in expected_items if item.lower() in response_lower)
    return found / len(expected_items)


async def score_task(
    task: EvalTask,
    response: str,
    model: BaseModel,
) -> tuple[float, str]:
    """Dispatch to the correct scorer based on the task's scoring method.

    Args:
        task: The evaluation task with scoring method specified.
        response: The model's response to score.
        model: A model wrapper for LLM judge scoring.

    Returns:
        A tuple of (score 0-1, explanation string).
    """
    if task.scoring == "exact":
        score = exact_match(response, task.expected)
        explanation = "Exact match." if score == 1.0 else "No exact match."
        return score, explanation

    if task.scoring == "fuzzy":
        score = fuzzy_match(response, task.expected)
        explanation = (
            f"Fuzzy match score: {score:.3f}."
            if score > 0.0
            else "Below fuzzy match threshold."
        )
        return score, explanation

    if task.scoring == "multi_match":
        items = task.expected_actions or []
        if task.expected_risks:
            items = items + task.expected_risks
        # Fall back to splitting expected on commas if no lists provided
        if not items:
            items = [s.strip() for s in task.expected.split(",") if s.strip()]
        score = multi_match(response, items)
        explanation = f"Matched {score * len(items):.0f}/{len(items)} expected items."
        return score, explanation

    if task.scoring == "llm_judge":
        return await llm_judge(model, task.prompt, response, task.expected)

    # Unknown scoring method: fall back to fuzzy
    score = fuzzy_match(response, task.expected)
    return score, f"Fallback fuzzy match (unknown scoring: {task.scoring})."
