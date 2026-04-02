"""Execution gate logic for the Silent Observer.

Provides reversibility scoring, dynamic threshold computation, and the
gate decision function that determines whether to ACT, ASK_HUMAN, WAIT,
GATHER_EVIDENCE, or REFUSE.
"""

from __future__ import annotations

from src.pipeline.schemas import GateDecision, PrincipleFlag

# Reversibility lookup table: how reversible is an action keyword?
# 1.0 = fully reversible / safe to retry, 0.0 = irreversible
REVERSIBILITY_TABLE: dict[str, float] = {
    "look": 1.0,
    "observe": 1.0,
    "query": 1.0,
    "read": 1.0,
    "check": 1.0,
    "inspect": 1.0,
    "describe": 1.0,
    "move": 0.7,
    "pick": 0.7,
    "open": 0.7,
    "place": 0.7,
    "push": 0.7,
    "pull": 0.7,
    "pour": 0.3,
    "mix": 0.3,
    "cut": 0.3,
    "send": 0.3,
    "write": 0.3,
    "submit": 0.3,
    "delete": 0.0,
    "break": 0.0,
    "drop": 0.0,
    "destroy": 0.0,
    "crush": 0.0,
    "erase": 0.0,
}

DEFAULT_REVERSIBILITY = 0.5
BASE_THRESHOLD = 0.7


def score_reversibility(action: str) -> float:
    """Score the reversibility of a proposed action based on keyword matching.

    Scans the action string for known keywords and returns the lowest
    reversibility score found (most dangerous keyword wins).

    Args:
        action: A string describing the proposed action.

    Returns:
        A float between 0.0 (irreversible) and 1.0 (fully reversible).
    """
    action_lower = action.lower()
    scores: list[float] = []

    for keyword, rev_score in REVERSIBILITY_TABLE.items():
        if keyword in action_lower:
            scores.append(rev_score)

    if not scores:
        return DEFAULT_REVERSIBILITY

    # Use the minimum (most conservative) score found
    return min(scores)


def compute_threshold(
    reversibility: float,
    principle_flags: list[PrincipleFlag],
) -> float:
    """Compute the dynamic confidence threshold for gate decisions.

    Starts from a base threshold of 0.7, then adjusts based on how
    reversible the action is and whether any principle flags are raised.

    Lower reversibility raises the threshold (require more confidence).
    Hard principle flags raise the threshold significantly.

    Args:
        reversibility: The reversibility score of the proposed action (0-1).
        principle_flags: List of principle flags raised by the checker.

    Returns:
        The confidence threshold needed to approve the action.
    """
    threshold = BASE_THRESHOLD

    # Less reversible actions need higher confidence
    # reversibility=1.0 -> no adjustment, reversibility=0.0 -> +0.2
    reversibility_modifier = (1.0 - reversibility) * 0.2
    threshold += reversibility_modifier

    # Principle flags raise the bar
    for flag in principle_flags:
        if flag.severity == "hard":
            threshold += 0.15
        elif flag.severity == "soft":
            threshold += 0.05

    return min(threshold, 0.99)


def decide_gate(
    calibrated_confidence: float,
    reversibility: float,
    principle_flags: list[PrincipleFlag],
    conflicts: list,
) -> GateDecision:
    """Make the gate decision based on confidence, reversibility, and flags.

    Decision priority (highest to lowest):
    1. Hard principle flags -> REFUSE unconditionally
    2. Belief conflicts -> GATHER_EVIDENCE
    3. Confidence above threshold -> ACT
    4. Low confidence + reversible -> WAIT
    5. Low confidence + irreversible -> ASK_HUMAN

    Args:
        calibrated_confidence: The calibrated confidence score (0-1).
        reversibility: The reversibility score of the proposed action (0-1).
        principle_flags: List of principle flags from the values checker.
        conflicts: List of belief conflicts from the consistency checker.

    Returns:
        One of the five GateDecision values.
    """
    # Hard principle violations ALWAYS refuse, regardless of confidence
    hard_flags = [f for f in principle_flags if f.severity == "hard"]
    if hard_flags:
        return GateDecision.REFUSE

    # Belief conflicts require more evidence before acting
    if conflicts:
        return GateDecision.GATHER_EVIDENCE

    # Confidence above dynamic threshold: approve the action
    threshold = compute_threshold(reversibility, principle_flags)
    if calibrated_confidence >= threshold:
        return GateDecision.ACT

    # Low confidence on a reversible action: gather more evidence rather than
    # freezing. GATHER_EVIDENCE triggers a safe observation action (look/rotate)
    # which lets the agent rebuild prediction trust through successful observations.
    # WAIT would deadlock because no actions means no new trust signal.
    if reversibility >= 0.7:
        return GateDecision.GATHER_EVIDENCE
    else:
        return GateDecision.ASK_HUMAN
