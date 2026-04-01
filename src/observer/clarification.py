"""Ask-before-act policy. Stub for v0.1. Full implementation in v0.2."""


def should_clarify(proposed_action: str, confidence: float) -> bool:
    """Determine if clarification should be requested before acting.

    Stub: always returns False. In v0.2 this will evaluate whether the action
    is ambiguous, whether confidence is below a dynamic threshold, and whether
    the user's intent is clear enough to proceed.

    Args:
        proposed_action: Description of the action being proposed.
        confidence: The calibrated confidence score for the proposed action.

    Returns:
        True if clarification is needed, False otherwise. Always False in v0.1.
    """
    return False
