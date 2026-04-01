"""Belief conflict detection. Stub for v0.1, returns empty list. Full implementation in v0.2."""


def check_consistency(proposed_action: str, world_state: dict, belief_state: dict) -> list:
    """Check for conflicts between proposed action and current beliefs.

    Stub: always returns no conflicts. In v0.2 this will compare the proposed
    action against the agent's belief graph, detect contradictions, and flag
    actions that rely on unverified or stale beliefs.

    Args:
        proposed_action: Description of the action being proposed.
        world_state: Current world state as a dictionary.
        belief_state: Current belief state as a dictionary.

    Returns:
        A list of conflict descriptions. Empty in v0.1.
    """
    return []
