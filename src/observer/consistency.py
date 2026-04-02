"""Belief conflict detection for the Silent Observer.

v0.1: stub returning empty list.
v0.5: real conflict detection against action history and scene state.
The observer.py assess_action method handles THOR-specific consistency.
This module provides the generic interface for non-THOR contexts.
"""


def check_consistency(proposed_action: str, world_state: dict, belief_state: dict) -> list:
    """Check for conflicts between proposed action and current beliefs.

    For THOR environments, use SilentObserver.assess_action() instead,
    which has access to typed state objects and action history.

    For non-THOR (v0.1 eval tasks), returns empty list when world_state
    is empty, and performs generic dict-based checks otherwise.

    Args:
        proposed_action: Description of the action being proposed.
        world_state: Current world state as a dictionary.
        belief_state: Current belief state as a dictionary.

    Returns:
        A list of conflict descriptions. Empty when world_state is empty.
    """
    # Non-empty world_state means we have real state to check
    if not world_state:
        return []

    conflicts: list[str] = []

    # Generic checks that work with dict-based world state
    if "held_object" in world_state and world_state["held_object"]:
        if "pick" in proposed_action.lower():
            conflicts.append("Already holding an object")

    return conflicts
