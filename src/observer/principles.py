"""Values/constitution checker. Stub for v0.1. Full implementation in v0.2."""

from src.pipeline.schemas import PrincipleFlag


def check_principles(proposed_action: str, world_state: dict) -> list[PrincipleFlag]:
    """Check proposed action against guiding principles.

    Stub: always returns no flags. In v0.2 this will load principles from
    configs/principles.yaml and evaluate the proposed action against each
    principle, returning typed PrincipleFlag objects for any violations.

    Args:
        proposed_action: Description of the action being proposed.
        world_state: Current world state as a dictionary.

    Returns:
        A list of PrincipleFlag objects. Empty in v0.1.
    """
    return []
