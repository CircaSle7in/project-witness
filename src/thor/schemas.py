"""Pydantic v2 models for AI2-THOR integration.

Defines data shapes for THOR objects, scene state, action proposals,
state diffs, action results, and task definitions. These schemas form
the contract between the THOR controller, the LLM planner, and the
Silent Observer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class THORObject(BaseModel):
    """An object in an AI2-THOR scene."""

    object_id: str
    object_type: str
    position: dict  # {x, y, z}
    rotation: dict  # {x, y, z}
    is_pickupable: bool = False
    is_openable: bool = False
    is_toggleable: bool = False
    is_breakable: bool = False
    is_picked_up: bool = False
    parent_receptacles: list[str] = Field(default_factory=list)
    visible: bool = True


class THORState(BaseModel):
    """Full scene state snapshot from AI2-THOR."""

    scene_name: str
    step_number: int
    agent_position: dict  # {x, y, z}
    agent_rotation: dict  # {x, y, z}
    objects: list[THORObject]
    held_object: str | None = None
    last_action: str | None = None
    last_action_success: bool = True


class ActionProposal(BaseModel):
    """A proposed action from the LLM planner.

    The planner generates these proposals; they are NOT executed directly.
    The observer evaluates each proposal before the controller acts.
    """

    action: str  # THOR action name
    target_object: str | None = None
    predicted_state_changes: list[str]  # What the planner thinks will happen
    planner_confidence: float  # 0-1
    reasoning: str


class StateDelta(BaseModel):
    """Difference between predicted and observed state after an action.

    Tracks how well the planner's predictions matched reality. A high
    match_ratio means the planner has a good model of the environment;
    a low ratio signals the observer should reduce trust.
    """

    predicted_changes: list[str]
    observed_changes: list[str]
    matches: list[str]  # Predictions that matched observed changes
    mismatches: list[str]  # Predictions that did not match
    unexpected: list[str]  # Observed changes that were not predicted
    match_ratio: float  # len(matches) / max(len(predicted), 1)


class THORActionResult(BaseModel):
    """Result of executing an action in THOR.

    Bundles the before/after state, the computed delta, and optional
    observer assessment into a single audit record.
    """

    action: str
    target_object: str | None = None
    success: bool
    state_before: THORState
    state_after: THORState
    state_delta: StateDelta
    observer_assessment: Any | None = None  # ObserverAssessment; avoid circular import
    step_number: int
    error_message: str | None = None


class TaskDefinition(BaseModel):
    """A rearrangement task: move an object to a target location.

    Each task specifies a scene, an object to find, and a receptacle
    to place it on. Difficulty varies by navigation distance, object
    visibility, and whether the object is inside a closed container.
    """

    task_id: str
    scene_name: str
    target_object_type: str  # e.g. "Mug"
    target_receptacle_type: str  # e.g. "DiningTable"
    max_steps: int = 30
    description: str
