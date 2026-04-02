"""LLM-based action planner for AI2-THOR environments.

The planner generates candidate actions but does NOT execute them.
All actions are proposals that the observer evaluates before execution.
This separation between proposal and execution is the core architectural
principle of Project Witness.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.thor.schemas import (
    ActionProposal,
    TaskDefinition,
    THORActionResult,
    THORState,
)

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# THOR actions available to the planner. Navigation actions have no target;
# object actions require an objectId parameter.
NAVIGATION_ACTIONS: list[str] = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Stand",
    "Crouch",
]

OBJECT_ACTIONS: list[str] = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "DropHandObject",
]

# Safe fallback action when the planner fails to produce valid output.
FALLBACK_ACTION = "LookDown"


class ActionPlanner:
    """LLM-based planner that proposes actions for THOR environments.

    The planner generates candidate actions but does NOT execute them.
    All actions are proposals that the observer evaluates before execution.
    """

    def __init__(self, model: BaseModel) -> None:
        """Initialize with a model wrapper for planning.

        Args:
            model: The LLM to use for planning (Gemini Flash or Qwen).
        """
        self._model = model

    async def propose_action(
        self,
        task: TaskDefinition,
        state: THORState,
        action_history: list[THORActionResult],
    ) -> ActionProposal:
        """Propose the next action for the given task and state.

        The planner receives the current state and history, then generates
        a structured action proposal with predicted consequences.

        Args:
            task: The task being worked on.
            state: Current THOR scene state.
            action_history: Previous actions and their results.

        Returns:
            An ActionProposal with action, target, predictions, and confidence.
        """
        prompt = self._build_planning_prompt(task, state, action_history)

        try:
            response_text, confidence = await self._model.query(prompt)
            return self._parse_proposal(response_text, confidence)
        except Exception as exc:
            logger.warning(
                "Planner query failed: %s. Returning safe fallback action.", exc
            )
            return ActionProposal(
                action=FALLBACK_ACTION,
                target_object=None,
                predicted_state_changes=["agent_looks_down"],
                planner_confidence=0.1,
                reasoning=f"Fallback action due to planner error: {exc}",
            )

    def _build_planning_prompt(
        self,
        task: TaskDefinition,
        state: THORState,
        action_history: list[THORActionResult],
    ) -> str:
        """Build the LLM prompt for action planning.

        Includes the task description, current visible objects, what the
        agent is holding, recent action history, and available actions.
        Requests a JSON response matching the ActionProposal schema.

        Args:
            task: The task being worked on.
            state: Current THOR scene state.
            action_history: Previous actions and their results.

        Returns:
            A formatted prompt string for the LLM.
        """
        # Summarize visible objects
        visible_objects = [
            obj for obj in state.objects if obj.visible
        ]
        object_summaries: list[str] = []
        for obj in visible_objects[:20]:  # Limit to avoid prompt overflow
            props: list[str] = []
            if obj.is_pickupable:
                props.append("pickupable")
            if obj.is_openable:
                props.append("openable")
            if obj.is_toggleable:
                props.append("toggleable")
            prop_str = f" [{', '.join(props)}]" if props else ""
            object_summaries.append(f"  - {obj.object_id} ({obj.object_type}){prop_str}")

        objects_text = (
            "\n".join(object_summaries) if object_summaries
            else "  (no visible objects)"
        )

        # Summarize recent action history
        history_lines: list[str] = []
        for result in action_history[-5:]:  # Last 5 actions
            status = "OK" if result.success else f"FAILED: {result.error_message}"
            target_str = f" -> {result.target_object}" if result.target_object else ""
            line = f"  Step {result.step_number}: {result.action}"
            line += f"{target_str} [{status}]"
            history_lines.append(line)

        history_text = "\n".join(history_lines) if history_lines else "  (no actions taken yet)"

        # Held object
        held_text = state.held_object if state.held_object else "nothing"

        # Available actions
        nav_actions = ", ".join(NAVIGATION_ACTIONS)
        obj_actions = ", ".join(OBJECT_ACTIONS)

        pos = state.agent_position
        rot = state.agent_rotation
        pos_str = f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})"
        rot_str = f"({rot['x']:.1f}, {rot['y']:.1f}, {rot['z']:.1f})"
        remaining = task.max_steps - state.step_number

        prompt = f"""You are an action planner for a household robot.

TASK: {task.description}
Goal: Find a {task.target_object_type} and place it on a {task.target_receptacle_type}.
Scene: {task.scene_name}
Max steps remaining: {remaining}

CURRENT STATE:
- Agent position: {pos_str}
- Agent rotation: {rot_str}
- Holding: {held_text}

VISIBLE OBJECTS:
{objects_text}

RECENT ACTIONS:
{history_text}

AVAILABLE ACTIONS:
Navigation (no target needed): {nav_actions}
Object interaction (requires objectId): {obj_actions}

For PutObject, the agent places the held object on/in the nearest valid receptacle.

Respond with ONLY a JSON object in this exact format:
{{
  "action": "<action_name>",
  "target_object": "<objectId or null>",
  "predicted_state_changes": ["<change1>", "<change2>"],
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<brief explanation of why this action>"
}}

Choose the single best next action. Be specific about predicted state changes."""

        return prompt

    def _parse_proposal(self, response: str, confidence: float) -> ActionProposal:
        """Parse the LLM response into an ActionProposal.

        Extracts JSON from the response, validates it against the schema,
        and falls back to a safe action (LookDown) on parse failure.

        Args:
            response: Raw text response from the LLM.
            confidence: Confidence score from the model query.

        Returns:
            A validated ActionProposal.
        """
        # Try to extract JSON from the response
        try:
            # Handle responses that wrap JSON in markdown code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Strip code fences
                lines = cleaned.split("\n")
                json_lines: list[str] = []
                inside_fence = False
                for line in lines:
                    if line.strip().startswith("```") and not inside_fence:
                        inside_fence = True
                        continue
                    if line.strip().startswith("```") and inside_fence:
                        break
                    if inside_fence:
                        json_lines.append(line)
                cleaned = "\n".join(json_lines)

            parsed = json.loads(cleaned)

            action = parsed.get("action", FALLBACK_ACTION)
            target = parsed.get("target_object")
            changes = parsed.get("predicted_state_changes", [])
            reasoning = parsed.get("reasoning", "No reasoning provided.")

            # Validate action name is known
            all_actions = NAVIGATION_ACTIONS + OBJECT_ACTIONS
            if action not in all_actions:
                logger.warning(
                    "Unknown action '%s' from planner, falling back to %s.",
                    action,
                    FALLBACK_ACTION,
                )
                action = FALLBACK_ACTION
                target = None

            # Prefer confidence from the JSON if present
            json_confidence = parsed.get("confidence")
            if isinstance(json_confidence, (int, float)):
                final_confidence = max(0.0, min(1.0, float(json_confidence)))
            else:
                final_confidence = confidence

            return ActionProposal(
                action=action,
                target_object=target if target != "null" else None,
                predicted_state_changes=changes if isinstance(changes, list) else [],
                planner_confidence=final_confidence,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse planner response: %s. Falling back to safe action.",
                exc,
            )
            return ActionProposal(
                action=FALLBACK_ACTION,
                target_object=None,
                predicted_state_changes=["agent_looks_down"],
                planner_confidence=0.1,
                reasoning=f"Parse failure, safe fallback. Raw response: {response[:200]}",
            )
