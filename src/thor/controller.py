"""Thin wrapper around the AI2-THOR Controller for Project Witness.

Provides state extraction, action execution, and state diffing. Handles
the case where AI2-THOR cannot launch gracefully, returning clear errors
instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any

from src.thor.schemas import (
    StateDelta,
    THORActionResult,
    THORObject,
    THORState,
)

logger = logging.getLogger(__name__)


class THORLaunchError(Exception):
    """Raised when the AI2-THOR controller cannot start."""


class WitnessController:
    """Thin wrapper around AI2-THOR Controller for Project Witness.

    Provides state extraction, action execution, and state diffing.
    All actions produce a full audit trail with before/after state
    snapshots and a delta comparing predicted vs observed changes.
    """

    # Reversibility scores for THOR actions. More precise than the v0.1
    # keyword-based lookup because THOR knows exactly which actions are
    # reversible and which destroy state.
    ACTION_REVERSIBILITY: dict[str, float] = {
        "MoveAhead": 1.0,
        "MoveBack": 1.0,
        "MoveLeft": 1.0,
        "MoveRight": 1.0,
        "RotateLeft": 1.0,
        "RotateRight": 1.0,
        "LookUp": 1.0,
        "LookDown": 1.0,
        "Stand": 1.0,
        "Crouch": 1.0,
        "PickupObject": 0.8,
        "PutObject": 0.8,
        "DropHandObject": 0.6,
        "ThrowObject": 0.3,
        "PushObject": 0.6,
        "PullObject": 0.6,
        "OpenObject": 0.9,
        "CloseObject": 0.9,
        "ToggleObjectOn": 0.9,
        "ToggleObjectOff": 0.9,
        "FillObjectWithLiquid": 0.7,
        "EmptyLiquidFromObject": 0.7,
        "DirtyObject": 0.5,
        "CleanObject": 0.5,
        "UseUpObject": 0.1,
        "BreakObject": 0.0,
        "SliceObject": 0.0,
        "CookObject": 0.1,
    }

    DEFAULT_REVERSIBILITY: float = 0.5

    def __init__(self, scene: str = "FloorPlan1", headless: bool = True) -> None:
        """Initialize the THOR controller.

        Attempts CloudRendering first for headless operation, then falls
        back to the default platform renderer if that fails. Raises
        THORLaunchError with a descriptive message if neither works.

        Args:
            scene: AI2-THOR scene name (e.g. "FloorPlan1").
            headless: Run without a visible display window.
        """
        self._scene = scene
        self._headless = headless
        self._step_count: int = 0
        self._action_history: list[THORActionResult] = []
        self._controller: Any = None

        self._launch(scene, headless)

    def _launch(self, scene: str, headless: bool) -> None:
        """Attempt to launch the AI2-THOR controller.

        Args:
            scene: AI2-THOR scene name.
            headless: Whether to run without display.

        Raises:
            THORLaunchError: If the controller cannot start.
        """
        try:
            from ai2thor.controller import Controller
        except ImportError as exc:
            raise THORLaunchError(
                "ai2thor is not installed. Install it with: "
                "pip install ai2thor"
            ) from exc

        # Try CloudRendering for headless environments first
        if headless:
            try:
                self._controller = Controller(
                    scene=scene,
                    platform="CloudRendering",
                    renderDepthImage=False,
                    renderInstanceSegmentation=False,
                )
                logger.info("Launched THOR with CloudRendering for scene %s", scene)
                return
            except Exception:
                logger.warning(
                    "CloudRendering unavailable, falling back to default renderer."
                )

        # Fall back to default platform renderer with timeout
        import signal

        def _timeout_handler(signum: int, frame: object) -> None:
            raise TimeoutError(f"THOR launch timed out for scene '{scene}'")

        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)  # 30 second timeout
            self._controller = Controller(
                scene=scene,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
            )
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
            logger.info("Launched THOR with default renderer for scene %s", scene)
        except (TimeoutError, Exception) as exc:
            signal.alarm(0)
            raise THORLaunchError(
                f"Failed to launch AI2-THOR for scene '{scene}'. "
                f"Ensure a display is available or use headless=True. "
                f"Original error: {exc}"
            ) from exc

    def get_state(self) -> THORState:
        """Extract current scene state as a THORState.

        Reads the controller's last event metadata to build a complete
        snapshot of every object in the scene, the agent's position,
        and what the agent is holding.

        Returns:
            A THORState snapshot of the current scene.
        """
        event = self._controller.last_event
        metadata = event.metadata

        objects: list[THORObject] = []
        for obj in metadata["objects"]:
            thor_obj = THORObject(
                object_id=obj["objectId"],
                object_type=obj["objectType"],
                position={
                    "x": obj["position"]["x"],
                    "y": obj["position"]["y"],
                    "z": obj["position"]["z"],
                },
                rotation={
                    "x": obj["rotation"]["x"],
                    "y": obj["rotation"]["y"],
                    "z": obj["rotation"]["z"],
                },
                is_pickupable=obj.get("pickupable", False),
                is_openable=obj.get("openable", False),
                is_toggleable=obj.get("toggleable", False),
                is_breakable=obj.get("breakable", False),
                is_picked_up=obj.get("isPickedUp", False),
                parent_receptacles=obj.get("parentReceptacles") or [],
                visible=obj.get("visible", True),
            )
            objects.append(thor_obj)

        # Determine held object from inventory
        held_object: str | None = None
        inventory = metadata.get("inventoryObjects", [])
        if inventory:
            held_object = inventory[0]["objectId"]

        agent = metadata["agent"]
        return THORState(
            scene_name=self._scene,
            step_number=self._step_count,
            agent_position={
                "x": agent["position"]["x"],
                "y": agent["position"]["y"],
                "z": agent["position"]["z"],
            },
            agent_rotation={
                "x": agent["rotation"]["x"],
                "y": agent["rotation"]["y"],
                "z": agent["rotation"]["z"],
            },
            objects=objects,
            held_object=held_object,
            last_action=metadata.get("lastAction"),
            last_action_success=metadata.get("lastActionSuccess", True),
        )

    def execute_action(
        self,
        action: str,
        target_object: str | None = None,
    ) -> THORActionResult:
        """Execute an action and return the result with state diff.

        Captures the full before/after state and computes a StateDelta
        comparing actual changes against an empty prediction list. To
        compare against planner predictions, use compute_state_delta
        directly.

        Args:
            action: THOR action name (MoveAhead, PickupObject, etc.).
            target_object: Object ID for actions that need a target.

        Returns:
            A THORActionResult with states, delta, and success flag.
        """
        state_before = self.get_state()

        # Build the action dict for THOR
        action_params: dict[str, Any] = {"action": action}
        if target_object is not None:
            action_params["objectId"] = target_object

        event = self._controller.step(**action_params)
        self._step_count += 1

        state_after = self.get_state()
        state_delta = self.compute_state_delta(
            before=state_before,
            after=state_after,
            predicted_changes=[],
        )

        error_message: str | None = None
        if not event.metadata["lastActionSuccess"]:
            error_message = event.metadata.get("errorMessage", "Action failed.")

        result = THORActionResult(
            action=action,
            target_object=target_object,
            success=event.metadata["lastActionSuccess"],
            state_before=state_before,
            state_after=state_after,
            state_delta=state_delta,
            step_number=self._step_count,
            error_message=error_message,
        )
        self._action_history.append(result)
        return result

    def compute_state_delta(
        self,
        before: THORState,
        after: THORState,
        predicted_changes: list[str],
    ) -> StateDelta:
        """Compare predicted vs observed state changes.

        Builds a list of observed changes by diffing the before/after
        snapshots, then compares those against the planner's predicted
        changes using substring matching.

        Args:
            before: Scene state before the action.
            after: Scene state after the action.
            predicted_changes: What the planner expected to happen.

        Returns:
            A StateDelta with matches, mismatches, unexpected changes,
            and the match ratio.
        """
        observed: list[str] = []

        # Check agent position change
        if before.agent_position != after.agent_position:
            observed.append("agent_moved")
        if before.agent_rotation != after.agent_rotation:
            observed.append("agent_rotated")

        # Check held object change
        if before.held_object != after.held_object:
            if after.held_object is not None and before.held_object is None:
                observed.append(f"picked_up:{after.held_object}")
            elif after.held_object is None and before.held_object is not None:
                observed.append(f"put_down:{before.held_object}")
            else:
                observed.append(f"held_changed:{before.held_object}->{after.held_object}")

        # Build lookup dicts for object comparison
        before_objects = {obj.object_id: obj for obj in before.objects}
        after_objects = {obj.object_id: obj for obj in after.objects}

        for obj_id, after_obj in after_objects.items():
            before_obj = before_objects.get(obj_id)
            if before_obj is None:
                observed.append(f"object_appeared:{obj_id}")
                continue

            if before_obj.position != after_obj.position:
                observed.append(f"object_moved:{obj_id}")
            if before_obj.is_picked_up != after_obj.is_picked_up:
                observed.append(f"pickup_state_changed:{obj_id}")
            if before_obj.parent_receptacles != after_obj.parent_receptacles:
                observed.append(f"receptacle_changed:{obj_id}")
            if before_obj.visible != after_obj.visible:
                observed.append(f"visibility_changed:{obj_id}")

        for obj_id in before_objects:
            if obj_id not in after_objects:
                observed.append(f"object_disappeared:{obj_id}")

        # Compare predictions against observations
        matches: list[str] = []
        mismatches: list[str] = []

        for prediction in predicted_changes:
            prediction_lower = prediction.lower()
            matched = False
            for obs in observed:
                if prediction_lower in obs.lower() or obs.lower() in prediction_lower:
                    matched = True
                    break
            if matched:
                matches.append(prediction)
            else:
                mismatches.append(prediction)

        # Find unexpected changes (observed but not predicted)
        unexpected: list[str] = []
        for obs in observed:
            obs_lower = obs.lower()
            was_predicted = False
            for prediction in predicted_changes:
                if prediction.lower() in obs_lower or obs_lower in prediction.lower():
                    was_predicted = True
                    break
            if not was_predicted:
                unexpected.append(obs)

        denominator = max(len(predicted_changes), 1)
        match_ratio = len(matches) / denominator

        return StateDelta(
            predicted_changes=predicted_changes,
            observed_changes=observed,
            matches=matches,
            mismatches=mismatches,
            unexpected=unexpected,
            match_ratio=match_ratio,
        )

    def get_action_reversibility(self, action: str) -> float:
        """Get reversibility score from THOR action metadata.

        More precise than v0.1 keyword matching because THOR knows
        which actions are reversible. Movement is fully reversible,
        toggling is nearly reversible, and breaking or slicing is
        permanent.

        Args:
            action: The THOR action name.

        Returns:
            A float between 0.0 (irreversible) and 1.0 (fully reversible).
        """
        return self.ACTION_REVERSIBILITY.get(action, self.DEFAULT_REVERSIBILITY)

    def find_object(self, object_type: str) -> THORObject | None:
        """Find a visible object of the given type in the current scene.

        Searches the current state for the first visible object matching
        the requested type.

        Args:
            object_type: The THOR object type (e.g. "Mug", "Apple").

        Returns:
            The first matching visible THORObject, or None if not found.
        """
        state = self.get_state()
        for obj in state.objects:
            if obj.object_type == object_type and obj.visible:
                return obj
        return None

    def find_receptacle(self, receptacle_type: str) -> THORObject | None:
        """Find a visible receptacle of the given type.

        Receptacles are objects that can hold other objects (tables,
        shelves, countertops, sinks, etc.).

        Args:
            receptacle_type: The THOR receptacle type (e.g. "DiningTable").

        Returns:
            The first matching visible THORObject, or None if not found.
        """
        state = self.get_state()
        for obj in state.objects:
            if obj.object_type == receptacle_type and obj.visible:
                return obj
        return None

    def get_navigable_positions(self) -> list[dict]:
        """Get all navigable positions in the scene.

        Uses the THOR GetReachablePositions action to return every
        position the agent can move to.

        Returns:
            A list of position dicts, each with x, y, z keys.
        """
        event = self._controller.step("GetReachablePositions")
        positions = event.metadata.get("actionReturn", [])
        return [
            {"x": p["x"], "y": p["y"], "z": p["z"]}
            for p in positions
        ]

    def reset(self, scene: str | None = None) -> None:
        """Reset the scene to its initial state.

        Optionally switch to a different scene. Clears the step counter
        and action history.

        Args:
            scene: New scene name, or None to reload the current scene.
        """
        if scene is not None:
            self._scene = scene

        self._controller.reset(scene=self._scene)
        self._step_count = 0
        self._action_history.clear()
        logger.info("Reset scene to %s", self._scene)

    def stop(self) -> None:
        """Stop the controller and release resources.

        Safe to call multiple times. After stopping, the controller
        cannot be used again.
        """
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception:
                logger.warning("Error stopping THOR controller; ignoring.")
            finally:
                self._controller = None
            logger.info("THOR controller stopped.")

    @property
    def step_count(self) -> int:
        """Return the number of actions executed so far."""
        return self._step_count

    @property
    def action_history(self) -> list[THORActionResult]:
        """Return the full action history for this session."""
        return list(self._action_history)
