"""Closed-loop agent for AI2-THOR rearrangement tasks.

Runs the Witness loop: Observe -> Propose -> (Witness) -> Execute -> Log.
Can run in baseline mode (no observer) or observed mode (observer gates
actions before execution).
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from src.pipeline.schemas import GateDecision, ObserverAssessment
from src.thor.schemas import (
    ActionProposal,
    TaskDefinition,
    TaskResult,
    THORActionResult,
    THORState,
)

if TYPE_CHECKING:
    import duckdb

    from src.observer.observer import SilentObserver
    from src.thor.controller import WitnessController
    from src.thor.planner import ActionPlanner

logger = logging.getLogger(__name__)

# Navigation actions used as evidence-gathering fallbacks when the
# observer issues a GATHER_EVIDENCE gate decision.
_EVIDENCE_ACTIONS: list[str] = [
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
]

# Maximum consecutive skipped actions before the agent gives up.
# Prevents infinite loops when the observer blocks everything.
_MAX_CONSECUTIVE_SKIPS: int = 5


class WitnessAgent:
    """Closed-loop agent for AI2-THOR environments.

    Runs the Witness loop: Observe -> Propose -> (Witness) -> Execute -> Log.
    Can run in baseline mode (no observer) or observed mode (observer gates
    actions).
    """

    def __init__(
        self,
        controller: WitnessController,
        planner: ActionPlanner,
        observer: SilentObserver | None = None,
        db: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            controller: The THOR controller for scene interaction.
            planner: The LLM-based action planner.
            observer: If provided, gates actions before execution. None = baseline.
            db: DuckDB connection for logging results.
        """
        self._controller = controller
        self._planner = planner
        self._observer = observer
        self._db = db

    async def run_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a complete task from start to finish.

        The loop:
        1. Get current state from THOR
        2. Ask planner to propose next action
        3. If observer exists, assess the proposal
           - ACT: execute the action
           - ASK_HUMAN / WAIT: skip this action, try again
           - GATHER_EVIDENCE: execute a look/rotate action instead
           - REFUSE: skip and log
        4. Execute action in THOR
        5. Record result
        6. Check if task is complete (target object on target receptacle)
        7. Repeat until success, max_steps, or stuck

        Args:
            task: The rearrangement task to execute.

        Returns:
            A TaskResult with success, steps taken, actions, and observer stats.
        """
        start_time = time.monotonic()

        action_log: list[dict] = []
        action_history: list[THORActionResult] = []
        gate_counts: dict[str, int] = {g.value: 0 for g in GateDecision}
        confidence_values: list[float] = []
        prediction_trust_values: list[float] = []

        total_proposed = 0
        total_executed = 0
        total_gated = 0
        consecutive_skips = 0

        success = False

        for step in range(task.max_steps):
            # 1. Observe: get current state
            state = self._controller.get_state()

            # Check completion before proposing
            if self._check_task_complete(state, task):
                success = True
                logger.info(
                    "Task %s completed at step %d", task.task_id, step
                )
                break

            # 2. Propose: ask planner for next action
            proposal = await self._planner.propose_action(
                task, state, action_history
            )
            total_proposed += 1

            # 3. Witness: if observer exists, assess the proposal
            action_to_execute: str | None = proposal.action
            target_to_execute: str | None = proposal.target_object
            assessment: ObserverAssessment | None = None

            if self._observer is not None:
                assessment = self._observer.assess_action(
                    proposal=proposal,
                    current_state=state,
                    action_history=action_history,
                )
                gate_counts[assessment.gate.value] = (
                    gate_counts.get(assessment.gate.value, 0) + 1
                )
                confidence_values.append(assessment.confidence)

                # Compute prediction trust for logging
                trust = self._observer._compute_prediction_trust(action_history)
                prediction_trust_values.append(trust)

                resolved = self._handle_gate_decision(
                    assessment.gate, proposal, state
                )
                if resolved is None:
                    # Action was blocked
                    total_gated += 1
                    consecutive_skips += 1
                    action_log.append({
                        "step": step,
                        "proposed_action": proposal.action,
                        "proposed_target": proposal.target_object,
                        "gate": assessment.gate.value,
                        "confidence": round(assessment.confidence, 3),
                        "executed": False,
                        "reason": assessment.reasoning,
                    })
                    logger.debug(
                        "Step %d: %s gated (%s)",
                        step, proposal.action, assessment.gate.value,
                    )
                    if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                        logger.warning(
                            "Task %s: %d consecutive skips, giving up.",
                            task.task_id, consecutive_skips,
                        )
                        break
                    continue
                else:
                    # Action approved (possibly replaced with evidence action)
                    action_to_execute = resolved
                    # If the observer replaced the action with a look/rotate,
                    # clear the target since navigation actions have no target.
                    if resolved != proposal.action:
                        target_to_execute = None
                    consecutive_skips = 0
            else:
                # Baseline mode: no observer, always execute
                confidence_values.append(proposal.planner_confidence)
                consecutive_skips = 0

            # 4. Execute: run the action in THOR
            result = self._controller.execute_action(
                action_to_execute, target_to_execute
            )
            action_history.append(result)
            total_executed += 1

            # 5. Record
            log_entry: dict = {
                "step": step,
                "proposed_action": proposal.action,
                "proposed_target": proposal.target_object,
                "executed_action": action_to_execute,
                "executed_target": target_to_execute,
                "success": result.success,
                "executed": True,
            }
            if assessment is not None:
                log_entry["gate"] = assessment.gate.value
                log_entry["confidence"] = round(assessment.confidence, 3)
            if result.error_message:
                log_entry["error"] = result.error_message

            action_log.append(log_entry)
            logger.debug(
                "Step %d: %s -> %s (success=%s)",
                step, action_to_execute, target_to_execute, result.success,
            )

        # Final completion check after loop exits
        if not success:
            final_state = self._controller.get_state()
            success = self._check_task_complete(final_state, task)

        elapsed = time.monotonic() - start_time

        mean_conf = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        mean_trust = (
            sum(prediction_trust_values) / len(prediction_trust_values)
            if prediction_trust_values
            else 0.0
        )

        return TaskResult(
            task_id=task.task_id,
            scene_name=task.scene_name,
            success=success,
            steps_taken=total_executed,
            max_steps=task.max_steps,
            total_actions_proposed=total_proposed,
            actions_executed=total_executed,
            actions_gated=total_gated,
            observer_active=self._observer is not None,
            action_log=action_log,
            gate_distribution=gate_counts,
            mean_confidence=round(mean_conf, 4),
            mean_prediction_trust=round(mean_trust, 4),
            completion_time_s=round(elapsed, 2),
        )

    def _check_task_complete(
        self, state: THORState, task: TaskDefinition
    ) -> bool:
        """Check if the target object is on the target receptacle.

        Iterates through scene objects looking for the target object type.
        If found, checks whether its parent_receptacles include the target
        receptacle type (matched by substring, since THOR receptacle IDs
        contain the type name).

        Args:
            state: Current scene state snapshot.
            task: The task definition with target object and receptacle.

        Returns:
            True if any matching object is on the target receptacle.
        """
        for obj in state.objects:
            if obj.object_type != task.target_object_type:
                continue
            for receptacle_id in obj.parent_receptacles:
                if task.target_receptacle_type in receptacle_id:
                    return True
        return False

    def _handle_gate_decision(
        self,
        gate: GateDecision,
        proposal: ActionProposal,
        state: THORState,
    ) -> str | None:
        """Decide what to do based on observer gate.

        Returns the action to execute, or None to skip.
        - ACT: return the proposed action
        - GATHER_EVIDENCE: return a look/rotate action
        - ASK_HUMAN / WAIT / REFUSE: return None (skip)

        Args:
            gate: The observer's gate decision.
            proposal: The planner's action proposal.
            state: Current scene state (used to pick evidence actions).

        Returns:
            The action string to execute, or None to skip this step.
        """
        if gate == GateDecision.ACT:
            return proposal.action

        if gate == GateDecision.GATHER_EVIDENCE:
            # Pick a random look/rotate action to gather more visual info
            return random.choice(_EVIDENCE_ACTIONS)

        # ASK_HUMAN, WAIT, REFUSE: skip this action
        return None
