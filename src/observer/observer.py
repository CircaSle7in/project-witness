"""The SilentObserver -- the core of Project Witness.

The observer receives signals from every layer but generates no plans and
takes no actions. It emits only structured audit metadata: confidence
assessments, principle flags, gate decisions, and short structured
explanations for the audit trail.

The observer is the agent. The LLM, the world model, and the tools are
processes the observer watches and orchestrates.
"""

from __future__ import annotations

import duckdb

from src.observer.consistency import check_consistency
from src.observer.gate import compute_threshold, decide_gate, score_reversibility
from src.observer.principles import check_principles
from src.observer.uncertainty import apply_platt, platt_scale
from src.pipeline.schemas import (
    CalibrationEntry,
    GateDecision,
    ObserverAssessment,
    PrincipleFlag,
)


class SilentObserver:
    """The Silent Observer metacognitive layer.

    Structurally separate from the planning, reasoning, and world-modeling
    processes it monitors. Emits only structured audit metadata.
    """

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        """Initialize the observer with a DuckDB connection.

        Creates the calibration_log table if it does not exist and loads
        any existing calibration data for Platt scaling.

        Args:
            db: An open DuckDB connection for calibration storage.
        """
        self._db = db
        self._platt_params: dict[str, tuple[float, float]] = {}
        self._ensure_tables()
        self._load_calibration_data()

    def _ensure_tables(self) -> None:
        """Create the calibration_log table if it does not exist."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS calibration_log (
                task_id VARCHAR,
                category VARCHAR,
                model_name VARCHAR,
                raw_confidence DOUBLE,
                calibrated_confidence DOUBLE,
                prediction VARCHAR,
                ground_truth VARCHAR,
                correct BOOLEAN,
                observer_gate VARCHAR
            )
        """)

    def _load_calibration_data(self) -> None:
        """Load existing calibration data and fit Platt scaling per category."""
        try:
            categories = self._db.execute(
                "SELECT DISTINCT category FROM calibration_log"
            ).fetchall()
        except duckdb.Error:
            return

        for (category,) in categories:
            rows = self._db.execute(
                "SELECT raw_confidence, correct FROM calibration_log "
                "WHERE category = ?",
                [category],
            ).fetchall()

            if len(rows) >= 30:
                raw_confs = [r[0] for r in rows]
                correctness = [bool(r[1]) for r in rows]
                a, b = platt_scale(raw_confs, correctness)
                self._platt_params[category] = (a, b)

    def assess(
        self,
        proposed_action: str,
        world_state: dict,
        model_confidence: float,
        belief_state: dict,
        category: str,
    ) -> ObserverAssessment:
        """Assess a proposed action and produce a gate decision.

        This is the main entry point for the observer. It calibrates
        confidence, checks reversibility, runs principle and consistency
        checks, computes the threshold, and decides the gate.

        Args:
            proposed_action: Description of what the model wants to do.
            world_state: Current world state dictionary.
            model_confidence: Raw confidence score from the model (0-1).
            belief_state: Current belief state dictionary.
            category: The evaluation category for calibration lookup.

        Returns:
            An ObserverAssessment with the gate decision and audit metadata.
        """
        # Calibrate confidence
        calibrated = self.calibrate_confidence(model_confidence, category)

        # Score reversibility
        reversibility = self.score_reversibility(proposed_action)

        # Check principles (stub in v0.1)
        principle_flags = self.check_principles(proposed_action, world_state)

        # Check consistency (stub in v0.1)
        conflicts = self.check_consistency(proposed_action, world_state, belief_state)

        # Make the gate decision
        gate = decide_gate(calibrated, reversibility, principle_flags, conflicts)

        # Build audit metadata
        threshold = compute_threshold(reversibility, principle_flags)
        reasoning = self._build_reasoning(gate, calibrated, threshold, conflicts)
        alternative = self._suggest_alternative(gate, proposed_action)

        return ObserverAssessment(
            gate=gate,
            confidence=calibrated,
            raw_confidence=model_confidence,
            reversibility=reversibility,
            principle_flags=principle_flags,
            reasoning=reasoning,
            suggested_alternative=alternative,
        )

    def calibrate_confidence(self, raw_confidence: float, category: str) -> float:
        """Calibrate a raw confidence score using Platt scaling if available.

        Uses Platt scaling when 30+ calibration samples exist for the given
        category. Otherwise applies a conservative multiplier of 0.85.

        Args:
            raw_confidence: The raw confidence score from the model (0-1).
            category: The evaluation category for calibration lookup.

        Returns:
            The calibrated confidence score (0-1).
        """
        if category in self._platt_params:
            a, b = self._platt_params[category]
            return apply_platt(raw_confidence, a, b)

        # Conservative fallback: reduce overconfident predictions
        return raw_confidence * 0.85

    def score_reversibility(self, action: str) -> float:
        """Score how reversible a proposed action is.

        Delegates to the gate module's keyword-based lookup table.

        Args:
            action: Description of the proposed action.

        Returns:
            A float between 0.0 (irreversible) and 1.0 (fully reversible).
        """
        return score_reversibility(action)

    def check_consistency(
        self,
        proposed_action: str,
        world_state: dict,
        belief_state: dict,
    ) -> list:
        """Check for conflicts between action and beliefs.

        Stub in v0.1: always returns an empty list.

        Args:
            proposed_action: Description of the proposed action.
            world_state: Current world state dictionary.
            belief_state: Current belief state dictionary.

        Returns:
            A list of conflict descriptions. Empty in v0.1.
        """
        return check_consistency(proposed_action, world_state, belief_state)

    def check_principles(
        self,
        proposed_action: str,
        world_state: dict,
    ) -> list[PrincipleFlag]:
        """Check proposed action against guiding principles.

        Stub in v0.1: always returns an empty list.

        Args:
            proposed_action: Description of the proposed action.
            world_state: Current world state dictionary.

        Returns:
            A list of PrincipleFlag objects. Empty in v0.1.
        """
        return check_principles(proposed_action, world_state)

    def _build_reasoning(
        self,
        gate: GateDecision,
        calibrated: float,
        threshold: float,
        conflicts: list,
    ) -> str:
        """Build structured reasoning text for the audit trail.

        This is audit metadata describing why the gate decision was made.
        It is NOT a plan or proposal.

        Args:
            gate: The gate decision that was made.
            calibrated: The calibrated confidence score.
            threshold: The dynamic threshold that was computed.
            conflicts: List of belief conflicts detected.

        Returns:
            A human-readable reasoning string for the audit log.
        """
        parts = [
            f"Gate decision: {gate.value}.",
            f"Calibrated confidence: {calibrated:.3f} (threshold: {threshold:.3f}).",
        ]

        if gate == GateDecision.ACT:
            parts.append("Confidence meets threshold. Action approved.")
        elif gate == GateDecision.ASK_HUMAN:
            parts.append(
                "Confidence below threshold for a low-reversibility action. "
                "Human approval required."
            )
        elif gate == GateDecision.WAIT:
            parts.append(
                "Confidence below threshold but action is reversible. "
                "Waiting for more information."
            )
        elif gate == GateDecision.GATHER_EVIDENCE:
            parts.append(
                f"Detected {len(conflicts)} belief conflict(s). "
                "More evidence needed before proceeding."
            )
        elif gate == GateDecision.REFUSE:
            parts.append(
                "Hard principle flag raised with insufficient confidence. "
                "Action refused."
            )

        return " ".join(parts)

    def _suggest_alternative(
        self,
        gate: GateDecision,
        proposed_action: str,
    ) -> str | None:
        """Suggest an alternative approach when the action is not approved.

        This is audit metadata for the trace log, NOT a plan or proposal.

        Args:
            gate: The gate decision that was made.
            proposed_action: The action that was proposed.

        Returns:
            A brief suggestion string, or None if the action was approved.
        """
        if gate == GateDecision.ACT:
            return None
        if gate == GateDecision.ASK_HUMAN:
            return "Request human confirmation before proceeding."
        if gate == GateDecision.WAIT:
            return "Pause and re-evaluate after gathering more context."
        if gate == GateDecision.GATHER_EVIDENCE:
            return "Collect additional observations to resolve belief conflicts."
        if gate == GateDecision.REFUSE:
            return "This action conflicts with hard principles. Consider a less invasive approach."
        return None

    def log_calibration(self, entry: CalibrationEntry) -> None:
        """Write a calibration entry to the calibration_log DuckDB table.

        Args:
            entry: The calibration data point to record.
        """
        self._db.execute(
            "INSERT INTO calibration_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                entry.task_id,
                entry.category,
                entry.model_name,
                entry.raw_confidence,
                entry.calibrated_confidence,
                entry.prediction,
                entry.ground_truth,
                entry.correct,
                entry.observer_gate,
            ],
        )

        # Reload Platt params for this category if we have enough data
        rows = self._db.execute(
            "SELECT raw_confidence, correct FROM calibration_log WHERE category = ?",
            [entry.category],
        ).fetchall()

        if len(rows) >= 30:
            raw_confs = [r[0] for r in rows]
            correctness = [bool(r[1]) for r in rows]
            a, b = platt_scale(raw_confs, correctness)
            self._platt_params[entry.category] = (a, b)

    def get_calibration_stats(self, category: str) -> dict:
        """Get calibration statistics for a category.

        Args:
            category: The evaluation category to query.

        Returns:
            A dict with count, mean_confidence, and mean_accuracy.
        """
        try:
            result = self._db.execute(
                "SELECT "
                "  COUNT(*) AS count, "
                "  AVG(calibrated_confidence) AS mean_confidence, "
                "  AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) AS mean_accuracy "
                "FROM calibration_log "
                "WHERE category = ?",
                [category],
            ).fetchone()

            if result:
                return {
                    "count": int(result[0]),
                    "mean_confidence": float(result[1]) if result[1] is not None else 0.0,
                    "mean_accuracy": float(result[2]) if result[2] is not None else 0.0,
                }
        except duckdb.Error:
            pass

        return {"count": 0, "mean_confidence": 0.0, "mean_accuracy": 0.0}
