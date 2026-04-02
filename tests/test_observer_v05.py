"""Tests for the Silent Observer v0.5 multi-signal integration.

Uses mock THOR objects to avoid requiring the AI2-THOR runtime.
The existing v0.1 tests in test_observer.py must continue to pass;
these tests cover only the new assess_action method and helpers.
"""

from __future__ import annotations

import duckdb
import pytest

from src.observer.observer import SilentObserver
from src.pipeline.schemas import GateDecision, ObserverAssessment  # noqa: F401
from src.thor.schemas import ActionProposal, StateDelta, THORActionResult, THORObject, THORState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB for calibration storage."""
    return duckdb.connect(":memory:")


@pytest.fixture()
def observer(db: duckdb.DuckDBPyConnection) -> SilentObserver:
    """SilentObserver backed by the in-memory DuckDB."""
    return SilentObserver(db)


def _make_object(
    object_id: str = "Mug|1",
    object_type: str = "Mug",
    visible: bool = True,
) -> THORObject:
    """Create a minimal THORObject for testing."""
    return THORObject(
        object_id=object_id,
        object_type=object_type,
        position={"x": 0.0, "y": 0.9, "z": 0.0},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        visible=visible,
    )


def _make_state(
    objects: list[THORObject] | None = None,
    held_object: str | None = None,
) -> THORState:
    """Create a minimal THORState for testing."""
    return THORState(
        scene_name="FloorPlan1",
        step_number=0,
        agent_position={"x": 0.0, "y": 0.9, "z": 0.0},
        agent_rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        objects=objects or [_make_object()],
        held_object=held_object,
    )


def _make_proposal(
    action: str = "PickupObject",
    target_object: str | None = "Mug|1",
    planner_confidence: float = 0.8,
) -> ActionProposal:
    """Create a minimal ActionProposal for testing."""
    return ActionProposal(
        action=action,
        target_object=target_object,
        predicted_state_changes=["Mug|1 picked up"],
        planner_confidence=planner_confidence,
        reasoning="The mug is on the table, pick it up.",
    )


def _make_action_result(
    action: str = "MoveAhead",
    target_object: str | None = None,
    success: bool = True,
    match_ratio: float = 0.9,
    step_number: int = 1,
) -> THORActionResult:
    """Create a minimal THORActionResult for testing."""
    state = _make_state()
    return THORActionResult(
        action=action,
        target_object=target_object,
        success=success,
        state_before=state,
        state_after=state,
        state_delta=StateDelta(
            predicted_changes=["moved forward"],
            observed_changes=["moved forward"],
            matches=["moved forward"],
            mismatches=[],
            unexpected=[],
            match_ratio=match_ratio,
        ),
        step_number=step_number,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssessAction:
    """Tests for the new assess_action multi-signal method."""

    def test_assess_action_returns_assessment(
        self, observer: SilentObserver
    ) -> None:
        """Basic smoke test: assess_action returns a valid ObserverAssessment."""
        proposal = _make_proposal(action="MoveAhead", target_object=None)
        state = _make_state()

        result = observer.assess_action(proposal, state, action_history=[])

        assert isinstance(result, ObserverAssessment)
        assert isinstance(result.gate, GateDecision)
        assert 0.0 <= result.confidence <= 1.0
        assert result.raw_confidence == proposal.planner_confidence
        assert result.reasoning  # non-empty

    def test_prediction_trust_affects_confidence(
        self, observer: SilentObserver
    ) -> None:
        """Low prediction trust (bad history) should lower combined confidence."""
        proposal = _make_proposal(action="MoveAhead", target_object=None)
        state = _make_state()

        # History with high match ratios
        good_history = [
            _make_action_result(match_ratio=0.95, step_number=i)
            for i in range(5)
        ]
        result_good = observer.assess_action(proposal, state, good_history)

        # History with low match ratios
        bad_history = [
            _make_action_result(match_ratio=0.1, step_number=i)
            for i in range(5)
        ]
        result_bad = observer.assess_action(proposal, state, bad_history)

        assert result_good.confidence > result_bad.confidence

    def test_target_not_visible_creates_conflict(
        self, observer: SilentObserver
    ) -> None:
        """An invisible target object should trigger GATHER_EVIDENCE."""
        invisible_mug = _make_object(object_id="Mug|1", visible=False)
        state = _make_state(objects=[invisible_mug])
        proposal = _make_proposal(
            action="PickupObject",
            target_object="Mug|1",
            planner_confidence=0.8,
        )

        result = observer.assess_action(proposal, state, action_history=[])

        assert result.gate == GateDecision.GATHER_EVIDENCE
        assert any("not visible" in c for c in result.reasoning.split("."))

    def test_repeated_failure_creates_conflict(
        self, observer: SilentObserver
    ) -> None:
        """An action that failed recently on the same target should create a conflict."""
        state = _make_state()
        proposal = _make_proposal(
            action="PickupObject",
            target_object="Mug|1",
            planner_confidence=0.8,
        )

        failed_history = [
            _make_action_result(
                action="PickupObject",
                target_object="Mug|1",
                success=False,
                step_number=2,
            ),
        ]

        result = observer.assess_action(proposal, state, failed_history)

        assert result.gate == GateDecision.GATHER_EVIDENCE
        assert "failed" in result.reasoning.lower()

    def test_holding_object_blocks_pickup(
        self, observer: SilentObserver
    ) -> None:
        """Trying to pick up while already holding something triggers conflict."""
        state = _make_state(held_object="Apple|1")
        proposal = _make_proposal(
            action="PickupObject",
            target_object="Mug|1",
            planner_confidence=0.9,
        )

        result = observer.assess_action(proposal, state, action_history=[])

        assert result.gate == GateDecision.GATHER_EVIDENCE
        assert "already holding" in result.reasoning.lower()

    def test_irreversible_action_raises_threshold(
        self, observer: SilentObserver
    ) -> None:
        """BreakObject (reversibility=0.0) requires higher confidence than MoveAhead.

        With planner_confidence=0.95 and good prediction history, the
        combined confidence lands above the MoveAhead threshold (0.7) but
        below the BreakObject threshold (0.9). This proves the reversibility
        signal meaningfully affects gating.
        """
        state = _make_state()
        good_history = [
            _make_action_result(match_ratio=0.95, step_number=i)
            for i in range(5)
        ]

        break_proposal = _make_proposal(
            action="BreakObject",
            target_object="Mug|1",
            planner_confidence=0.95,
        )
        move_proposal = _make_proposal(
            action="MoveAhead",
            target_object=None,
            planner_confidence=0.95,
        )

        result_break = observer.assess_action(break_proposal, state, good_history)
        result_move = observer.assess_action(move_proposal, state, good_history)

        # MoveAhead (rev=1.0, threshold=0.7) should get ACT
        assert result_move.gate == GateDecision.ACT
        # BreakObject (rev=0.0, threshold=0.9) should NOT get ACT
        assert result_break.gate != GateDecision.ACT
        # Verify via reversibility on the assessment
        assert result_break.reversibility < result_move.reversibility


class TestThorReversibility:
    """Tests for the THOR-specific reversibility lookup."""

    def test_known_reversible_actions(self, observer: SilentObserver) -> None:
        """Movement and look actions should be fully reversible."""
        assert observer._thor_reversibility("MoveAhead") == 1.0
        assert observer._thor_reversibility("RotateLeft") == 1.0
        assert observer._thor_reversibility("LookUp") == 1.0

    def test_known_irreversible_actions(self, observer: SilentObserver) -> None:
        """Break and slice actions should be irreversible."""
        assert observer._thor_reversibility("BreakObject") == 0.0
        assert observer._thor_reversibility("SliceObject") == 0.0

    def test_unknown_action_gets_default(self, observer: SilentObserver) -> None:
        """An unknown action name should return 0.5."""
        assert observer._thor_reversibility("SomeNewAction") == 0.5


class TestPredictionTrust:
    """Tests for the prediction trust computation."""

    def test_empty_history_returns_neutral(self, observer: SilentObserver) -> None:
        """No history should return 0.5 (neutral trust)."""
        assert observer._compute_prediction_trust([]) == 0.5

    def test_perfect_history_returns_high(self, observer: SilentObserver) -> None:
        """All match_ratio=1.0 should return 1.0."""
        history = [_make_action_result(match_ratio=1.0, step_number=i) for i in range(3)]
        assert observer._compute_prediction_trust(history) == pytest.approx(1.0)

    def test_uses_last_five_only(self, observer: SilentObserver) -> None:
        """Only the last 5 results should be considered."""
        # 10 bad results followed by 5 good results
        old_bad = [_make_action_result(match_ratio=0.0, step_number=i) for i in range(10)]
        recent_good = [_make_action_result(match_ratio=1.0, step_number=i) for i in range(10, 15)]
        history = old_bad + recent_good

        trust = observer._compute_prediction_trust(history)
        assert trust == pytest.approx(1.0)
