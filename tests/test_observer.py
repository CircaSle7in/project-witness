"""Tests for the Silent Observer module."""

from __future__ import annotations

import duckdb
import pytest

from src.observer.gate import BASE_THRESHOLD, compute_threshold, score_reversibility
from src.observer.observer import SilentObserver
from src.pipeline.schemas import GateDecision, ObserverAssessment, PrincipleFlag


@pytest.fixture()
def db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with an empty calibration_log table."""
    conn = duckdb.connect(":memory:")
    return conn


@pytest.fixture()
def observer(db: duckdb.DuckDBPyConnection) -> SilentObserver:
    """SilentObserver backed by the in-memory DuckDB."""
    return SilentObserver(db)


class TestCalibration:
    """Confidence calibration before and after sufficient samples."""

    def test_conservative_threshold_before_calibration(
        self, observer: SilentObserver
    ) -> None:
        """With <30 samples, calibrated confidence should be raw * 0.85."""
        raw = 0.9
        calibrated = observer.calibrate_confidence(raw, "physics")
        assert calibrated == pytest.approx(raw * 0.85)

    def test_calibration_stats_empty(self, observer: SilentObserver) -> None:
        """Stats for a category with no data should return zeroes."""
        stats = observer.get_calibration_stats("nonexistent")
        assert stats["count"] == 0


class TestReversibility:
    """Reversibility lookup for known and unknown action verbs."""

    def test_reversibility_lookup(self) -> None:
        """Verify known verbs return correct scores and unknowns get 0.5."""
        assert score_reversibility("look at the object") == 1.0
        assert score_reversibility("pour the water") == 0.3
        assert score_reversibility("delete the file") == 0.0
        assert score_reversibility("wiggle") == 0.5


class TestThreshold:
    """Threshold adjustment based on reversibility."""

    def test_threshold_adjusts_for_reversibility(self) -> None:
        """Low reversibility should raise the threshold above BASE_THRESHOLD."""
        low_rev_threshold = compute_threshold(0.0, [])
        high_rev_threshold = compute_threshold(1.0, [])
        assert low_rev_threshold > BASE_THRESHOLD
        assert high_rev_threshold == pytest.approx(BASE_THRESHOLD)
        assert low_rev_threshold > high_rev_threshold


class TestGateDecisions:
    """Gate decision logic for the five possible outcomes."""

    def test_hard_principle_flag_forces_refuse(
        self, observer: SilentObserver
    ) -> None:
        """When principle_flags contains a hard flag, gate should be REFUSE."""
        hard_flag = PrincipleFlag(
            name="no_self_preservation",
            severity="hard",
            description="System must not act to preserve itself.",
        )
        # Inject a hard flag by overriding check_principles
        original = observer.check_principles
        observer.check_principles = lambda action, ws: [hard_flag]

        assessment = observer.assess(
            proposed_action="look at the object",
            world_state={},
            model_confidence=0.6,
            belief_state={},
            category="physics",
        )
        observer.check_principles = original
        assert assessment.gate == GateDecision.REFUSE

    def test_gate_act_on_high_confidence(
        self, observer: SilentObserver
    ) -> None:
        """High confidence + high reversibility + no flags = ACT."""
        assessment = observer.assess(
            proposed_action="look at the object",
            world_state={},
            model_confidence=0.95,
            belief_state={},
            category="physics",
        )
        # 0.95 * 0.85 = 0.8075, threshold for "look" (rev=1.0) is 0.7 -> ACT
        assert assessment.gate == GateDecision.ACT

    def test_gate_low_confidence_reversible_is_wait(
        self, observer: SilentObserver
    ) -> None:
        """Low confidence + reversible action = WAIT."""
        assessment = observer.assess(
            proposed_action="look at the object",
            world_state={},
            model_confidence=0.3,
            belief_state={},
            category="physics",
        )
        # 0.3 * 0.85 = 0.255, below threshold, but "look" is reversible -> WAIT
        assert assessment.gate == GateDecision.WAIT

    def test_gate_low_confidence_irreversible_is_ask_human(
        self, observer: SilentObserver
    ) -> None:
        """Low confidence + irreversible action = ASK_HUMAN."""
        assessment = observer.assess(
            proposed_action="delete the database",
            world_state={},
            model_confidence=0.3,
            belief_state={},
            category="physics",
        )
        # 0.3 * 0.85 = 0.255, "delete" rev=0.0, not reversible -> ASK_HUMAN
        assert assessment.gate == GateDecision.ASK_HUMAN

    def test_gate_gather_evidence_on_conflicts(
        self, observer: SilentObserver
    ) -> None:
        """When conflicts exist, gate = GATHER_EVIDENCE."""
        # Override check_consistency to return conflicts
        original = observer.check_consistency
        observer.check_consistency = lambda a, ws, bs: ["belief A contradicts belief B"]

        assessment = observer.assess(
            proposed_action="pour the water",
            world_state={},
            model_confidence=0.5,
            belief_state={},
            category="physics",
        )
        observer.check_consistency = original
        # Low confidence + conflicts -> GATHER_EVIDENCE
        assert assessment.gate == GateDecision.GATHER_EVIDENCE


class TestAssessmentTypes:
    """Structural checks on the ObserverAssessment output."""

    def test_assessment_fields_are_correct_types(
        self, observer: SilentObserver
    ) -> None:
        """Verify ObserverAssessment has correct field types."""
        assessment = observer.assess(
            proposed_action="look at the object",
            world_state={},
            model_confidence=0.95,
            belief_state={},
            category="physics",
        )
        assert isinstance(assessment, ObserverAssessment)
        assert all(isinstance(f, PrincipleFlag) for f in assessment.principle_flags)
        assert isinstance(assessment.gate, GateDecision)
        assert isinstance(assessment.confidence, float)
        assert isinstance(assessment.reversibility, float)
        assert isinstance(assessment.reasoning, str)
