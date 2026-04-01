"""Tests for all Pydantic schemas in src.pipeline.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline.schemas import (
    CalibrationEntry,
    EvalTask,
    FrameExtraction,
    GateDecision,
    ObserverAssessment,
    PrincipleFlag,
    SelfModel,
)


class TestGateDecision:
    """GateDecision enum values."""

    def test_gate_decision_values(self) -> None:
        """Verify all 5 enum values exist."""
        assert GateDecision.ACT.value == "act"
        assert GateDecision.ASK_HUMAN.value == "ask"
        assert GateDecision.WAIT.value == "wait"
        assert GateDecision.GATHER_EVIDENCE.value == "gather"
        assert GateDecision.REFUSE.value == "refuse"
        assert len(GateDecision) == 5


class TestPrincipleFlag:
    """PrincipleFlag validation."""

    def test_principle_flag_requires_severity(self) -> None:
        """Severity must be 'soft' or 'hard'; other values are rejected."""
        soft = PrincipleFlag(name="test", severity="soft", description="A soft flag.")
        assert soft.severity == "soft"

        hard = PrincipleFlag(name="test", severity="hard", description="A hard flag.")
        assert hard.severity == "hard"

        with pytest.raises(ValidationError):
            PrincipleFlag(name="test", severity="medium", description="Invalid.")


class TestEvalTask:
    """EvalTask schema validation."""

    def test_eval_task_scoring_validation(self) -> None:
        """Scoring must be one of the 4 valid values."""
        base = {
            "task_id": "t1",
            "category": "physics",
            "subcategory": "gravity",
            "prompt": "What happens?",
            "expected": "It falls.",
            "difficulty": "easy",
            "requires_uncertainty": False,
        }
        for valid in ("exact", "fuzzy", "llm_judge", "multi_match"):
            task = EvalTask(**base, scoring=valid)
            assert task.scoring == valid

        with pytest.raises(ValidationError):
            EvalTask(**base, scoring="invalid_method")


class TestSelfModel:
    """SelfModel default values."""

    def test_self_model_defaults(self) -> None:
        """cycle_count defaults to 0, last_calibration_error defaults to 0.0."""
        model = SelfModel()
        assert model.cycle_count == 0
        assert model.last_calibration_error == 0.0
        assert model.allowed_actions == []
        assert model.current_beliefs == {}
        assert model.uncertainty_areas == []
        assert model.active_commitments == []
        assert model.unverifiable_claims == []
        assert model.requires_human_approval == []


class TestObserverAssessment:
    """ObserverAssessment serialization round-trip."""

    def test_observer_assessment_serialization(self) -> None:
        """Can serialize to dict and back."""
        flag = PrincipleFlag(
            name="truthfulness",
            severity="soft",
            description="No deception.",
        )
        assessment = ObserverAssessment(
            gate=GateDecision.ACT,
            confidence=0.85,
            raw_confidence=0.9,
            reversibility=1.0,
            principle_flags=[flag],
            reasoning="Sufficient confidence to proceed.",
        )

        data = assessment.model_dump()
        assert isinstance(data, dict)
        assert data["gate"] == "act"
        assert data["confidence"] == 0.85
        assert len(data["principle_flags"]) == 1
        assert data["principle_flags"][0]["severity"] == "soft"

        restored = ObserverAssessment.model_validate(data)
        assert restored.gate == GateDecision.ACT
        assert restored.confidence == 0.85
        assert isinstance(restored.principle_flags[0], PrincipleFlag)


class TestFrameExtraction:
    """FrameExtraction field validation."""

    def test_frame_extraction_fields(self) -> None:
        """Verify FrameExtraction accepts expected structure."""
        frame = FrameExtraction(
            frame_number=42,
            timestamp_s=1.5,
            entities=[{"type": "cup", "bbox": [10, 20, 50, 60]}],
            actions=[{"verb": "pour", "confidence": 0.8}],
            predictions=["liquid will spill"],
            uncertainties=["cup orientation unclear"],
        )
        assert frame.frame_number == 42
        assert frame.timestamp_s == 1.5
        assert len(frame.entities) == 1
        assert frame.entities[0]["type"] == "cup"
        assert len(frame.actions) == 1
        assert len(frame.predictions) == 1
        assert len(frame.uncertainties) == 1

    def test_frame_extraction_defaults(self) -> None:
        """Lists default to empty."""
        frame = FrameExtraction(frame_number=0, timestamp_s=0.0)
        assert frame.entities == []
        assert frame.actions == []
        assert frame.predictions == []
        assert frame.uncertainties == []


class TestCalibrationEntry:
    """CalibrationEntry field validation."""

    def test_calibration_entry_fields(self) -> None:
        """Verify CalibrationEntry accepts and stores all fields."""
        entry = CalibrationEntry(
            task_id="phys_001",
            category="basic_physics",
            model_name="gemini-flash",
            raw_confidence=0.9,
            calibrated_confidence=0.765,
            prediction="it falls",
            ground_truth="it falls",
            correct=True,
            observer_gate="act",
        )
        assert entry.task_id == "phys_001"
        assert entry.category == "basic_physics"
        assert entry.model_name == "gemini-flash"
        assert entry.raw_confidence == 0.9
        assert entry.calibrated_confidence == 0.765
        assert entry.correct is True
        assert entry.observer_gate == "act"

    def test_calibration_entry_optional_gate(self) -> None:
        """observer_gate is optional and defaults to None."""
        entry = CalibrationEntry(
            task_id="t1",
            category="c1",
            model_name="m1",
            raw_confidence=0.5,
            calibrated_confidence=0.4,
            prediction="a",
            ground_truth="b",
            correct=False,
        )
        assert entry.observer_gate is None
