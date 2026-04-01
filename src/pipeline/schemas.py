"""Pydantic v2 models used across the Project Witness pipeline.

All data shapes for evaluation tasks, observer assessments, calibration
entries, and frame extractions are defined here as the single source of truth.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class EvalTask(BaseModel):
    """A single evaluation task loaded from a YAML benchmark file."""

    task_id: str
    video: str | None = None
    image: str | None = None
    category: str
    subcategory: str
    prompt: str
    expected: str
    expected_actions: list[str] | None = None
    expected_risks: list[str] | None = None
    scoring: Literal["exact", "fuzzy", "llm_judge", "multi_match"]
    difficulty: Literal["easy", "medium", "hard"]
    requires_uncertainty: bool
    metadata: dict | None = None


class PrincipleFlag(BaseModel):
    """A typed flag raised when a guiding principle may be violated.

    Always a structured Pydantic object, never a plain string.
    """

    name: str
    severity: Literal["soft", "hard"]
    description: str


class GateDecision(StrEnum):
    """The five possible outcomes of the observer execution gate."""

    ACT = "act"
    ASK_HUMAN = "ask"
    WAIT = "wait"
    GATHER_EVIDENCE = "gather"
    REFUSE = "refuse"


class SelfModel(BaseModel):
    """What the system knows about its own state. Updated each cycle."""

    allowed_actions: list[str] = Field(default_factory=list)
    current_beliefs: dict = Field(default_factory=dict)
    uncertainty_areas: list[str] = Field(default_factory=list)
    active_commitments: list[str] = Field(default_factory=list)
    unverifiable_claims: list[str] = Field(default_factory=list)
    requires_human_approval: list[str] = Field(default_factory=list)
    last_calibration_error: float = 0.0
    cycle_count: int = 0


class ObserverAssessment(BaseModel):
    """Structured audit metadata emitted by the Silent Observer.

    The reasoning and suggested_alternative fields are audit metadata for the
    cockpit and trace log. They are NOT plans or proposals.
    """

    gate: GateDecision
    confidence: float
    raw_confidence: float
    reversibility: float
    principle_flags: list[PrincipleFlag] = Field(default_factory=list)
    reasoning: str
    suggested_alternative: str | None = None


class EvalResult(BaseModel):
    """Result of running a single evaluation task against a model."""

    run_id: str
    timestamp: str
    task_id: str
    category: str
    subcategory: str
    model_name: str
    model_response: str
    expected: str
    score: float
    scoring_method: str
    judge_explanation: str | None = None
    raw_confidence: float
    observer_gate: str | None = None
    observer_confidence: float | None = None
    latency_ms: int
    token_count: int


class FrameExtraction(BaseModel):
    """Structured extraction from a single video frame via VLM."""

    frame_number: int
    timestamp_s: float
    entities: list[dict] = Field(default_factory=list)
    actions: list[dict] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class CalibrationEntry(BaseModel):
    """A single calibration data point for confidence tracking."""

    task_id: str
    category: str
    model_name: str
    raw_confidence: float
    calibrated_confidence: float
    prediction: str
    ground_truth: str
    correct: bool
    observer_gate: str | None = None
