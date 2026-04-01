"""Tests for the evaluation harness module."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml

from src.eval.harness import load_tasks
from src.eval.judge import exact_match, multi_match
from src.pipeline.schemas import EvalResult, EvalTask


def _sample_task_list() -> list[dict]:
    """Return a minimal valid list of task dicts."""
    return [
        {
            "task_id": "phys_001",
            "category": "basic_physics",
            "subcategory": "gravity",
            "prompt": "What happens when you drop a ball?",
            "expected": "It falls down.",
            "scoring": "exact",
            "difficulty": "easy",
            "requires_uncertainty": False,
        },
        {
            "task_id": "phys_002",
            "category": "basic_physics",
            "subcategory": "collision",
            "prompt": "What happens when two balls collide?",
            "expected": "They bounce off each other.",
            "scoring": "fuzzy",
            "difficulty": "medium",
            "requires_uncertainty": True,
        },
    ]


def _sample_eval_result(task_id: str = "phys_001", run_id: str = "run_1") -> EvalResult:
    """Return a minimal valid EvalResult."""
    return EvalResult(
        run_id=run_id,
        timestamp="2026-04-01T00:00:00Z",
        task_id=task_id,
        category="basic_physics",
        subcategory="gravity",
        model_name="gemini-flash",
        model_response="It falls down.",
        expected="It falls down.",
        score=1.0,
        scoring_method="exact",
        judge_explanation=None,
        raw_confidence=0.95,
        observer_gate="act",
        observer_confidence=0.9,
        latency_ms=120,
        token_count=42,
    )


class TestLoadTasks:
    """Loading benchmark tasks from YAML files."""

    def test_load_tasks_from_yaml(self, tmp_path: Path) -> None:
        """Create a temp YAML dir with a file, load it, verify EvalTask objects."""
        yaml_file = tmp_path / "test_tasks.yaml"
        yaml_file.write_text(yaml.dump(_sample_task_list()))

        tasks = load_tasks(str(tmp_path))

        assert len(tasks) == 2
        assert all(isinstance(t, EvalTask) for t in tasks)
        assert tasks[0].task_id == "phys_001"
        assert tasks[1].task_id == "phys_002"
        assert tasks[0].scoring == "exact"
        assert tasks[1].requires_uncertainty is True

    def test_load_tasks_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory should return an empty list."""
        tasks = load_tasks(str(tmp_path))
        assert tasks == []

    def test_load_tasks_missing_dir(self) -> None:
        """Missing directory should return an empty list."""
        tasks = load_tasks("/nonexistent/path")
        assert tasks == []


class TestEvalResultSchema:
    """Structural checks on EvalResult."""

    def test_eval_result_schema(self) -> None:
        """Create an EvalResult, verify all fields are present and typed."""
        result = _sample_eval_result()
        assert result.run_id == "run_1"
        assert result.task_id == "phys_001"
        assert isinstance(result.score, float)
        assert isinstance(result.latency_ms, int)
        assert isinstance(result.token_count, int)
        assert result.judge_explanation is None
        assert result.observer_gate == "act"


class TestDuckDBRoundTrip:
    """Round-trip storage of results in DuckDB."""

    def test_store_and_retrieve_results(self, tmp_path: Path) -> None:
        """Write results to DuckDB via store_results, read them back."""
        from src.eval.harness import store_results

        db_path = str(tmp_path / "test.duckdb")
        results = [
            _sample_eval_result("phys_001", "run_1"),
            _sample_eval_result("phys_002", "run_1"),
        ]

        store_results(results, db_path)

        db = duckdb.connect(db_path)
        rows = db.execute(
            "SELECT task_id, run_id, score FROM eval_results WHERE run_id = 'run_1'"
        ).fetchall()
        db.close()

        assert len(rows) == 2
        task_ids = {r[0] for r in rows}
        assert task_ids == {"phys_001", "phys_002"}


class TestScoring:
    """Scoring functions: exact match and multi-match."""

    def test_exact_match_scoring(self) -> None:
        """'yes' matches 'yes' (case insensitive)."""
        assert exact_match("yes", "yes") == 1.0
        assert exact_match("Yes", "yes") == 1.0
        assert exact_match("YES", "yes") == 1.0
        assert exact_match("no", "yes") == 0.0

    def test_multi_match_scoring(self) -> None:
        """Verify fraction of expected items found."""
        response = "The cup can be filled with water and used to drink."
        expected = ["fill", "drink", "pour", "stack"]

        score = multi_match(response, expected)
        # "fill" and "drink" are present, "pour" and "stack" are not.
        assert score == pytest.approx(2.0 / 4.0)

        # All items present.
        full_response = "fill, drink, pour, and stack them up"
        assert multi_match(full_response, expected) == pytest.approx(1.0)

    def test_multi_match_empty_list(self) -> None:
        """Empty expected list should return 1.0 (vacuously true)."""
        assert multi_match("anything", []) == pytest.approx(1.0)

    def test_multi_match_underscore_splitting(self) -> None:
        """Compound items like 'fill_with_liquid' match when all words appear."""
        response = "You can fill it with liquid and drink from the mug."
        expected = ["fill_with_liquid", "drink_from", "burn_hand"]

        score = multi_match(response, expected)
        # "fill_with_liquid" -> "fill","with","liquid" all present
        # "drink_from" -> "drink","from" all present
        # "burn_hand" -> "burn" not present
        assert score == pytest.approx(2.0 / 3.0)
