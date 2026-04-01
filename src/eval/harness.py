"""Evaluation harness for running benchmark tasks against models.

Loads tasks from YAML configs, runs them with and without the Silent Observer,
stores results in DuckDB, and provides a CLI entry point.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import yaml

from src.eval.judge import score_task
from src.observer.observer import SilentObserver
from src.pipeline.schemas import CalibrationEntry, EvalResult, EvalTask

if TYPE_CHECKING:
    from src.models.base import BaseModel


def load_tasks(config_dir: str) -> list[EvalTask]:
    """Load all evaluation tasks from YAML files in the given directory.

    Each YAML file can contain a single task dict or a list of task dicts
    under a top-level 'tasks' key.

    Args:
        config_dir: Path to the directory containing benchmark YAML files.

    Returns:
        A list of validated EvalTask objects.
    """
    tasks: list[EvalTask] = []
    config_path = Path(config_dir)

    if not config_path.exists():
        return tasks

    for yaml_file in sorted(config_path.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if data is None:
            continue

        if isinstance(data, dict) and "tasks" in data:
            raw_tasks = data["tasks"]
        elif isinstance(data, list):
            raw_tasks = data
        elif isinstance(data, dict):
            raw_tasks = [data]
        else:
            continue

        for raw in raw_tasks:
            tasks.append(EvalTask(**raw))

    return tasks


def _ensure_tables(db: duckdb.DuckDBPyConnection) -> None:
    """Create DuckDB tables if they do not exist.

    Args:
        db: An open DuckDB connection.
    """
    db.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id VARCHAR,
            timestamp VARCHAR,
            task_id VARCHAR,
            category VARCHAR,
            subcategory VARCHAR,
            model_name VARCHAR,
            model_response VARCHAR,
            expected VARCHAR,
            score DOUBLE,
            scoring_method VARCHAR,
            judge_explanation VARCHAR,
            raw_confidence DOUBLE,
            observer_gate VARCHAR,
            observer_confidence DOUBLE,
            latency_ms INTEGER,
            token_count INTEGER
        )
    """)

    db.execute("""
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


async def run_baseline(
    tasks: list[EvalTask],
    model: BaseModel,
) -> list[EvalResult]:
    """Run all tasks without the observer (baseline evaluation).

    Args:
        tasks: List of evaluation tasks to run.
        model: The model wrapper to query.

    Returns:
        A list of EvalResult objects with no observer gating.
    """
    run_id = str(uuid.uuid4())[:8]
    results: list[EvalResult] = []

    for task in tasks:
        start = time.monotonic()

        response, confidence = await model.query(
            prompt=task.prompt,
            image_path=task.image,
            video_path=task.video,
        )

        latency_ms = int((time.monotonic() - start) * 1000)

        score, explanation = await score_task(task, response, model)

        result = EvalResult(
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            task_id=task.task_id,
            category=task.category,
            subcategory=task.subcategory,
            model_name=model.name,
            model_response=response,
            expected=task.expected,
            score=score,
            scoring_method=task.scoring,
            judge_explanation=explanation,
            raw_confidence=confidence,
            observer_gate=None,
            observer_confidence=None,
            latency_ms=latency_ms,
            token_count=len(response.split()),
        )
        results.append(result)

    return results


async def run_observed(
    tasks: list[EvalTask],
    model: BaseModel,
    observer: SilentObserver,
) -> list[EvalResult]:
    """Run all tasks with the Silent Observer gating decisions.

    The observer assesses each model response and may gate it. The gate
    decision and calibrated confidence are recorded in the result.

    Args:
        tasks: List of evaluation tasks to run.
        model: The model wrapper to query.
        observer: The SilentObserver instance.

    Returns:
        A list of EvalResult objects with observer metadata populated.
    """
    run_id = str(uuid.uuid4())[:8]
    results: list[EvalResult] = []

    for task in tasks:
        start = time.monotonic()

        response, confidence = await model.query(
            prompt=task.prompt,
            image_path=task.image,
            video_path=task.video,
        )

        latency_ms = int((time.monotonic() - start) * 1000)

        # Observer assessment
        assessment = observer.assess(
            proposed_action=response,
            world_state={},
            model_confidence=confidence,
            belief_state={},
            category=task.category,
        )

        score, explanation = await score_task(task, response, model)

        result = EvalResult(
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            task_id=task.task_id,
            category=task.category,
            subcategory=task.subcategory,
            model_name=model.name,
            model_response=response,
            expected=task.expected,
            score=score,
            scoring_method=task.scoring,
            judge_explanation=explanation,
            raw_confidence=confidence,
            observer_gate=assessment.gate.value,
            observer_confidence=assessment.confidence,
            latency_ms=latency_ms,
            token_count=len(response.split()),
        )
        results.append(result)

        # Log calibration data
        calibration_entry = CalibrationEntry(
            task_id=task.task_id,
            category=task.category,
            model_name=model.name,
            raw_confidence=confidence,
            calibrated_confidence=assessment.confidence,
            prediction=response[:500],
            ground_truth=task.expected[:500],
            correct=score >= 0.5,
            observer_gate=assessment.gate.value,
        )
        observer.log_calibration(calibration_entry)

    return results


def store_results(results: list[EvalResult], db_path: str) -> None:
    """Write evaluation results to a DuckDB database.

    Args:
        results: List of EvalResult objects to persist.
        db_path: Path to the DuckDB database file.
    """
    db = duckdb.connect(db_path)
    _ensure_tables(db)

    for r in results:
        db.execute(
            "INSERT INTO eval_results VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r.run_id,
                r.timestamp,
                r.task_id,
                r.category,
                r.subcategory,
                r.model_name,
                r.model_response,
                r.expected,
                r.score,
                r.scoring_method,
                r.judge_explanation,
                r.raw_confidence,
                r.observer_gate,
                r.observer_confidence,
                r.latency_ms,
                r.token_count,
            ],
        )

    db.close()


def main() -> None:
    """CLI entry point for running the evaluation harness.

    Usage: python -m src.eval.harness
    """
    import argparse

    parser = argparse.ArgumentParser(description="Project Witness Evaluation Harness")
    parser.add_argument(
        "--config-dir",
        default="configs/benchmarks",
        help="Directory containing benchmark YAML files (default: configs/benchmarks)",
    )
    parser.add_argument(
        "--db-path",
        default="data/results/eval.duckdb",
        help="Path to DuckDB results file (default: data/results/eval.duckdb)",
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "qwen", "both"],
        default="gemini",
        help="Which model to evaluate (default: gemini)",
    )
    parser.add_argument(
        "--observed",
        action="store_true",
        help="Run with observer gating (default: baseline only)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_tasks(args.config_dir)
    if not tasks:
        print(f"No tasks found in {args.config_dir}")
        return

    print(f"Loaded {len(tasks)} evaluation tasks")

    # Build models
    models: list[BaseModel] = []

    if args.model in ("gemini", "both"):
        from src.models.gemini import GeminiModel

        models.append(GeminiModel())

    if args.model in ("qwen", "both"):
        from src.models.qwen_vl import QwenVLModel

        models.append(QwenVLModel())

    # Run evaluation
    all_results: list[EvalResult] = []

    for model in models:
        print(f"\nEvaluating {model.name}...")

        if args.observed:
            db = duckdb.connect(args.db_path)
            _ensure_tables(db)
            observer = SilentObserver(db)
            results = asyncio.run(run_observed(tasks, model, observer))
            db.close()
        else:
            results = asyncio.run(run_baseline(tasks, model))

        all_results.extend(results)
        print(f"  Completed {len(results)} tasks for {model.name}")

    # Store results
    store_results(all_results, args.db_path)
    print(f"\nResults stored in {args.db_path}")

    # Print summary
    from src.eval.metrics import accuracy as calc_accuracy
    from src.eval.metrics import coverage as calc_coverage
    from src.eval.metrics import selective_accuracy as calc_selective

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total tasks:         {len(all_results)}")
    print(f"Accuracy:            {calc_accuracy(all_results):.3f}")
    print(f"Coverage:            {calc_coverage(all_results):.3f}")
    print(f"Selective accuracy:  {calc_selective(all_results):.3f}")


if __name__ == "__main__":
    main()
