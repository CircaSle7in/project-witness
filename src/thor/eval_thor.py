"""v0.5 THOR evaluation runner.

Runs rearrangement tasks in baseline (no observer) and observed (observer
gates actions) modes, stores results in DuckDB, and compares performance.

Handles the case where AI2-THOR cannot launch gracefully: logs the error
and returns empty results instead of crashing.

Usage:
    python -m src.thor.eval_thor
    python -m src.thor.eval_thor --mode baseline
    python -m src.thor.eval_thor --mode observed
    python -m src.thor.eval_thor --tasks 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import duckdb

from src.thor.schemas import TaskDefinition, TaskResult
from src.thor.tasks import get_rearrangement_tasks

logger = logging.getLogger(__name__)

# Default DuckDB path for THOR evaluation results.
_DEFAULT_DB_PATH = "data/results/thor_eval.duckdb"


async def run_thor_eval(
    tasks: list[TaskDefinition],
    model_name: str = "gemini",
    mode: str = "both",
    db_path: str = _DEFAULT_DB_PATH,
) -> dict:
    """Run THOR evaluation in baseline and/or observed mode.

    For each task, creates a controller for the scene, a planner with the
    selected model, and runs the task in baseline and/or observed mode.
    Results are stored in DuckDB.

    If AI2-THOR cannot launch, logs the error and returns empty results.

    Args:
        tasks: List of rearrangement tasks to run.
        model_name: Which model to use for planning ("gemini" or "qwen").
        mode: Which modes to run: "baseline", "observed", or "both".
        db_path: Path to the DuckDB file for storing results.

    Returns:
        Dict with "baseline_results" and/or "observed_results" lists,
        plus a "comparison" dict if both modes were run.
    """
    # Lazy imports to allow the module to load even without ai2thor or API keys
    model = _create_model(model_name)
    if model is None:
        logger.error(
            "Could not create model '%s'. Check API keys.", model_name
        )
        return {"error": f"Could not create model '{model_name}'."}

    results: dict = {}
    baseline_results: list[TaskResult] = []
    observed_results: list[TaskResult] = []

    run_baseline = mode in ("baseline", "both")
    run_observed = mode in ("observed", "both")

    for i, task in enumerate(tasks):
        logger.info(
            "Task %d/%d: %s (scene=%s)",
            i + 1, len(tasks), task.task_id, task.scene_name,
        )

        # --- Baseline run ---
        if run_baseline:
            result = await _run_single_task(
                task, model, observer_active=False
            )
            if result is not None:
                baseline_results.append(result)
                logger.info(
                    "  Baseline: success=%s, steps=%d",
                    result.success, result.steps_taken,
                )
            else:
                logger.warning("  Baseline: skipped (THOR launch failed)")

        # --- Observed run ---
        if run_observed:
            result = await _run_single_task(
                task, model, observer_active=True
            )
            if result is not None:
                observed_results.append(result)
                logger.info(
                    "  Observed: success=%s, steps=%d, gated=%d",
                    result.success, result.steps_taken, result.actions_gated,
                )
            else:
                logger.warning("  Observed: skipped (THOR launch failed)")

    # Store results
    if baseline_results:
        store_thor_results(baseline_results, db_path, "baseline")
        results["baseline_results"] = baseline_results

    if observed_results:
        store_thor_results(observed_results, db_path, "observed")
        results["observed_results"] = observed_results

    # Compare if both modes ran
    if baseline_results and observed_results:
        results["comparison"] = compare_modes(baseline_results, observed_results)

    return results


async def _run_single_task(
    task: TaskDefinition,
    model: object,
    observer_active: bool,
) -> TaskResult | None:
    """Run a single task with or without the observer.

    Creates a fresh THOR controller, planner, and optionally an observer
    for each run. Handles THOR launch failures gracefully.

    Args:
        task: The task to run.
        model: The LLM model instance for planning.
        observer_active: Whether to use the observer.

    Returns:
        A TaskResult, or None if THOR could not launch.
    """
    from src.thor.agent import WitnessAgent
    from src.thor.controller import THORLaunchError, WitnessController
    from src.thor.planner import ActionPlanner

    controller: WitnessController | None = None
    db: duckdb.DuckDBPyConnection | None = None

    try:
        controller = WitnessController(
            scene=task.scene_name, headless=True
        )
    except THORLaunchError as exc:
        logger.warning(
            "Cannot launch THOR for scene %s: %s", task.scene_name, exc
        )
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected error launching THOR for scene %s: %s",
            task.scene_name, exc,
        )
        return None

    try:
        planner = ActionPlanner(model)

        observer = None
        if observer_active:
            from src.observer.observer import SilentObserver

            db = duckdb.connect(":memory:")
            observer = SilentObserver(db=db)

        agent = WitnessAgent(
            controller=controller,
            planner=planner,
            observer=observer,
        )

        result = await agent.run_task(task)
        return result

    except Exception as exc:
        logger.error(
            "Error running task %s: %s", task.task_id, exc, exc_info=True
        )
        return None
    finally:
        if controller is not None:
            controller.stop()
        if db is not None:
            db.close()


def _create_model(model_name: str) -> object | None:
    """Create a model instance by name, or None if credentials are missing.

    Args:
        model_name: "gemini" or "qwen".

    Returns:
        A model instance, or None on failure.
    """
    try:
        if model_name == "gemini":
            from src.models.gemini import GeminiModel
            return GeminiModel()
        elif model_name == "qwen":
            from src.models.qwen_vl import QwenModel
            return QwenModel()
        else:
            logger.error("Unknown model name: %s", model_name)
            return None
    except (ValueError, ImportError) as exc:
        logger.warning("Could not create model '%s': %s", model_name, exc)
        return None


def store_thor_results(
    results: list[TaskResult],
    db_path: str,
    mode: str,
) -> None:
    """Store THOR results in DuckDB.

    Creates the thor_results table if it does not exist, then inserts
    each result as a row.

    Args:
        results: List of TaskResult objects to store.
        db_path: Path to the DuckDB database file.
        mode: Run mode label ("baseline" or "observed").
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(db_path)

    try:
        db.execute("""
            CREATE TABLE IF NOT EXISTS thor_results (
                task_id VARCHAR,
                scene_name VARCHAR,
                mode VARCHAR,
                success BOOLEAN,
                steps_taken INTEGER,
                max_steps INTEGER,
                total_actions_proposed INTEGER,
                actions_executed INTEGER,
                actions_gated INTEGER,
                observer_active BOOLEAN,
                action_log VARCHAR,
                gate_distribution VARCHAR,
                mean_confidence DOUBLE,
                mean_prediction_trust DOUBLE,
                completion_time_s DOUBLE
            )
        """)

        for r in results:
            db.execute(
                """
                INSERT INTO thor_results VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    r.task_id,
                    r.scene_name,
                    mode,
                    r.success,
                    r.steps_taken,
                    r.max_steps,
                    r.total_actions_proposed,
                    r.actions_executed,
                    r.actions_gated,
                    r.observer_active,
                    json.dumps(r.action_log),
                    json.dumps(r.gate_distribution),
                    r.mean_confidence,
                    r.mean_prediction_trust,
                    r.completion_time_s,
                ],
            )
        logger.info(
            "Stored %d %s results to %s", len(results), mode, db_path
        )
    finally:
        db.close()


def compare_modes(
    baseline: list[TaskResult],
    observed: list[TaskResult],
) -> dict:
    """Compare baseline vs observed results.

    Computes success rates, average steps, wasted actions, and
    observer gate distribution across all tasks.

    Args:
        baseline: Results from baseline (no observer) runs.
        observed: Results from observed (observer active) runs.

    Returns:
        A dict with comparison metrics including improvement percentages.
    """
    def _success_rate(results: list[TaskResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)

    def _avg_steps(results: list[TaskResult]) -> float:
        if not results:
            return 0.0
        return sum(r.steps_taken for r in results) / len(results)

    def _wasted_actions(results: list[TaskResult]) -> float:
        """Count actions that failed (wasted effort) per task."""
        if not results:
            return 0.0
        total_wasted = 0
        for r in results:
            for entry in r.action_log:
                if entry.get("executed") and not entry.get("success", True):
                    total_wasted += 1
        return total_wasted / len(results)

    baseline_sr = _success_rate(baseline)
    observed_sr = _success_rate(observed)

    # Aggregate observer gate distribution from observed runs
    gate_dist: dict[str, int] = {}
    for r in observed:
        for gate_name, count in r.gate_distribution.items():
            gate_dist[gate_name] = gate_dist.get(gate_name, 0) + count

    improvement = 0.0
    if baseline_sr > 0:
        improvement = ((observed_sr - baseline_sr) / baseline_sr) * 100.0

    comparison = {
        "baseline_success_rate": round(baseline_sr, 4),
        "observed_success_rate": round(observed_sr, 4),
        "baseline_avg_steps": round(_avg_steps(baseline), 2),
        "observed_avg_steps": round(_avg_steps(observed), 2),
        "baseline_wasted_actions": round(_wasted_actions(baseline), 2),
        "observed_wasted_actions": round(_wasted_actions(observed), 2),
        "observer_gate_distribution": gate_dist,
        "improvement_pct": round(improvement, 2),
        "baseline_tasks": len(baseline),
        "observed_tasks": len(observed),
    }

    return comparison


def _print_comparison(comparison: dict) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        comparison: The comparison dict from compare_modes().
    """
    print("\n" + "=" * 60)
    print("  THOR Assessment: Baseline vs Observer")
    print("=" * 60)
    print(
        f"  Tasks assessed:         "
        f"{comparison['baseline_tasks']} baseline, "
        f"{comparison['observed_tasks']} observed"
    )
    print(
        f"  Baseline success rate:  "
        f"{comparison['baseline_success_rate']:.1%}"
    )
    print(
        f"  Observed success rate:  "
        f"{comparison['observed_success_rate']:.1%}"
    )
    print(
        f"  Improvement:            "
        f"{comparison['improvement_pct']:+.1f}%"
    )
    print(f"  Baseline avg steps:     {comparison['baseline_avg_steps']:.1f}")
    print(f"  Observed avg steps:     {comparison['observed_avg_steps']:.1f}")
    print(
        f"  Baseline wasted/task:   "
        f"{comparison['baseline_wasted_actions']:.1f}"
    )
    print(
        f"  Observed wasted/task:   "
        f"{comparison['observed_wasted_actions']:.1f}"
    )

    gate_dist = comparison.get("observer_gate_distribution", {})
    if gate_dist:
        print("\n  Observer Gate Distribution:")
        for gate_name, count in sorted(gate_dist.items()):
            print(f"    {gate_name:>16s}: {count}")

    print("=" * 60 + "\n")


def main() -> None:
    """CLI entry point: python -m src.thor.eval_thor

    Parses command-line arguments and runs the THOR assessment. Prints
    a comparison table if both modes are run.
    """
    parser = argparse.ArgumentParser(
        description="Run THOR rearrangement assessment (v0.5)."
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "qwen"],
        default="gemini",
        help="Which model to use for planning (default: gemini).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "observed", "both"],
        default="both",
        help="Run mode: baseline only, observed only, or both (default: both).",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=0,
        help="Number of tasks to run (0 = all, default: all).",
    )
    parser.add_argument(
        "--db-path",
        default=_DEFAULT_DB_PATH,
        help=f"DuckDB output path (default: {_DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    tasks = get_rearrangement_tasks()
    if args.tasks > 0:
        tasks = tasks[: args.tasks]

    logger.info(
        "Running THOR assessment: model=%s, mode=%s, tasks=%d",
        args.model, args.mode, len(tasks),
    )

    results = asyncio.run(
        run_thor_eval(
            tasks=tasks,
            model_name=args.model,
            mode=args.mode,
            db_path=args.db_path,
        )
    )

    if "error" in results:
        print(f"\nError: {results['error']}")
        sys.exit(1)

    # Print summary
    if "comparison" in results:
        _print_comparison(results["comparison"])
    else:
        # Single mode: print basic stats
        for mode_key in ("baseline_results", "observed_results"):
            mode_results = results.get(mode_key, [])
            if mode_results:
                mode_label = mode_key.replace("_results", "")
                successes = sum(1 for r in mode_results if r.success)
                print(
                    f"\n{mode_label.capitalize()}: "
                    f"{successes}/{len(mode_results)} tasks succeeded"
                )

    # Store path info
    db_path = args.db_path
    if any(results.get(k) for k in ("baseline_results", "observed_results")):
        print(f"Results stored in: {db_path}")


if __name__ == "__main__":
    main()
