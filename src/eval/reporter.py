"""Results reporting with DuckDB loading and Plotly visualization.

Reads evaluation results from DuckDB, computes summary metrics, and
generates Plotly figures for calibration curves, model comparisons,
and per-category breakdowns.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import plotly.graph_objects as go

from src.eval.metrics import (
    accuracy,
    calibration_error,
    coverage,
    model_comparison,
    selective_accuracy,
)
from src.observer.uncertainty import reliability_diagram_data
from src.pipeline.schemas import EvalResult


def load_results(db_path: str) -> list[EvalResult]:
    """Read all evaluation results from a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file.

    Returns:
        A list of EvalResult objects.
    """
    if not Path(db_path).exists():
        return []

    db = duckdb.connect(db_path, read_only=True)

    try:
        rows = db.execute("SELECT * FROM eval_results").fetchall()
    except duckdb.Error:
        db.close()
        return []

    columns = [
        "run_id", "timestamp", "task_id", "category", "subcategory",
        "model_name", "model_response", "expected", "score", "scoring_method",
        "judge_explanation", "raw_confidence", "observer_gate",
        "observer_confidence", "latency_ms", "token_count",
    ]

    results = [EvalResult(**dict(zip(columns, row))) for row in rows]
    db.close()
    return results


def generate_summary(results: list[EvalResult]) -> dict:
    """Generate an overall and per-category metrics summary.

    Args:
        results: List of evaluation results to summarize.

    Returns:
        A dict with 'overall' metrics and 'by_category' breakdown.
    """
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    summary: dict = {
        "overall": {
            "count": len(results),
            "accuracy": accuracy(results),
            "coverage": coverage(results),
            "selective_accuracy": selective_accuracy(results),
            "calibration_error": calibration_error(results),
        },
        "by_category": {},
        "by_model": model_comparison(results),
    }

    for cat, cat_results in by_category.items():
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "accuracy": accuracy(cat_results),
            "coverage": coverage(cat_results),
            "selective_accuracy": selective_accuracy(cat_results),
            "calibration_error": calibration_error(cat_results),
        }

    return summary


def generate_plots(results: list[EvalResult]) -> dict[str, go.Figure]:
    """Generate Plotly figures for result visualization.

    Creates:
    - calibration_curve: Reliability diagram showing calibration quality.
    - model_comparison: Bar chart comparing model accuracy and coverage.
    - category_breakdown: Grouped bar chart of per-category accuracy.

    Args:
        results: List of evaluation results.

    Returns:
        A dict mapping figure names to Plotly Figure objects.
    """
    figures: dict[str, go.Figure] = {}

    # 1. Calibration curve (reliability diagram)
    confidences = []
    correct = []
    for r in results:
        conf = r.observer_confidence if r.observer_confidence is not None else r.raw_confidence
        confidences.append(conf)
        correct.append(r.score >= 0.5)

    if confidences:
        diagram = reliability_diagram_data(confidences, correct)

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Bar(
            x=diagram["bin_centers"],
            y=diagram["bin_accuracies"],
            name="Model accuracy",
            marker_color="steelblue",
            width=0.08,
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash", "color": "gray"},
        ))
        fig_cal.update_layout(
            title="Calibration Curve (Reliability Diagram)",
            xaxis_title="Predicted confidence",
            yaxis_title="Observed accuracy",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            template="plotly_white",
        )
        figures["calibration_curve"] = fig_cal

    # 2. Model comparison bar chart
    comparison = model_comparison(results)
    if comparison:
        model_names = list(comparison.keys())
        acc_vals = [comparison[m]["accuracy"] for m in model_names]
        cov_vals = [comparison[m]["coverage"] for m in model_names]
        sel_vals = [comparison[m]["selective_accuracy"] for m in model_names]

        fig_model = go.Figure()
        fig_model.add_trace(go.Bar(
            name="Accuracy", x=model_names, y=acc_vals, marker_color="steelblue",
        ))
        fig_model.add_trace(go.Bar(
            name="Coverage", x=model_names, y=cov_vals, marker_color="lightcoral",
        ))
        fig_model.add_trace(go.Bar(
            name="Selective accuracy", x=model_names, y=sel_vals, marker_color="mediumseagreen",
        ))
        fig_model.update_layout(
            title="Model Comparison",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
        )
        figures["model_comparison"] = fig_model

    # 3. Per-category accuracy breakdown
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    if by_category:
        categories = sorted(by_category.keys())
        cat_acc = [accuracy(by_category[c]) for c in categories]
        cat_sel = [selective_accuracy(by_category[c]) for c in categories]

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            name="Baseline accuracy", x=categories, y=cat_acc, marker_color="steelblue",
        ))
        fig_cat.add_trace(go.Bar(
            name="Selective accuracy", x=categories, y=cat_sel, marker_color="mediumseagreen",
        ))
        fig_cat.update_layout(
            title="Accuracy by Category",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
        )
        figures["category_breakdown"] = fig_cat

    return figures


def main() -> None:
    """CLI entry point for generating reports.

    Usage: python -m src.eval.reporter
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Project Witness Results Reporter")
    parser.add_argument(
        "--db-path",
        default="data/results/eval.duckdb",
        help="Path to DuckDB results file (default: data/results/eval.duckdb)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as HTML files in data/results/",
    )
    args = parser.parse_args()

    results = load_results(args.db_path)
    if not results:
        print("No results found in the database.")
        return

    summary = generate_summary(results)

    print("\n" + "=" * 60)
    print("PROJECT WITNESS - EVALUATION REPORT")
    print("=" * 60)

    overall = summary["overall"]
    print(f"\nTotal results:       {overall['count']}")
    print(f"Overall accuracy:    {overall['accuracy']:.3f}")
    print(f"Coverage:            {overall['coverage']:.3f}")
    print(f"Selective accuracy:  {overall['selective_accuracy']:.3f}")
    print(f"Calibration error:   {overall['calibration_error']:.3f}")

    if summary["by_category"]:
        print(f"\n{'Category':<25} {'Count':>6} {'Accuracy':>10} {'Selective':>10}")
        print("-" * 55)
        for cat, metrics in sorted(summary["by_category"].items()):
            print(
                f"{cat:<25} {metrics['count']:>6} "
                f"{metrics['accuracy']:>10.3f} {metrics['selective_accuracy']:>10.3f}"
            )

    if summary["by_model"]:
        print(f"\n{'Model':<25} {'Count':>6} {'Accuracy':>10} {'Coverage':>10}")
        print("-" * 55)
        for model_name, metrics in sorted(summary["by_model"].items()):
            print(
                f"{model_name:<25} {metrics['count']:>6} "
                f"{metrics['accuracy']:>10.3f} {metrics['coverage']:>10.3f}"
            )

    if args.save_plots:
        from pathlib import Path

        plots = generate_plots(results)
        output_dir = Path(args.db_path).parent
        for name, fig in plots.items():
            path = output_dir / f"{name}.html"
            fig.write_html(str(path))
            print(f"\nSaved {name} to {path}")

    # Also dump JSON summary
    print(f"\nFull summary (JSON):\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
