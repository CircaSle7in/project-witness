"""Gradio dashboard for the Project Witness operator cockpit.

Local-first, read-only dashboard that displays evaluation results,
observer gate decisions, confidence scores, calibration plots, and
audit trail information from a DuckDB database.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from src.eval.reporter import generate_plots, generate_summary, load_results
from src.pipeline.schemas import EvalResult


def _build_task_choices(results: list[EvalResult]) -> list[str]:
    """Build dropdown choices from evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        A list of "task_id (model_name)" strings for the dropdown.
    """
    choices = []
    for r in results:
        label = f"{r.task_id} ({r.model_name})"
        if label not in choices:
            choices.append(label)
    return choices


def _find_result(
    results: list[EvalResult],
    selection: str,
) -> EvalResult | None:
    """Find a result matching the dropdown selection.

    Args:
        results: List of evaluation results.
        selection: The dropdown selection string.

    Returns:
        The matching EvalResult, or None if not found.
    """
    for r in results:
        label = f"{r.task_id} ({r.model_name})"
        if label == selection:
            return r
    return None


def _format_detail(result: EvalResult) -> str:
    """Format a single result into a readable detail string.

    Args:
        result: The evaluation result to format.

    Returns:
        A multi-line string with result details.
    """
    lines = [
        f"Task ID: {result.task_id}",
        f"Category: {result.category} / {result.subcategory}",
        f"Model: {result.model_name}",
        f"Score: {result.score:.3f} ({result.scoring_method})",
        "",
        "--- Model Response ---",
        result.model_response[:1000],
        "",
        "--- Expected ---",
        result.expected[:500],
    ]

    if result.judge_explanation:
        lines.extend(["", "--- Judge Explanation ---", result.judge_explanation])

    return "\n".join(lines)


def _format_observer(result: EvalResult) -> str:
    """Format observer metadata for a result.

    Args:
        result: The evaluation result with observer data.

    Returns:
        A multi-line string with observer details.
    """
    if result.observer_gate is None:
        return "No observer data (baseline run)"

    lines = [
        f"Gate Decision: {result.observer_gate.upper()}",
        f"Raw Confidence: {result.raw_confidence:.3f}",
        f"Observer Confidence: {result.observer_confidence:.3f}"
        if result.observer_confidence is not None
        else "Observer Confidence: N/A",
        f"Latency: {result.latency_ms}ms",
        f"Token Count: {result.token_count}",
    ]

    return "\n".join(lines)


def _format_audit_log(results: list[EvalResult]) -> str:
    """Format an audit log summary from all results.

    Args:
        results: List of all evaluation results.

    Returns:
        A multi-line string summarizing the audit trail.
    """
    lines = ["AUDIT LOG", "=" * 50, ""]

    for r in results:
        gate = r.observer_gate or "baseline"
        conf = (
            f"{r.observer_confidence:.3f}"
            if r.observer_confidence is not None
            else f"{r.raw_confidence:.3f}"
        )
        lines.append(
            f"[{r.timestamp[:19]}] {r.task_id} | {r.model_name} | "
            f"gate={gate} | conf={conf} | score={r.score:.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    """Launch the Gradio cockpit dashboard on localhost.

    Reads evaluation results from the default DuckDB path and builds
    an interactive dashboard with task selection, detail views,
    observer metadata, calibration plots, and an audit log.
    """
    db_path = "data/results/eval.duckdb"

    # Load results (empty list if DB does not exist)
    results: list[EvalResult] = []
    if Path(db_path).exists():
        results = load_results(db_path)

    task_choices = _build_task_choices(results) if results else ["No results loaded"]

    # Pre-generate plots
    plots: dict[str, go.Figure] = {}
    summary: dict = {}
    if results:
        plots = generate_plots(results)
        summary = generate_summary(results)

    def on_task_select(selection: str) -> tuple[str, str]:
        """Handle task dropdown selection.

        Args:
            selection: The selected task string.

        Returns:
            Tuple of (detail_text, observer_text).
        """
        if not results:
            return "No results loaded.", "No observer data."

        result = _find_result(results, selection)
        if result is None:
            return "Result not found.", "No observer data."

        return _format_detail(result), _format_observer(result)

    # Build the Gradio interface
    with gr.Blocks(title="Project Witness - Operator Cockpit") as app:
        gr.Markdown("# Project Witness - Operator Cockpit")
        gr.Markdown(
            "Read-only dashboard for evaluating Silent Observer performance. "
            "Select a task below to inspect model responses, observer gate decisions, "
            "and confidence scores."
        )

        if summary:
            overall = summary["overall"]
            gr.Markdown(
                f"**Results loaded:** {overall['count']} tasks | "
                f"**Accuracy:** {overall['accuracy']:.3f} | "
                f"**Coverage:** {overall['coverage']:.3f} | "
                f"**Selective accuracy:** {overall['selective_accuracy']:.3f}"
            )

        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=task_choices,
                label="Select Task",
                value=task_choices[0] if task_choices else None,
            )

        with gr.Row():
            with gr.Column():
                detail_box = gr.Textbox(
                    label="Task Detail",
                    lines=15,
                    interactive=False,
                )
            with gr.Column():
                observer_box = gr.Textbox(
                    label="Observer Assessment",
                    lines=15,
                    interactive=False,
                )

        # Calibration plot
        if "calibration_curve" in plots:
            gr.Markdown("## Calibration Curve")
            gr.Plot(value=plots["calibration_curve"])

        # Per-category accuracy
        if "category_breakdown" in plots:
            gr.Markdown("## Accuracy by Category")
            gr.Plot(value=plots["category_breakdown"])

        # Model comparison
        if "model_comparison" in plots:
            gr.Markdown("## Model Comparison")
            gr.Plot(value=plots["model_comparison"])

        # Audit log
        gr.Markdown("## Audit Log")
        audit_text = _format_audit_log(results) if results else "No audit data available."
        gr.Textbox(
            value=audit_text,
            label="Full Audit Trail",
            lines=20,
            interactive=False,
        )

        # Wire up the dropdown
        task_dropdown.change(
            fn=on_task_select,
            inputs=[task_dropdown],
            outputs=[detail_box, observer_box],
        )

        # Load initial selection
        if results and task_choices:
            initial_result = _find_result(results, task_choices[0])
            if initial_result:
                detail_box.value = _format_detail(initial_result)
                observer_box.value = _format_observer(initial_result)

    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
