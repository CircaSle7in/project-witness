-- Staging view: clean and type eval results for downstream analysis
CREATE OR REPLACE VIEW stg_eval_results AS
SELECT
    run_id,
    timestamp,
    task_id,
    category,
    subcategory,
    model_name,
    model_response,
    expected,
    score,
    scoring_method,
    judge_explanation,
    raw_confidence,
    observer_gate,
    observer_confidence,
    latency_ms,
    token_count,
    CASE WHEN score >= 0.5 THEN true ELSE false END AS is_correct,
    CASE WHEN observer_gate IS NOT NULL THEN true ELSE false END AS has_observer
FROM eval_results;
