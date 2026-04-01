-- Model comparison: aggregate metrics per model and category
CREATE OR REPLACE VIEW model_comparison AS
SELECT
    model_name,
    category,
    COUNT(*) AS total_tasks,
    AVG(score) AS avg_score,
    SUM(CASE WHEN score >= 0.5 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS accuracy,
    AVG(raw_confidence) AS avg_raw_confidence,
    AVG(latency_ms) AS avg_latency_ms,
    AVG(token_count) AS avg_tokens
FROM stg_eval_results
GROUP BY model_name, category
ORDER BY model_name, category;
