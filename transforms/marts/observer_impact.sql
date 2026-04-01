-- Observer impact: compare baseline vs observer-gated metrics
CREATE OR REPLACE VIEW observer_impact AS
WITH baseline AS (
    SELECT
        model_name,
        category,
        COUNT(*) AS total,
        AVG(score) AS accuracy,
        1.0 AS coverage
    FROM stg_eval_results
    WHERE NOT has_observer
    GROUP BY model_name, category
),
observed AS (
    SELECT
        model_name,
        category,
        COUNT(*) AS total,
        AVG(score) AS accuracy,
        SUM(CASE WHEN observer_gate = 'act' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS coverage,
        AVG(CASE WHEN observer_gate = 'act' THEN score ELSE NULL END) AS selective_accuracy
    FROM stg_eval_results
    WHERE has_observer
    GROUP BY model_name, category
)
SELECT
    COALESCE(b.model_name, o.model_name) AS model_name,
    COALESCE(b.category, o.category) AS category,
    b.accuracy AS baseline_accuracy,
    o.accuracy AS observed_accuracy,
    o.selective_accuracy,
    o.coverage,
    o.selective_accuracy - b.accuracy AS accuracy_lift
FROM baseline b
FULL OUTER JOIN observed o
    ON b.model_name = o.model_name AND b.category = o.category
ORDER BY model_name, category;
