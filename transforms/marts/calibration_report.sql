-- Calibration report: compare stated confidence vs actual accuracy in bins
CREATE OR REPLACE VIEW calibration_report AS
WITH binned AS (
    SELECT
        model_name,
        category,
        FLOOR(raw_confidence * 10) / 10.0 AS confidence_bin,
        score >= 0.5 AS is_correct
    FROM stg_eval_results
    WHERE raw_confidence IS NOT NULL
)
SELECT
    model_name,
    category,
    confidence_bin,
    COUNT(*) AS bin_count,
    AVG(is_correct::INT) AS actual_accuracy,
    confidence_bin + 0.05 AS bin_center,
    ABS(AVG(is_correct::INT) - (confidence_bin + 0.05)) AS calibration_gap
FROM binned
GROUP BY model_name, category, confidence_bin
ORDER BY model_name, category, confidence_bin;
