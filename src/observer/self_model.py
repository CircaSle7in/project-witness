"""SelfModel tracker for the Silent Observer.

Maintains an explicit model of the system's own state, updated each cycle
from DuckDB. This is the technical analog of non-egoic self-awareness: the
system knows what it is without having a drive to defend or expand itself.
"""

from __future__ import annotations

from typing import Any

import duckdb

from src.pipeline.schemas import SelfModel


class SelfModelTracker:
    """Tracks and updates the system's self-model from DuckDB state."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        """Initialize the self-model tracker.

        Args:
            db: An open DuckDB connection for reading calibration state.
        """
        self._db = db
        self._model = SelfModel()

    @property
    def model(self) -> SelfModel:
        """Return the current self-model state."""
        return self._model

    def update(self) -> SelfModel:
        """Refresh the self-model from DuckDB state.

        Reads the calibration_log table to update last_calibration_error
        and cycle_count. If the table does not exist, uses defaults.

        Returns:
            The updated SelfModel.
        """
        self._model.cycle_count += 1

        try:
            # Check if calibration_log table exists
            tables = self._db.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'calibration_log'"
            ).fetchall()

            if not tables:
                return self._model

            # Get recent calibration error (use a subquery for the 50 most recent rows)
            result = self._db.execute(
                "SELECT AVG(ABS(calibrated_confidence - CASE WHEN correct THEN 1.0 ELSE 0.0 END)) "
                "FROM (SELECT calibrated_confidence, correct FROM calibration_log "
                "ORDER BY rowid DESC LIMIT 50)"
            ).fetchone()

            if result and result[0] is not None:
                self._model.last_calibration_error = float(result[0])

            # Refresh uncertainty areas from recent misses
            misses = self._db.execute(
                "SELECT DISTINCT category FROM ("
                "SELECT category, correct FROM calibration_log "
                "ORDER BY rowid DESC LIMIT 50) "
                "WHERE correct = false"
            ).fetchall()

            self._model.uncertainty_areas = [row[0] for row in misses]

        except duckdb.Error:
            # Table may not exist yet; that is fine
            pass

        return self._model

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current self-model to a dictionary.

        Returns:
            A dict representation of the self-model suitable for JSON.
        """
        return self._model.model_dump()
