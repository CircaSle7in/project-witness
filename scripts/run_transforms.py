"""Run dbt-style SQL transforms against the eval results DuckDB.

Executes SQL files in transforms/staging/ then transforms/marts/
to create analytical views on top of raw eval_results data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def run_transforms(db_path: str) -> None:
    """Execute all SQL transform files against the given DuckDB database.

    Runs staging transforms first, then marts, since marts depend on staging views.

    Args:
        db_path: Path to the DuckDB database file.
    """
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    db = duckdb.connect(db_path)
    transforms_dir = Path("transforms")

    # Run in order: staging first, then marts
    for stage in ["staging", "marts"]:
        stage_dir = transforms_dir / stage
        if not stage_dir.exists():
            continue

        for sql_file in sorted(stage_dir.glob("*.sql")):
            print(f"Running {sql_file}...")
            sql = sql_file.read_text()
            try:
                db.execute(sql)
                print(f"  OK: {sql_file.name}")
            except duckdb.Error as e:
                print(f"  ERROR in {sql_file.name}: {e}")

    # Verify views were created
    print("\nCreated views:")
    views = db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type = 'VIEW' ORDER BY table_name"
    ).fetchall()
    for (view_name,) in views:
        count = db.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
        print(f"  {view_name}: {count} rows")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQL transforms")
    parser.add_argument(
        "--db-path",
        default="data/results/eval.duckdb",
        help="Path to DuckDB results file",
    )
    args = parser.parse_args()
    run_transforms(args.db_path)
