"""
SQLite persistence for simulation runs -- standing in for the base
paper's PostgreSQL (metadata) + MinIO (VTK/HDF5 artifacts), which
need a running server. SQLite gives the same queryable run-history
guarantee the Database Agent needs (list_recent_runs,
get_run_status) without any external service.
"""
from __future__ import annotations
import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Optional


class RunStore:
    def __init__(self, db_path: str = "runs.db"):
        self.db_path = db_path
        # check_same_thread=False: LangGraph runs tools in a ThreadPoolExecutor;
        # our writes are sequential so SQLite's internal serialisation is sufficient.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                description TEXT,
                config_json TEXT,
                status TEXT,
                dofs INTEGER,
                solve_time_s REAL,
                t_min REAL,
                t_max REAL,
                t_mean REAL,
                warnings_json TEXT,
                created_at REAL
            )
        """)
        self._conn.commit()

    def create_run(self, description: str, config: dict) -> str:
        run_id = uuid.uuid4().hex[:8]
        self._conn.execute(
            "INSERT INTO runs (run_id, description, config_json, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, description, json.dumps(config), "running", time.time()),
        )
        self._conn.commit()
        return run_id

    def complete_run(self, run_id: str, dofs: int, solve_time_s: float,
                      t_min: float, t_max: float, t_mean: float, warnings: list):
        self._conn.execute(
            "UPDATE runs SET status=?, dofs=?, solve_time_s=?, t_min=?, t_max=?, "
            "t_mean=?, warnings_json=? WHERE run_id=?",
            ("completed", dofs, solve_time_s, t_min, t_max, t_mean,
             json.dumps(warnings), run_id),
        )
        self._conn.commit()

    def fail_run(self, run_id: str, error: str):
        self._conn.execute(
            "UPDATE runs SET status=?, warnings_json=? WHERE run_id=?",
            ("failed", json.dumps([error]), run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def list_recent(self, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def success_rate(self) -> float:
        cur = self._conn.execute("SELECT status FROM runs")
        rows = [r[0] for r in cur.fetchall()]
        if not rows:
            return 0.0
        return sum(1 for r in rows if r == "completed") / len(rows)
