"""SQLite experiment tracker.

One file (``tracking.db`` by default) holds every run: parameters, metric
histories, and artifact paths. The query side (``list_runs``, ``best_run``,
``metric_history``) is the surface papers pull real numbers from.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    topic       TEXT NOT NULL,
    model       TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS params (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key    TEXT NOT NULL,
    step   INTEGER,
    value  REAL NOT NULL,
    ts     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    kind   TEXT NOT NULL,
    path   TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tracker:
    def __init__(self, path: str | Path = "tracking.db"):
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def start_run(self, name: str, topic: str, model: str, dataset: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (name, topic, model, dataset, started_at) VALUES (?, ?, ?, ?, ?)",
            (name, topic, model, dataset, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_params(self, run_id: int, params: dict) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
            [(run_id, k, str(v)) for k, v in params.items()],
        )
        self._conn.commit()

    def log_metric(self, run_id: int, key: str, value: float, step: int | None = None) -> None:
        self._conn.execute(
            "INSERT INTO metrics (run_id, key, step, value, ts) VALUES (?, ?, ?, ?, ?)",
            (run_id, key, step, float(value), _now()),
        )
        self._conn.commit()

    def log_artifact(self, run_id: int, kind: str, path: str | Path) -> None:
        self._conn.execute(
            "INSERT INTO artifacts (run_id, kind, path) VALUES (?, ?, ?)",
            (run_id, kind, str(path)),
        )
        self._conn.commit()

    def finish_run(self, run_id: int, status: str = "finished") -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
            (status, _now(), run_id),
        )
        self._conn.commit()

    # -- querying ---------------------------------------------------------

    def list_runs(self, topic: str | None = None) -> list[dict]:
        sql = "SELECT * FROM runs"
        args: tuple = ()
        if topic is not None:
            sql += " WHERE topic = ?"
            args = (topic,)
        sql += " ORDER BY id"
        return [dict(r) for r in self._conn.execute(sql, args)]

    def get_params(self, run_id: int) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM params WHERE run_id = ?", (run_id,)
        )
        return {r["key"]: r["value"] for r in rows}

    def last_metric(self, run_id: int, key: str) -> float | None:
        row = self._conn.execute(
            "SELECT value FROM metrics WHERE run_id = ? AND key = ?"
            " ORDER BY step IS NULL, step DESC, rowid DESC LIMIT 1",
            (run_id, key),
        ).fetchone()
        return None if row is None else row["value"]

    def best_run(self, topic: str, metric: str, mode: str = "max") -> dict | None:
        """The finished run whose final value of ``metric`` is best."""
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        best: dict | None = None
        for run in self.list_runs(topic):
            if run["status"] != "finished":
                continue
            value = self.last_metric(run["id"], metric)
            if value is None:
                continue
            if (
                best is None
                or (mode == "max" and value > best["value"])
                or (mode == "min" and value < best["value"])
            ):
                best = {**run, "value": value}
        return best

    def metric_keys(self, run_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT key FROM metrics WHERE run_id = ? ORDER BY key", (run_id,)
        )
        return [r["key"] for r in rows]

    def metric_history(self, run_id: int, key: str) -> list[tuple[int | None, float]]:
        rows = self._conn.execute(
            "SELECT step, value FROM metrics WHERE run_id = ? AND key = ? ORDER BY rowid",
            (run_id, key),
        )
        return [(r["step"], r["value"]) for r in rows]
