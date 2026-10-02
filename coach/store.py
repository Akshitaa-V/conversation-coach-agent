"""SQLite store for sessions, turns, feedback, LLM calls and trainee focus areas.

SQLite keeps the demo self-contained. On Cloud Run the filesystem is per
instance and not durable, so a real deployment points ``DB_PATH`` at a mounted
volume for experiments or swaps this class for Cloud SQL (Postgres); the schema
is plain SQL and ports directly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    trainee_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_trainee ON sessions(trainee_id);

CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    idx INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('trainee', 'customer')),
    content TEXT NOT NULL,
    model TEXT,
    ttft_ms REAL,
    degraded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, idx)
);

CREATE TABLE IF NOT EXISTS feedback (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    payload TEXT NOT NULL,
    overall REAL NOT NULL,
    model TEXT NOT NULL,
    repairs INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    session_id TEXT NOT NULL,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_session ON llm_calls(session_id);

CREATE TABLE IF NOT EXISTS trainee_focus (
    trainee_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    score INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (trainee_id, dimension)
);

CREATE TABLE IF NOT EXISTS online_evals (
    session_id TEXT PRIMARY KEY,
    judge_model TEXT NOT NULL,
    judge_overall REAL NOT NULL,
    coach_overall REAL NOT NULL,
    created_at REAL NOT NULL
);
"""


class SessionStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ----------------------------------------------------------- sessions
    def create_session(self, trainee_id: str, scenario_id: str, variant: str) -> str:
        session_id = uuid.uuid4().hex[:16]
        self._exec(
            "INSERT INTO sessions (id, trainee_id, scenario_id, variant, created_at) VALUES (?,?,?,?,?)",
            (session_id, trainee_id, scenario_id, variant, time.time()),
        )
        return session_id

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        rows = self._all("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return rows[0] if rows else None

    def set_status(self, session_id: str, status: str) -> None:
        self._exec("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        model: str | None = None,
        ttft_ms: float | None = None,
        degraded: bool = False,
    ) -> None:
        idx = self._all("SELECT COUNT(*) AS n FROM turns WHERE session_id = ?", (session_id,))[0][
            "n"
        ]
        self._exec(
            "INSERT INTO turns (session_id, idx, role, content, model, ttft_ms, degraded) VALUES (?,?,?,?,?,?,?)",
            (session_id, idx, role, content, model, ttft_ms, int(degraded)),
        )

    def turns(self, session_id: str) -> list[sqlite3.Row]:
        return self._all("SELECT * FROM turns WHERE session_id = ? ORDER BY idx", (session_id,))

    # ----------------------------------------------------------- feedback
    def save_feedback(
        self, session_id: str, payload: dict, overall: float, model: str, repairs: int
    ) -> None:
        self._exec(
            "INSERT OR REPLACE INTO feedback (session_id, payload, overall, model, repairs, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, json.dumps(payload), overall, model, repairs, time.time()),
        )

    def get_feedback(self, session_id: str) -> dict | None:
        rows = self._all("SELECT payload FROM feedback WHERE session_id = ?", (session_id,))
        return json.loads(rows[0]["payload"]) if rows else None

    # ----------------------------------------------------------- llm calls
    def add_call(self, record) -> None:  # record: coach.llm.router.CallRecord
        self._exec(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                record.conversation_id,
                record.task,
                record.model,
                record.attempt,
                record.outcome,
                record.latency_ms,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                time.time(),
            ),
        )

    def session_cost(self, session_id: str) -> dict:
        rows = self._all(
            "SELECT task, COUNT(*) AS calls, SUM(cost_usd) AS cost, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens, SUM(outcome != 'ok') AS failed_calls "
            "FROM llm_calls WHERE session_id = ? GROUP BY task",
            (session_id,),
        )
        by_task = {r["task"]: {k: r[k] for k in r.keys() if k != "task"} for r in rows}
        return {
            "total_usd": round(sum(v["cost"] or 0 for v in by_task.values()), 6),
            "by_task": by_task,
        }

    # ----------------------------------------------------------- long-term memory
    def save_focus(self, trainee_id: str, scores: dict[str, int]) -> None:
        now = time.time()
        for dimension, score in scores.items():
            self._exec(
                "INSERT INTO trainee_focus (trainee_id, dimension, score, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(trainee_id, dimension) DO UPDATE SET score = excluded.score, updated_at = excluded.updated_at",
                (trainee_id, dimension, score, now),
            )

    def weakest_dimensions(self, trainee_id: str, k: int = 2) -> list[str]:
        rows = self._all(
            "SELECT dimension FROM trainee_focus WHERE trainee_id = ? AND score <= 3 "
            "ORDER BY score ASC, dimension ASC LIMIT ?",
            (trainee_id, k),
        )
        return [r["dimension"] for r in rows]

    # ----------------------------------------------------------- online evals
    def save_online_eval(
        self, session_id: str, judge_model: str, judge_overall: float, coach_overall: float
    ) -> None:
        self._exec(
            "INSERT OR REPLACE INTO online_evals VALUES (?,?,?,?,?)",
            (session_id, judge_model, judge_overall, coach_overall, time.time()),
        )

    def experiment_rows(self) -> list[dict]:
        """One row per finished session: variant, feedback score, cost. Input for experiments.analysis."""
        rows = self._all(
            "SELECT s.id, s.variant, f.overall, COALESCE(SUM(c.cost_usd), 0) AS cost_usd "
            "FROM sessions s JOIN feedback f ON f.session_id = s.id "
            "LEFT JOIN llm_calls c ON c.session_id = s.id GROUP BY s.id"
        )
        return [dict(r) for r in rows]
