"""SQLite persistence for recognition attempts and ground-truth review history."""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from label_display import canonical_label


_ATTEMPT_COLUMNS = {
    "session_id": "TEXT NOT NULL DEFAULT 'legacy'",
    "original_model_label": "TEXT NOT NULL DEFAULT ''",
    "canonical_label": "TEXT NOT NULL DEFAULT ''",
    "class_index": "INTEGER",
    "top_k_predictions": "TEXT NOT NULL DEFAULT '[]'",
    "timestamp": "TEXT",
    "model_version": "TEXT NOT NULL DEFAULT 'unknown'",
    "preprocessing_version": "TEXT NOT NULL DEFAULT 'unknown'",
    "inference_latency_ms": "REAL",
    "status": "TEXT NOT NULL DEFAULT 'completed'",
    "started_at": "TEXT",
    "ended_at": "TEXT",
    "duration_seconds": "REAL",
    "error": "TEXT",
    "warning": "TEXT",
}

_TERMINAL_ATTEMPT_STATUSES = {
    "completed",
    "no_prediction",
    "invalid",
    "cancelled",
    "interrupted",
    "error",
}


def _validate_top_k(top_k_predictions: list[dict] | None) -> str:
    top_k = top_k_predictions or []
    for prediction in top_k:
        probability = prediction.get("confidence")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("each top-K prediction must have a valid confidence")
        if not all(
            isinstance(prediction.get(key), str) and prediction[key].strip()
            for key in ("canonical_label", "original_model_label")
        ):
            raise ValueError("each top-K prediction must include both labels")
        class_index_value = prediction.get("class_index")
        if (
            not isinstance(class_index_value, int)
            or isinstance(class_index_value, bool)
            or class_index_value < 0
        ):
            raise ValueError("each top-K prediction must include a non-negative class index")
    return json.dumps(top_k, ensure_ascii=False, allow_nan=False)


class RecognitionAttemptStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS testing_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    ended_at TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('active', 'completed', 'interrupted', 'error')
                    ),
                    model_version TEXT NOT NULL,
                    vocabulary_version TEXT NOT NULL,
                    configuration_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    session_id TEXT NOT NULL DEFAULT 'legacy',
                    target_label TEXT NOT NULL,
                    demo_video_id TEXT NOT NULL,
                    demo_video_version TEXT NOT NULL,
                    demo_watched INTEGER NOT NULL CHECK (demo_watched IN (0, 1)),
                    demo_watch_duration REAL NOT NULL CHECK (demo_watch_duration >= 0),
                    predicted_label TEXT NOT NULL,
                    original_model_label TEXT NOT NULL DEFAULT '',
                    canonical_label TEXT NOT NULL DEFAULT '',
                    class_index INTEGER,
                    top_k_predictions TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    ground_truth_label TEXT,
                    correctness_status TEXT NOT NULL DEFAULT 'unverified'
                        CHECK (correctness_status IN ('unverified', 'correct', 'incorrect')),
                    verified_by TEXT,
                    verified_at TEXT,
                    timestamp TEXT,
                    model_version TEXT NOT NULL DEFAULT 'unknown',
                    preprocessing_version TEXT NOT NULL DEFAULT 'unknown',
                    inference_latency_ms REAL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    started_at TEXT,
                    ended_at TEXT,
                    duration_seconds REAL,
                    error TEXT,
                    warning TEXT
                )
                """
            )
            existing = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(recognition_attempts)")
            }
            for name, definition in _ATTEMPT_COLUMNS.items():
                if name not in existing:
                    connection.execute(
                        f"ALTER TABLE recognition_attempts ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                UPDATE recognition_attempts
                SET original_model_label = predicted_label
                WHERE original_model_label = ''
                """
            )
            connection.execute(
                """
                UPDATE recognition_attempts
                SET canonical_label = predicted_label
                WHERE canonical_label = ''
                """
            )
            connection.execute(
                """
                UPDATE recognition_attempts SET timestamp = created_at
                WHERE timestamp IS NULL
                """
            )
            connection.execute(
                """
                UPDATE recognition_attempts SET started_at = created_at
                WHERE started_at IS NULL
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_attempt_reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id INTEGER NOT NULL REFERENCES recognition_attempts(attempt_id),
                    action TEXT NOT NULL CHECK (action IN ('verify', 'correct')),
                    previous_ground_truth_label TEXT,
                    new_ground_truth_label TEXT NOT NULL,
                    previous_correctness_status TEXT NOT NULL,
                    new_correctness_status TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reason TEXT,
                    reviewed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS testing_attempt_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    attempt_id INTEGER,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    details_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_events_attempt "
                "ON testing_attempt_events(attempt_id, event_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_session "
                "ON recognition_attempts(session_id, attempt_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_unverified "
                "ON recognition_attempts(correctness_status, attempt_id)"
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def start_session(
        self,
        session_id: str,
        *,
        model_version: str,
        vocabulary_version: str,
        configuration: dict,
    ) -> str:
        if not session_id.strip() or not model_version.strip() or not vocabulary_version.strip():
            raise ValueError("session ID and version identifiers must not be empty")
        encoded_configuration = json.dumps(
            configuration, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO testing_sessions (
                    session_id, status, model_version, vocabulary_version,
                    configuration_json
                ) VALUES (?, 'active', ?, ?, ?)
                """,
                (session_id, model_version, vocabulary_version, encoded_configuration),
            )
            self._insert_event(
                connection, session_id, None, "session_started",
                {"model_version": model_version, "vocabulary_version": vocabulary_version},
            )
        return session_id

    def end_session(self, session_id: str, status: str = "completed") -> str:
        if status not in {"completed", "interrupted", "error"}:
            raise ValueError("session status must be completed, interrupted, or error")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE testing_sessions
                SET status = ?,
                    ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE session_id = ? AND status = 'active'
                """,
                (status, session_id),
            )
            if cursor.rowcount == 0:
                exists = connection.execute(
                    "SELECT status FROM testing_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"Testing session {session_id} does not exist")
                if exists["status"] != status:
                    raise ValueError(
                        f"Testing session {session_id} is already {exists['status']}"
                    )
                return status
            self._insert_event(
                connection, session_id, None, f"session_{status}", {}
            )
        return status

    def record_event(
        self,
        session_id: str,
        event_type: str,
        *,
        attempt_id: int | None = None,
        details: dict | None = None,
    ) -> None:
        if not session_id.strip() or not event_type.strip():
            raise ValueError("session_id and event_type must not be empty")
        encoded_details = details or {}
        json.dumps(encoded_details, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            self._insert_event(
                connection, session_id, attempt_id, event_type, encoded_details
            )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        session_id: str,
        attempt_id: int | None,
        event_type: str,
        details: dict,
    ) -> None:
        connection.execute(
            """
            INSERT INTO testing_attempt_events (
                session_id, attempt_id, event_type, details_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                attempt_id,
                event_type,
                json.dumps(details, ensure_ascii=False, allow_nan=False, sort_keys=True),
            ),
        )

    def begin_attempt(
        self,
        *,
        session_id: str,
        target_label: str,
        model_version: str,
        preprocessing_version: str,
        demo_video_id: str = "",
        demo_video_version: str = "",
        demo_watched: bool = False,
        demo_watch_duration: float = 0.0,
    ) -> int:
        if not all(
            value.strip()
            for value in (session_id, target_label, model_version, preprocessing_version)
        ):
            raise ValueError("session, target, model, and preprocessing versions are required")
        if not isinstance(demo_watched, bool):
            raise ValueError("demo_watched must be a boolean")
        if (
            isinstance(demo_watch_duration, bool)
            or not math.isfinite(demo_watch_duration)
            or demo_watch_duration < 0
        ):
            raise ValueError("demo_watch_duration must be finite and non-negative")
        with self._connect() as connection:
            session = connection.execute(
                "SELECT status FROM testing_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None or session["status"] != "active":
                raise ValueError(f"Testing session {session_id} is not active")
            cursor = connection.execute(
                """
                INSERT INTO recognition_attempts (
                    session_id, target_label, demo_video_id, demo_video_version,
                    demo_watched, demo_watch_duration, predicted_label,
                    original_model_label, canonical_label, top_k_predictions,
                    confidence, model_version, preprocessing_version, status,
                    started_at, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, '', '', '', '[]', 0, ?, ?, 'in_progress',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (
                    session_id,
                    target_label,
                    demo_video_id,
                    demo_video_version,
                    int(demo_watched),
                    demo_watch_duration,
                    model_version,
                    preprocessing_version,
                ),
            )
            attempt_id = cursor.lastrowid
            if attempt_id is None:
                raise RuntimeError("SQLite did not return an ID for the initialized attempt")
            self._insert_event(
                connection, session_id, attempt_id, "target_selected",
                {"target_label": target_label, "demo_video_id": demo_video_id or None},
            )
            if demo_watched:
                self._insert_event(
                    connection,
                    session_id,
                    attempt_id,
                    "demo_watched",
                    {
                        "demo_video_id": demo_video_id or None,
                        "duration_seconds": demo_watch_duration,
                    },
                )
            self._insert_event(
                connection, session_id, attempt_id, "attempt_started", {}
            )
            self._insert_event(
                connection, session_id, attempt_id, "recognition_started", {}
            )
        return attempt_id

    def finish_attempt(
        self,
        attempt_id: int,
        *,
        status: str,
        predicted_label: str = "",
        original_model_label: str = "",
        canonical_label_value: str = "",
        class_index: int | None = None,
        top_k_predictions: list[dict] | None = None,
        confidence: float = 0.0,
        inference_latency_ms: float | None = None,
        error: str | None = None,
        warning: str | None = None,
    ) -> str:
        if status not in _TERMINAL_ATTEMPT_STATUSES:
            raise ValueError(f"Unsupported terminal attempt status: {status}")
        if status == "completed" and not all(
            value.strip()
            for value in (predicted_label, original_model_label, canonical_label_value)
        ):
            raise ValueError("completed attempts require a model prediction")
        if (
            isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1")
        if class_index is not None and (
            isinstance(class_index, bool) or class_index < 0
        ):
            raise ValueError("class_index must be non-negative")
        if inference_latency_ms is not None and (
            not math.isfinite(inference_latency_ms) or inference_latency_ms < 0
        ):
            raise ValueError("inference_latency_ms must be finite and non-negative")
        encoded_top_k = _validate_top_k(top_k_predictions)
        with self._connect() as connection:
            attempt = connection.execute(
                """
                SELECT session_id, status
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise KeyError(f"Recognition attempt {attempt_id} does not exist")
            if attempt["status"] != "in_progress":
                return attempt["status"]
            cursor = connection.execute(
                """
                UPDATE recognition_attempts
                SET predicted_label = ?, original_model_label = ?,
                    canonical_label = ?, class_index = ?, top_k_predictions = ?,
                    confidence = ?, inference_latency_ms = ?, status = ?,
                    ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    duration_seconds = MAX(0, (julianday('now') - julianday(started_at)) * 86400),
                    error = ?, warning = ?
                WHERE attempt_id = ? AND status = 'in_progress'
                """,
                (
                    predicted_label,
                    original_model_label,
                    canonical_label_value,
                    class_index,
                    encoded_top_k,
                    confidence,
                    inference_latency_ms,
                    status,
                    error,
                    warning,
                    attempt_id,
                ),
            )
            if cursor.rowcount == 0:
                status_row = connection.execute(
                    "SELECT status FROM recognition_attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                if status_row is None:
                    raise KeyError(f"Recognition attempt {attempt_id} does not exist")
                return status_row["status"]
            details = {
                "status": status,
                "prediction": canonical_label_value or None,
                "error": error,
                "warning": warning,
            }
            if status in {"completed", "no_prediction"}:
                self._insert_event(
                    connection, attempt["session_id"], attempt_id,
                    "attempt_completed", details,
                )
                self._insert_event(
                    connection, attempt["session_id"], attempt_id,
                    "attempt_saved", details,
                )
            else:
                self._insert_event(
                    connection, attempt["session_id"], attempt_id,
                    "attempt_ended", details,
                )
        return status

    def attempts(self, session_id: str | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT * FROM recognition_attempts
        """
        parameters: tuple = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            parameters = (session_id,)
        query += " ORDER BY attempt_id"
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def get_attempt(self, attempt_id: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM recognition_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Recognition attempt {attempt_id} does not exist")
        return row

    def attempt_events(self, attempt_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            attempt = connection.execute(
                "SELECT session_id FROM recognition_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise KeyError(f"Recognition attempt {attempt_id} does not exist")
            return connection.execute(
                """
                SELECT event_id, session_id, attempt_id, event_type,
                       occurred_at, details_json
                FROM testing_attempt_events
                WHERE attempt_id = ?
                   OR (session_id = ? AND attempt_id IS NULL)
                ORDER BY event_id
                """,
                (attempt_id, attempt["session_id"]),
            ).fetchall()

    def sessions(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM testing_sessions ORDER BY started_at DESC"
            ).fetchall()

    def create_attempt(
        self,
        *,
        target_label: str,
        demo_video_id: str,
        demo_video_version: str,
        demo_watched: bool,
        demo_watch_duration: float,
        predicted_label: str,
        confidence: float,
        session_id: str = "legacy",
        original_model_label: str | None = None,
        canonical_label: str | None = None,
        class_index: int | None = None,
        top_k_predictions: list[dict] | None = None,
        model_version: str = "unknown",
        preprocessing_version: str = "unknown",
        inference_latency_ms: float | None = None,
    ) -> int:
        if (
            isinstance(demo_watch_duration, bool)
            or not math.isfinite(demo_watch_duration)
            or demo_watch_duration < 0
        ):
            raise ValueError("demo_watch_duration must be a finite non-negative number")
        if (
            isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(demo_watched, bool):
            raise ValueError("demo_watched must be a boolean")
        if inference_latency_ms is not None and (
            not math.isfinite(inference_latency_ms) or inference_latency_ms < 0
        ):
            raise ValueError("inference_latency_ms must be finite and non-negative")
        if original_model_label is None:
            original_model_label = predicted_label
        if canonical_label is None:
            canonical_label = predicted_label
        if not all(
            value.strip()
            for value in (
                target_label,
                demo_video_id,
                demo_video_version,
                predicted_label,
                original_model_label,
                canonical_label,
                session_id,
                model_version,
                preprocessing_version,
            )
        ):
            raise ValueError("attempt labels, IDs, and versions must not be empty")
        top_k = top_k_predictions or []
        for prediction in top_k:
            probability = prediction.get("confidence")
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("each top-K prediction must have a valid confidence")
            if not all(
                isinstance(prediction.get(key), str) and prediction[key].strip()
                for key in ("canonical_label", "original_model_label")
            ):
                raise ValueError("each top-K prediction must include both labels")
            class_index_value = prediction.get("class_index")
            if (
                not isinstance(class_index_value, int)
                or isinstance(class_index_value, bool)
                or class_index_value < 0
            ):
                raise ValueError("each top-K prediction must include a non-negative class index")
        if class_index is not None and (
            isinstance(class_index, bool) or class_index < 0
        ):
            raise ValueError("class_index must be non-negative")
        encoded_top_k = json.dumps(top_k, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recognition_attempts (
                    session_id, target_label, demo_video_id, demo_video_version,
                    demo_watched, demo_watch_duration, predicted_label,
                    original_model_label, canonical_label, class_index,
                    top_k_predictions, confidence, timestamp, model_version,
                    preprocessing_version, inference_latency_ms, status,
                    started_at, ended_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?, 'completed',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
                """,
                (
                    session_id,
                    target_label,
                    demo_video_id,
                    demo_video_version,
                    int(demo_watched),
                    demo_watch_duration,
                    predicted_label,
                    original_model_label,
                    canonical_label,
                    class_index,
                    encoded_top_k,
                    confidence,
                    model_version,
                    preprocessing_version,
                    inference_latency_ms,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an ID for the saved attempt")
            return cursor.lastrowid

    def pending_attempts(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT attempt_id, created_at, target_label, predicted_label,
                       original_model_label, canonical_label, confidence,
                       session_id, model_version, class_index
                FROM recognition_attempts
                WHERE correctness_status = 'unverified'
                  AND status = 'completed' AND predicted_label != ''
                ORDER BY attempt_id
                """
            ).fetchall()

    def review_history(self, attempt_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM recognition_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"Recognition attempt {attempt_id} does not exist")
            return connection.execute(
                """
                SELECT action, previous_ground_truth_label, new_ground_truth_label,
                       previous_correctness_status, new_correctness_status,
                       reviewer, reason, reviewed_at
                FROM recognition_attempt_reviews
                WHERE attempt_id = ? ORDER BY review_id
                """,
                (attempt_id,),
            ).fetchall()

    def get_session(self, session_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM testing_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Testing session {session_id} does not exist")
        return row

    def verify_ground_truth(
        self, attempt_id: int, ground_truth_label: str, verified_by: str
    ) -> str:
        return self._review(
            attempt_id, ground_truth_label, verified_by, "verify", reason=None
        )

    def correct_ground_truth(
        self, attempt_id: int, ground_truth_label: str, verified_by: str, reason: str
    ) -> str:
        reason = reason.strip()
        if not reason:
            raise ValueError("A correction reason is required")
        return self._review(
            attempt_id, ground_truth_label, verified_by, "correct", reason
        )

    def _review(
        self,
        attempt_id: int,
        ground_truth_label: str,
        reviewer: str,
        action: str,
        reason: str | None,
    ) -> str:
        ground_truth_label = ground_truth_label.strip()
        reviewer = reviewer.strip()
        if not ground_truth_label:
            raise ValueError("ground_truth_label cannot be empty")
        if not reviewer:
            raise ValueError("reviewer cannot be empty")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, predicted_label, ground_truth_label,
                       correctness_status, canonical_label, status
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Recognition attempt {attempt_id} does not exist")
            if row["status"] != "completed" or not row["predicted_label"]:
                raise ValueError("Only completed attempts with a prediction can be verified")
            if action == "verify" and row["correctness_status"] != "unverified":
                raise ValueError(f"Recognition attempt {attempt_id} has already been reviewed")
            if action == "correct" and row["correctness_status"] == "unverified":
                raise ValueError("An unverified attempt must be verified before correction")

            status = (
                "correct"
                if canonical_label(row["canonical_label"] or row["predicted_label"])
                == canonical_label(ground_truth_label)
                else "incorrect"
            )
            connection.execute(
                """
                UPDATE recognition_attempts
                SET ground_truth_label = ?, correctness_status = ?, verified_by = ?,
                    verified_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE attempt_id = ?
                """,
                (ground_truth_label, status, reviewer, attempt_id),
            )
            connection.execute(
                """
                INSERT INTO recognition_attempt_reviews (
                    attempt_id, action, previous_ground_truth_label,
                    new_ground_truth_label, previous_correctness_status,
                    new_correctness_status, reviewer, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    action,
                    row["ground_truth_label"],
                    ground_truth_label,
                    row["correctness_status"],
                    status,
                    reviewer,
                    reason,
                ),
            )
            self._insert_event(
                connection,
                row["session_id"],
                attempt_id,
                "ground_truth_verified" if action == "verify" else "ground_truth_corrected",
                {
                    "ground_truth_label": ground_truth_label,
                    "reviewer": reviewer,
                    "correctness_status": status,
                    "reason": reason,
                },
            )
        return status
