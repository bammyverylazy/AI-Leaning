"""Review recognition attempts without conflating target and ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as C
from attempt_store import RecognitionAttemptStore
from testing_metrics import build_report, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=str, default=str(C.RECOGNITION_ATTEMPTS_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pending", help="list attempts awaiting ground-truth review")
    verify = subparsers.add_parser("verify", help="record independently verified ground truth")
    verify.add_argument("--attempt-id", type=int, required=True)
    verify.add_argument("--ground-truth", required=True)
    verify.add_argument("--verified-by", required=True)
    correct = subparsers.add_parser("correct", help="correct verified ground truth with audit")
    correct.add_argument("--attempt-id", type=int, required=True)
    correct.add_argument("--ground-truth", required=True)
    correct.add_argument("--verified-by", required=True)
    correct.add_argument("--reason", required=True)
    history = subparsers.add_parser("history", help="show ground-truth review history")
    history.add_argument("--attempt-id", type=int, required=True)
    trace = subparsers.add_parser("trace", help="show an attempt's event trace")
    trace.add_argument("--attempt-id", type=int, required=True)
    attempts = subparsers.add_parser("attempts", help="list persisted test attempts")
    attempts.add_argument("--session-id")
    subparsers.add_parser("sessions", help="list Testing Mode sessions")
    report = subparsers.add_parser("report", help="report metrics from Testing Mode attempts")
    report.add_argument("--session-id", help="defaults to the most recent Testing Mode session")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = C.ROOT / db_path
    store = RecognitionAttemptStore(db_path)
    if args.command == "pending":
        rows = store.pending_attempts()
        if not rows:
            print("No attempts are awaiting review.")
            return
        for row in rows:
            print(
                f"#{row['attempt_id']} target={row['target_label']} "
                f"prediction={row['predicted_label']} confidence={row['confidence']:.3f}"
            )
        return

    if args.command == "history":
        rows = store.review_history(args.attempt_id)
        if not rows:
            print(f"No review history for attempt #{args.attempt_id}.")
            return
        for row in rows:
            print(
                f"{row['reviewed_at']} {row['action']}: "
                f"{row['previous_ground_truth_label']} -> {row['new_ground_truth_label']} "
                f"({row['previous_correctness_status']} -> "
                f"{row['new_correctness_status']}) by {row['reviewer']}; "
                f"reason={row['reason'] or '-'}"
            )
        return

    if args.command == "trace":
        rows = store.attempt_events(args.attempt_id)
        if not rows:
            print(f"No event trace for attempt #{args.attempt_id}.")
            return
        for row in rows:
            print(
                f"{row['occurred_at']} {row['event_type']} "
                f"(attempt={row['attempt_id'] or 'session'}): "
                f"{json.dumps(json.loads(row['details_json']), ensure_ascii=False)}"
            )
        return

    if args.command == "attempts":
        rows = store.attempts(args.session_id)
        for row in rows:
            print(
                f"#{row['attempt_id']} {row['status']} target={row['target_label']} "
                f"prediction={row['canonical_label'] or '—'} "
                f"ground_truth={row['ground_truth_label'] or 'unverified'} "
                f"session={row['session_id']}"
            )
        if not rows:
            print("No attempts found.")
        return

    if args.command == "sessions":
        rows = store.sessions()
        for row in rows:
            print(
                f"{row['session_id']} {row['status']} started={row['started_at']} "
                f"ended={row['ended_at'] or '—'} model={row['model_version']}"
            )
        if not rows:
            print("No Testing Mode sessions found.")
        return

    if args.command == "report":
        sessions = store.sessions()
        session_id = args.session_id
        if session_id is None:
            if not sessions:
                print("No Testing Mode sessions found; no Testing Mode report is available.")
                return
            session_id = sessions[0]["session_id"]
        session = store.get_session(session_id)
        print(
            format_report(
                build_report(store.attempts(session_id), session=session)
            )
        )
        return

    if args.command == "correct":
        status = store.correct_ground_truth(
            args.attempt_id, args.ground_truth, args.verified_by, args.reason
        )
    else:
        status = store.verify_ground_truth(
            args.attempt_id, args.ground_truth, args.verified_by
        )
    print(f"Attempt #{args.attempt_id} reviewed: {status}")


if __name__ == "__main__":
    main()
