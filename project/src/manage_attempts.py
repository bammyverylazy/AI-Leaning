"""Review recognition attempts without conflating target and ground truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from attempt_store import RecognitionAttemptStore
from dataset import build_features, preprocess_sequence
from evaluation_sequences import load_attempt_sequence, resolve_sequence_path
from label_display import aggregate_predictions
from model import SeqClassifier
from model_vocabulary import checkpoint_vocabulary
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
    report.add_argument(
        "--all-sessions", action="store_true", help="combine attempts from all sessions"
    )
    inspect = subparsers.add_parser("inspect", help="show one complete attempt trace")
    inspect.add_argument("attempt_id", type=int)
    replay = subparsers.add_parser("replay", help="reproduce a prediction from its saved input")
    replay.add_argument("attempt_id", type=int)
    replay.add_argument(
        "--checkpoint",
        help="optional checkpoint override; its SHA-256 must match the recorded model",
    )
    export = subparsers.add_parser(
        "export", help="export independently verified attempts as CSV"
    )
    export.add_argument(
        "--output",
        default=str(C.REPORTS / "verified_testing_attempts.csv"),
        help="CSV output path",
    )
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
            attempt_count = len(store.attempts(row["session_id"]))
            print(
                f"{row['session_id']} {row['status']} started={row['started_at']} "
                f"ended={row['ended_at'] or '—'} model={row['model_version']} "
                f"signer={row['signer_id'] or 'not recorded'} "
                f"attempts={attempt_count}"
            )
        if not rows:
            print("No Testing Mode sessions found.")
        return

    if args.command == "report":
        sessions = store.sessions()
        if args.all_sessions:
            print(format_report(build_report(store.attempts())))
            return
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

    if args.command == "inspect":
        attempt = store.get_attempt(args.attempt_id)
        session = store.get_session(attempt["session_id"])
        print(f"Attempt ID: {attempt['attempt_id']}")
        print(f"Timestamp: {attempt['timestamp'] or attempt['created_at']}")
        print(f"Session: {attempt['session_id']}")
        print(f"Signer: {attempt['signer_id'] or session['signer_id'] or 'not recorded'}")
        print(f"Session notes: {session['notes'] or 'none'}")
        print(f"Environment: {session['environment_json']}")
        print(f"Target: {attempt['target_label']}")
        print(f"Ground truth: {attempt['ground_truth_label'] or 'unverified'}")
        print(f"Correctness: {attempt['correctness_status']}")
        print(f"Prediction: {attempt['canonical_label'] or 'none'}")
        print(f"Original model label: {attempt['original_model_label'] or 'none'}")
        print(f"Confidence: {attempt['confidence']:.4f}")
        print(f"Status: {attempt['status']}")
        print("Top-K predictions:")
        for index, prediction in enumerate(
            json.loads(attempt["top_k_predictions"] or "[]"), start=1
        ):
            print(
                f"  {index}. {prediction['canonical_label']} "
                f"({prediction['confidence']:.1%}; "
                f"class={prediction['class_index']}; "
                f"model_label={prediction['original_model_label']})"
            )
        print(f"Model checkpoint: {attempt['model_checkpoint'] or 'not recorded'}")
        print(f"Model ID: {attempt['model_id']}")
        print(f"Model version: {attempt['model_version']}")
        print(f"Vocabulary version: {attempt['vocabulary_version']}")
        print(f"Preprocessing version: {attempt['preprocessing_version']}")
        print(f"Application version: {attempt['application_version']}")
        print(f"Demo: {attempt['demo_video_id'] or 'none'}")
        print(f"Demo version: {attempt['demo_video_version'] or 'none'}")
        print(f"Sequence length: {attempt['sequence_length'] or 'not recorded'}")
        print(f"Feature dimension: {attempt['feature_dimension'] or 'not recorded'}")
        print(f"Frame count: {attempt['frame_count']}")
        print(f"Recording duration: {attempt['duration_seconds'] or 0:.3f}s")
        print(f"Sequence path: {attempt['sequence_path'] or 'not recorded'}")
        return

    if args.command == "replay":
        attempt = store.get_attempt(args.attempt_id)
        try:
            result = replay_attempt(attempt, checkpoint_override=args.checkpoint)
        except (FileNotFoundError, OSError, KeyError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(f"Attempt #{attempt['attempt_id']} replay")
        print(f"Input sequence: {attempt['sequence_path']}")
        print(f"Preprocessing recomputation: {result['preprocessing_matches']}")
        print(f"Recorded prediction: {attempt['canonical_label']}")
        print(f"Replayed prediction: {result['predictions'][0].canonical_label}")
        print(f"Prediction matches: {result['prediction_matches']}")
        print("Top-K predictions:")
        for index, prediction in enumerate(result["predictions"], start=1):
            print(
                f"  {index}. {prediction.canonical_label} "
                f"({prediction.confidence:.1%}; "
                f"class={prediction.class_index}; "
                f"model_label={prediction.original_model_label})"
            )
        return

    if args.command == "export":
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = C.ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        verified_rows = [
            row for row in store.attempts()
            if row["status"] == "completed"
            and row["correctness_status"] in {"correct", "incorrect"}
            and row["ground_truth_label"]
        ]
        sessions_by_id = {
            session["session_id"]: session for session in store.sessions()
        }
        fields = (
            "attempt_id",
            "timestamp",
            "session_id",
            "signer_id",
            "environment_json",
            "session_notes",
            "target_label",
            "ground_truth_label",
            "predicted_label",
            "canonical_label",
            "confidence",
            "correctness_status",
            "demo_video_id",
            "demo_video_version",
            "model_id",
            "model_checkpoint",
            "vocabulary_version",
            "model_version",
            "preprocessing_version",
            "application_version",
            "sequence_length",
            "feature_dimension",
            "frame_count",
            "sequence_path",
        )
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in verified_rows:
                session = sessions_by_id[row["session_id"]]
                writer.writerow(
                    {
                        **{field: row[field] for field in fields
                           if field not in {"environment_json", "session_notes"}},
                        "environment_json": session["environment_json"],
                        "session_notes": session["notes"],
                    }
                )
        print(
            f"Exported {len(verified_rows)} independently verified attempts to "
            f"{output_path}. Export does not add data to training."
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


def replay_attempt(attempt, *, checkpoint_override: str | None = None) -> dict:
    """Re-run a stored model input only with its recorded checkpoint version."""
    sequence_path_value = attempt["sequence_path"]
    if not sequence_path_value:
        raise ValueError(
            f"Attempt #{attempt['attempt_id']} has no saved sequence; "
            "historical replay is unavailable."
        )
    sequence_path = resolve_sequence_path(sequence_path_value)
    sequence = load_attempt_sequence(sequence_path)
    if attempt["status"] != "completed" or not attempt["original_model_label"]:
        raise ValueError(
            f"Attempt #{attempt['attempt_id']} has no completed model prediction."
        )

    checkpoint_value = checkpoint_override or attempt["model_checkpoint"]
    if not checkpoint_value:
        raise ValueError(
            f"Attempt #{attempt['attempt_id']} has no recorded checkpoint path."
        )
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = C.ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Recorded checkpoint is unavailable: {checkpoint_path}"
        )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != attempt["model_version"]:
        raise ValueError(
            "Checkpoint SHA-256 does not match the model version recorded for "
            f"attempt #{attempt['attempt_id']}."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = checkpoint_vocabulary(checkpoint)
    if int(checkpoint["feature_dim"]) != attempt["feature_dimension"]:
        raise ValueError("Checkpoint feature dimension differs from the saved attempt.")

    positions, present = preprocess_sequence(sequence["raw_landmarks"])
    recomputed_features = build_features(positions, present)
    preprocessing_matches = np.array_equal(
        recomputed_features,
        sequence["model_features"],
        equal_nan=True,
    )
    if sequence["model_features"].shape != (
        attempt["sequence_length"],
        attempt["feature_dimension"],
    ):
        raise ValueError("Saved model input dimensions do not match attempt metadata.")
    stored_top = json.loads(attempt["top_k_predictions"] or "[]")
    if stored_top and len(vocab) <= max(
        int(item["class_index"]) for item in stored_top
    ):
        raise ValueError("Saved class index is outside the recorded model vocabulary.")

    model_args = checkpoint["args"]
    model = SeqClassifier(
        checkpoint["feature_dim"],
        len(vocab),
        model_args["hidden"],
        2,
        model_args["dropout"],
        model_args["kind"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model_features = torch.from_numpy(
        np.asarray(sequence["model_features"], dtype=np.float32)
    )[None]
    with torch.no_grad():
        probabilities = torch.softmax(model(model_features), dim=1)[0]
    predictions = aggregate_predictions(
        probabilities, vocab, top_k=3
    )
    prediction_matches = bool(predictions) and (
        predictions[0].canonical_label == attempt["canonical_label"]
        and abs(predictions[0].confidence - float(attempt["confidence"])) <= 1e-6
        and bool(stored_top)
        and predictions[0].original_model_label
        == stored_top[0]["original_model_label"]
    )
    return {
        "predictions": predictions,
        "preprocessing_matches": preprocessing_matches,
        "prediction_matches": prediction_matches,
    }


if __name__ == "__main__":
    main()
