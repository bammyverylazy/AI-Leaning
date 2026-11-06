"""Targeted sign-demonstration and webcam testing flow for the desktop demo."""
from __future__ import annotations

import random
import hashlib
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch

import config as C
from attempt_store import RecognitionAttemptStore
from dataset import build_features, preprocess_sequence
from extract import make_landmarker, result_to_array
from label_display import aggregate_predictions, canonical_label
from model import SeqClassifier
from model_vocabulary import checkpoint_vocabulary
from sign_demo_catalog import SignDemo, load_catalog, write_missing_report


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7,
               color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _play_demo(label: str, demo: SignDemo) -> Optional[tuple[bool, float]]:
    capture = cv2.VideoCapture(str(demo.path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open demonstration video: {demo.path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps)) if np.isfinite(fps) and fps > 0 else 40
    window = f"Sign demonstration - {label}"
    watched_seconds = 0.0
    playback_started: Optional[float] = None
    displayed_frames = 0
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = capture.read()
            if not ok:
                if not displayed_frames:
                    raise RuntimeError(f"Demonstration video contains no readable frames: {demo.path}")
                if playback_started is not None:
                    watched_seconds += time.perf_counter() - playback_started
                    playback_started = None
                key = cv2.waitKey(30) & 0xFF
                if key == ord("r"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    playback_started = None
                elif key in (10, 13):
                    return True, watched_seconds
                elif key in (ord("q"), 27):
                    return None
                continue

            _draw_text(
                frame,
                f"TARGET: {canonical_label(label)}",
                (12, 30),
                0.85,
                (0, 220, 220),
            )
            _draw_text(frame, "R=replay  ENTER=ready  Q=quit", (12, 62), 0.6)
            cv2.imshow(window, frame)
            displayed_frames += 1
            if playback_started is None:
                playback_started = time.perf_counter()
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (10, 13):
                if playback_started is not None:
                    watched_seconds += time.perf_counter() - playback_started
                return True, watched_seconds
            if key in (ord("q"), 27):
                return None
            if key == ord("r"):
                if playback_started is not None:
                    watched_seconds += time.perf_counter() - playback_started
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                playback_started = None
    finally:
        capture.release()
        cv2.destroyWindow(window)


def _run_camera_attempt(
    *,
    args,
    model: SeqClassifier,
    vocab: list[str],
    label: str,
    demo: SignDemo,
    demo_watched: bool,
    watched_seconds: float,
    store: RecognitionAttemptStore,
    session_id: str,
    model_version: str,
) -> str:
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {args.camera}")

    window = f"Webcam attempt - {label}"
    recording = False
    frames: list[np.ndarray] = []
    result: Optional[tuple[str, float, int]] = None
    status = "SPACE=start attempt | Q=quit"
    frame_count = 0
    t_begin = time.perf_counter()
    last_ts = -1
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        with make_landmarker() as landmarker:
            while True:
                ok, raw_frame = camera.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames")
                display = cv2.flip(raw_frame, 1)

                if recording:
                    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb),
                    )
                    now = time.perf_counter()
                    timestamp_ms = max(last_ts + 1, int((now - t_begin) * 1000))
                    last_ts = timestamp_ms
                    result_mp = landmarker.detect_for_video(image, timestamp_ms)
                    frames.append(result_to_array(result_mp))
                    frame_count += 1

                _draw_text(
                    display,
                    f"TARGET: {canonical_label(label)}",
                    (12, 30),
                    0.85,
                    (0, 220, 220),
                )
                if result is None:
                    if recording:
                        _draw_text(display, f"RECORDING - {frame_count} frames", (12, 64), 0.65,
                                   (0, 0, 255))
                        _draw_text(display, "SPACE=finish  Q=quit", (12, 94), 0.55)
                    else:
                        _draw_text(display, status, (12, 64), 0.65)
                else:
                    predicted, confidence, attempt_id = result
                    decision = f"NOT SURE (best: {predicted})" if confidence < args.thr else predicted
                    _draw_text(display, f"PREDICTION: {decision}", (12, 68), 0.75, (50, 220, 50))
                    _draw_text(display, f"CONFIDENCE: {confidence:.1%}", (12, 100), 0.65)
                    _draw_text(display, f"Attempt #{attempt_id} saved; ground truth: UNVERIFIED",
                               (12, 132), 0.55, (0, 220, 220))
                    _draw_text(display, "R=repeat target  N=next target  Q=quit", (12, 164), 0.55)

                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"
                if key == ord("n") and result is not None:
                    return "next"
                if key == ord("r") and result is not None:
                    result = None
                    recording = False
                    frames = []
                    frame_count = 0
                    status = "SPACE=start attempt | Q=quit"
                    continue
                if key != ord(" "):
                    continue

                if not recording and result is None:
                    recording = True
                    frames = []
                    frame_count = 0
                    status = "RECORDING"
                    continue
                if recording:
                    recording = False
                    if len(frames) < 8:
                        status = "Too short; SPACE=try again | Q=quit"
                        frames = []
                        frame_count = 0
                        continue

                    inference_started = time.perf_counter()
                    positions, present = preprocess_sequence(np.stack(frames))
                    features = torch.from_numpy(build_features(positions, present))[None]
                    with torch.no_grad():
                        probabilities = torch.softmax(model(features), dim=1)[0]
                    predictions = aggregate_predictions(probabilities, vocab, top_k=3)
                    best = predictions[0]
                    predicted_label = best.canonical_label
                    confidence_value = best.confidence
                    attempt_id = store.create_attempt(
                        target_label=label,
                        demo_video_id=demo.video_id,
                        demo_video_version=demo.version,
                        demo_watched=demo_watched,
                        demo_watch_duration=watched_seconds,
                        predicted_label=predicted_label,
                        confidence=confidence_value,
                        session_id=session_id,
                        original_model_label=best.original_model_label,
                        canonical_label=best.canonical_label,
                        class_index=best.class_index,
                        top_k_predictions=[
                            {
                                "canonical_label": prediction.canonical_label,
                                "confidence": prediction.confidence,
                                "original_model_label": prediction.original_model_label,
                                "class_index": prediction.class_index,
                                "original_confidence": prediction.original_confidence,
                            }
                            for prediction in predictions
                        ],
                        model_version=model_version,
                        preprocessing_version=C.PREPROCESSING_VERSION,
                        inference_latency_ms=(
                            time.perf_counter() - inference_started
                        )
                        * 1000,
                    )
                    result = (predicted_label, confidence_value, attempt_id)
    finally:
        camera.release()
        cv2.destroyWindow(window)


def run(args) -> None:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    vocab = checkpoint_vocabulary(checkpoint)
    model = SeqClassifier(
        checkpoint["feature_dim"],
        len(vocab),
        checkpoint_args["hidden"],
        2,
        checkpoint_args["dropout"],
        checkpoint_args["kind"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model_digest = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()

    catalog, missing = load_catalog(vocab, Path(args.demo_manifest))
    write_missing_report(missing)
    if args.target:
        target = args.target.strip().upper()
        if target not in vocab:
            raise SystemExit(f"{target!r} is not in the checkpoint vocabulary")
        if target not in catalog:
            raise SystemExit(
                f"No verified demonstration video available for this sign "
                f"({target}): {missing.get(target, 'unavailable')}. "
                f"See {C.MISSING_SIGN_DEMOS_REPORT}"
            )
        targets = [target]
    else:
        if not catalog:
            raise SystemExit(
                f"No verified demonstration videos are available. "
                f"Missing list: {C.MISSING_SIGN_DEMOS_REPORT}"
            )
        targets = list(catalog)

    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    rng = random.Random(args.seed)
    store = RecognitionAttemptStore(Path(args.attempt_db))
    session_id = uuid.uuid4().hex
    print(f"Verified demonstration videos: {len(catalog)}/{len(vocab)}")
    print(f"Unavailable demonstrations listed in: {C.MISSING_SIGN_DEMOS_REPORT}")
    completed = 0
    while completed < args.attempts:
        label = args.target.strip().upper() if args.target else rng.choice(targets)
        demo = catalog[label]
        playback = _play_demo(label, demo)
        if playback is None:
            break
        demo_watched, watched_seconds = playback
        outcome = _run_camera_attempt(
            args=args,
            model=model,
            vocab=vocab,
            label=label,
            demo=demo,
            demo_watched=demo_watched,
            watched_seconds=watched_seconds if demo_watched else 0.0,
            store=store,
            session_id=session_id,
            model_version=model_digest,
        )
        if outcome == "quit":
            break
        completed += 1
    print(f"Completed {completed} test session(s). Attempts stored in: {store.path}")
