"""Unified desktop interface for ASL practice and verified sign testing."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
import statistics
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import torch
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog, ttk

import config as C
from attempt_store import RecognitionAttemptStore
from dataset import build_features, preprocess_sequence
from extract import make_landmarker, result_to_array
from label_display import CanonicalPrediction, aggregate_predictions, canonical_label
from model import SeqClassifier
from sign_demo_catalog import SignDemo, load_catalog, write_missing_report
from speech import SpeechService, spoken_text
from subtitle import State, SubtitleManager
from testing_metrics import build_report, format_report


BACKGROUND = "#101010"
CARD = "#171717"
INK = "#ffffff"
MUTED = "#b4b4b4"
ACCENT = "#32dc32"
ACCENT_DARK = "#20a820"
SUCCESS = "#32dc32"
WARNING = "#ffa514"
ERROR = "#3232dc"
PREVIEW_BG = "#000000"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 220, 50)
ORANGE = (20, 165, 255)
RED = (50, 50, 220)
GRAY = (180, 180, 180)
YELLOW = (0, 220, 220)
CYAN = (220, 200, 0)

STATE_COLOR = {
    State.IDLE: GRAY,
    State.CANDIDATE: ORANGE,
    State.CONFIRMED: GREEN,
    State.COOLDOWN: CYAN,
}
STATE_LABEL = {
    State.IDLE: "Waiting...",
    State.CANDIDATE: "Detecting",
    State.CONFIRMED: "Confirmed!",
    State.COOLDOWN: "Pause",
}


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
        thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def draw_filled_rect(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.55,
) -> None:
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return
    region = image[y1:y2, x1:x2]
    overlay = np.full_like(region, color)
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)


def draw_progress_bar(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: tuple[int, int, int] = GREEN,
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), GRAY, 1)
    filled = int(width * min(max(progress, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + height), color, -1)


def wrap_text(text: str, scale: float, thickness: int, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        if text_width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_landmark_overlay(frame: np.ndarray, landmarks: np.ndarray) -> None:
    height, width = frame.shape[:2]

    def draw_points(points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
        for point in points:
            if not np.isnan(point[0]):
                cv2.circle(
                    frame,
                    (int(point[0] * width), int(point[1] * height)),
                    radius,
                    color,
                    -1,
                )

    draw_points(landmarks[C.LH], (0, 220, 255), 3)
    draw_points(landmarks[C.RH], (0, 140, 255), 3)
    draw_points(landmarks[C.POSE_SL], (200, 200, 50), 2)

    hand_connections = (
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    for hand_slice, color in ((C.LH, (0, 220, 255)), (C.RH, (0, 140, 255))):
        points = landmarks[hand_slice]
        for start, end in hand_connections:
            first, second = points[start], points[end]
            if not (np.isnan(first[0]) or np.isnan(second[0])):
                cv2.line(
                    frame,
                    (int(first[0] * width), int(first[1] * height)),
                    (int(second[0] * width), int(second[1] * height)),
                    color,
                    1,
                    cv2.LINE_AA,
                )


class PerformanceMonitor:
    """Keep bounded timing samples and report only when debug mode is enabled."""

    def __init__(self) -> None:
        self._samples: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            samples = self._samples.setdefault(stage, collections.deque(maxlen=240))
            samples.append(milliseconds)

    def mean(self, stage: str) -> float:
        with self._lock:
            samples = self._samples.get(stage)
            return statistics.fmean(samples) if samples else 0.0

    def report(self) -> str:
        with self._lock:
            snapshot = {name: list(samples) for name, samples in self._samples.items()}
        parts = []
        for stage, values in snapshot.items():
            if not values:
                continue
            parts.append(
                f"{stage} avg={statistics.fmean(values):.1f}ms "
                f"p95={float(np.percentile(values, 95)):.1f}ms n={len(values)}"
            )
        return " | ".join(parts)


class CameraCapture(threading.Thread):
    """Read webcam frames independently and retain only the freshest unprocessed frame."""

    def __init__(
        self,
        camera_index: int,
        frames: queue.Queue[tuple[np.ndarray, float]],
        stop_event: threading.Event,
        events: queue.Queue[tuple[str, Any]],
        performance: PerformanceMonitor,
        mode: str,
    ) -> None:
        super().__init__(name="asl-camera-capture", daemon=True)
        self.camera_index = camera_index
        self.frames = frames
        self.stop_event = stop_event
        self.events = events
        self.performance = performance
        self.mode = mode
        self.dropped_frames = 0

    def run(self) -> None:
        camera: cv2.VideoCapture | None = None
        try:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera_index}; try another camera index."
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
            self.events.put(("camera_ready", self.mode))

            while not self.stop_event.is_set():
                started = time.perf_counter()
                ok, frame = camera.read()
                ended = time.perf_counter()
                self.performance.add("camera_read", (ended - started) * 1000)
                if not ok:
                    if not self.stop_event.is_set():
                        raise RuntimeError("The webcam stopped returning frames.")
                    break
                packet = (frame, ended)
                try:
                    self.frames.put_nowait(packet)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        self.dropped_frames += 1
                    self.frames.put_nowait(packet)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.events.put(("worker_error", str(exc)))
                self.stop_event.set()
        finally:
            if camera is not None:
                camera.release()


class VideoPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, *, height: int = 400) -> None:
        super().__init__(master, style="Preview.TFrame")
        self._last_frame: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self.image = tk.Label(
            self,
            text="Camera is off",
            bg=PREVIEW_BG,
            fg="#d7e0ef",
            font=("Segoe UI", 15),
            justify="center",
        )
        self.image.pack(fill="both", expand=True)
        self.configure(height=height)
        self.pack_propagate(False)

    def show_placeholder(self, text: str) -> None:
        self._last_frame = None
        self._photo = None
        self.image.configure(image="", text=text)

    def show_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        width = max(2, self.image.winfo_width())
        height = max(2, self.image.winfo_height())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.image.configure(image=self._photo, text="")


class RecognitionWorker(threading.Thread):
    """Owns the sole webcam and MediaPipe loop while a mode is active."""

    def __init__(
        self,
        *,
        mode: str,
        camera_index: int,
        model: SeqClassifier,
        vocab: list[str],
        args: argparse.Namespace,
        subtitle_manager: SubtitleManager,
        events: queue.Queue[tuple[str, Any]],
        demo: SignDemo | None = None,
        target_label: str | None = None,
        demo_watched: bool = False,
        demo_watch_duration: float = 0.0,
        model_version: str,
        session_id: str,
        store: RecognitionAttemptStore | None = None,
        attempt_id: int | None = None,
    ) -> None:
        super().__init__(name="asl-recognition", daemon=True)
        self.mode = mode
        self.camera_index = camera_index
        self.model = model
        self.vocab = vocab
        self.args = args
        self.subtitle_manager = subtitle_manager
        self.events = events
        self.demo = demo
        self.target_label = target_label
        self.demo_watched = demo_watched
        self.demo_watch_duration = demo_watch_duration
        self.model_version = model_version
        self.session_id = session_id
        self.store = store
        self.current_attempt_id = attempt_id
        self._stop_event = threading.Event()
        self._commands: queue.Queue[tuple[str, int | None]] = queue.Queue()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_version = 0
        self._commit_lock = threading.Lock()
        self._commit_sequence = 0
        self._recording_lock = threading.Lock()
        self._recording = False
        self.show_landmarks = True
        self._display_predictions: list[CanonicalPrediction] = []
        self._hands_present = False
        self._latest_commit: tuple[int, str, float, float] | None = None
        self.performance = PerformanceMonitor()
        self._fps_samples: collections.deque[float] = collections.deque(maxlen=30)
        self._commit_sequence = 0
        self._last_rendered_commit = 0
        self._candidate_label = ""
        self._candidate_started_at: float | None = None
        self._stable_predictions: list[CanonicalPrediction] = []
        self._attempt_inference_latencies: list[float] = []
        self._prediction_interval = 2
        self._frames: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(
            maxsize=1
        )
        self.capture = CameraCapture(
            camera_index,
            self._frames,
            self._stop_event,
            self.events,
            self.performance,
            mode,
        )

    @property
    def recording(self) -> bool:
        with self._recording_lock:
            return self._recording

    @property
    def commit_sequence(self) -> int:
        with self._commit_lock:
            return self._commit_sequence

    def command(self, value: str, attempt_id: int | None = None) -> None:
        self._commands.put((value, attempt_id))

    def stop(self) -> None:
        self._stop_event.set()
        self.capture.stop_event.set()

    @property
    def commit_info(self) -> tuple[int, str, float, float] | None:
        return self._latest_commit

    def latest_frame(self, after_version: int) -> tuple[int, np.ndarray | None]:
        with self._frame_lock:
            if self._frame_version <= after_version:
                return after_version, None
            return self._frame_version, self._latest_frame

    def run(self) -> None:
        try:
            recording_frames: list[np.ndarray] = []
            frame_buffer: collections.deque[np.ndarray] = collections.deque(
                maxlen=C.SEQ_LEN
            )
            frame_count = 0
            started_at = time.perf_counter()
            last_timestamp = -1
            previous_processed_at: float | None = None
            self.capture.start()
            with make_landmarker() as landmarker:
                while not self._stop_event.is_set():
                    self._process_commands(recording_frames, frame_buffer)
                    if self._stop_event.is_set():
                        break
                    try:
                        raw_frame, captured_at = self._frames.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    frame_started = time.perf_counter()
                    frame_count += 1
                    source_height, source_width = raw_frame.shape[:2]
                    scale = min(1280 / source_width, 720 / source_height)
                    display_width = max(1, round(source_width * scale))
                    display_height = max(1, round(source_height * scale))
                    offset_x = (1280 - display_width) // 2
                    offset_y = (720 - display_height) // 2
                    display_source = cv2.resize(
                        raw_frame,
                        (display_width, display_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    display_frame = np.zeros((720, 1280, 3), dtype=raw_frame.dtype)
                    display_frame[
                        offset_y : offset_y + display_height,
                        offset_x : offset_x + display_width,
                    ] = cv2.flip(display_source, 1)
                    if previous_processed_at is not None:
                        self._fps_samples.append(
                            1.0 / max(frame_started - previous_processed_at, 1e-6)
                        )
                    previous_processed_at = frame_started
                    self.performance.add(
                        "captured_frame_age",
                        max(0.0, (frame_started - captured_at) * 1000),
                    )

                    if self.mode == "demo" or (
                        self.mode == "testing" and self.recording
                    ):
                        stage_started = time.perf_counter()
                        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb),
                        )
                        self.performance.add(
                            "frame_to_mp_image",
                            (time.perf_counter() - stage_started) * 1000,
                        )
                        now = captured_at
                        timestamp_ms = max(
                            last_timestamp + 1,
                            int((now - started_at) * 1000),
                        )
                        last_timestamp = timestamp_ms
                        landmark_started = time.perf_counter()
                        landmarks = result_to_array(
                            landmarker.detect_for_video(image, timestamp_ms)
                        )
                        self.performance.add(
                            "mediapipe",
                            (time.perf_counter() - landmark_started) * 1000,
                        )
                        frame_buffer.append(landmarks)
                        if self.mode == "testing":
                            recording_frames.append(landmarks)
                            self.events.put(
                                ("recording_progress", len(recording_frames))
                            )
                        if frame_count % self._prediction_interval == 0:
                            self._recognize_live(frame_buffer, landmarks)
                        if self.mode == "demo":
                            display_landmarks = landmarks.copy()
                            display_landmarks[:, 0] = (
                                offset_x
                                + display_width * (1.0 - display_landmarks[:, 0])
                            ) / 1280
                            display_landmarks[:, 1] = (
                                offset_y + display_height * display_landmarks[:, 1]
                            ) / 720
                            self._draw_demo_overlay(
                                display_frame,
                                display_landmarks,
                                (
                                    float(np.mean(self._fps_samples))
                                    if self._fps_samples else 0.0
                                ),
                                self.performance.mean("mediapipe"),
                                self.performance.mean("model_inference"),
                            )
                    self._publish_frame(display_frame)
                    self.performance.add(
                        "frame_processing",
                        (time.perf_counter() - frame_started) * 1000,
                    )
        except Exception as exc:
            if self.mode == "testing" and self.current_attempt_id is not None:
                try:
                    self._finish_active_attempt(
                        "error", error=f"{type(exc).__name__}: {exc}"
                    )
                except Exception as save_error:
                    self.events.put(
                        (
                            "worker_error",
                            f"{exc}; additionally failed to save attempt status: {save_error}",
                        )
                    )
            self.events.put(("worker_error", str(exc)))
        finally:
            if self.mode == "testing" and self.current_attempt_id is not None:
                try:
                    self._finish_active_attempt(
                        "interrupted",
                        error="Camera worker stopped before the attempt was completed.",
                    )
                except Exception as exc:
                    self.events.put(
                        ("worker_error", f"Could not persist interrupted attempt: {exc}")
                    )
            self._stop_event.set()
            if self.capture.is_alive():
                self.capture.join()
            with self._recording_lock:
                self._recording = False
            self.events.put(("worker_stopped", self.mode))

    def _publish_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_version += 1

    def _process_commands(
        self,
        frames: list[np.ndarray],
        frame_buffer: collections.deque[np.ndarray],
    ) -> None:
        while True:
            try:
                command, attempt_id = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "stop":
                self._stop_event.set()
                return
            if command == "toggle_landmarks" and self.mode == "demo":
                self.show_landmarks = not self.show_landmarks
            if command == "clear_subtitle" and self.mode == "demo":
                self.subtitle_manager.clear()
                self._latest_commit = None
                self.events.put(("subtitle_cleared", self.commit_sequence))
                self.events.put(("subtitle", ""))
            elif command == "start_attempt" and self.mode == "testing":
                if attempt_id is None and self.current_attempt_id is None:
                    self.events.put(("worker_error", "No initialized attempt is attached."))
                    continue
                if attempt_id is not None:
                    self.current_attempt_id = attempt_id
                frames.clear()
                frame_buffer.clear()
                self.subtitle_manager.clear()
                self._latest_commit = None
                self._candidate_label = ""
                self._candidate_started_at = None
                self._stable_predictions = []
                self._attempt_inference_latencies = []
                with self._recording_lock:
                    self._recording = True
                self.events.put(("attempt_started", self.current_attempt_id))
            elif command == "finish_attempt" and self.mode == "testing":
                with self._recording_lock:
                    was_recording = self._recording
                    self._recording = False
                if was_recording:
                    self._finish_attempt(frames)

    def _recognize_live(
        self,
        frame_buffer: collections.deque[np.ndarray],
        landmarks: np.ndarray,
    ) -> None:
        hands_present = not (
            np.isnan(landmarks[C.LH, 0]).all()
            and np.isnan(landmarks[C.RH, 0]).all()
        )
        self._hands_present = hands_present
        predictions: list[CanonicalPrediction] = []
        if len(frame_buffer) == C.SEQ_LEN:
            preprocess_started = time.perf_counter()
            positions, present = preprocess_sequence(np.stack(frame_buffer))
            features = torch.from_numpy(build_features(positions, present))[None]
            self.performance.add(
                "sequence_preprocessing",
                (time.perf_counter() - preprocess_started) * 1000,
            )
            classifier_started = time.perf_counter()
            with torch.no_grad():
                probabilities = torch.softmax(self.model(features), dim=1)[0]
            predictions = aggregate_predictions(probabilities, self.vocab, top_k=3)
            inference_ms = (time.perf_counter() - classifier_started) * 1000
            self.performance.add("model_inference", inference_ms)
            if self.mode == "testing" and self.recording:
                self._attempt_inference_latencies.append(inference_ms)
        self._display_predictions = predictions
        if predictions:
            best = predictions[0]
            now = time.perf_counter()
            if (
                hands_present
                and self.subtitle_manager.state in (State.IDLE, State.CANDIDATE)
                and best.confidence >= self.subtitle_manager.conf_threshold
            ):
                if self._candidate_label != best.canonical_label:
                    self._candidate_label = best.canonical_label
                    self._candidate_started_at = now
            else:
                self._candidate_label = ""
                self._candidate_started_at = None
            stabilization_started = time.perf_counter()
            newly_committed, _ = self.subtitle_manager.update(
                best.canonical_label,
                best.confidence,
                hands_present,
                original_model_label=best.original_model_label,
                class_index=best.class_index,
            )
            self.performance.add(
                "stabilization",
                (time.perf_counter() - stabilization_started) * 1000,
            )
            self.events.put(("prediction", (predictions, hands_present)))
            if newly_committed is not None:
                committed_prediction = next(
                    item
                    for item in predictions
                    if item.canonical_label == newly_committed
                )
                if self.mode == "testing" and self.current_attempt_id is not None:
                    self._stable_predictions = predictions
                    if self.store is None:
                        raise RuntimeError("Testing attempt storage is unavailable")
                    self.store.record_event(
                        self.session_id,
                        "stable_prediction_confirmed",
                        attempt_id=self.current_attempt_id,
                        details={
                            "canonical_label": committed_prediction.canonical_label,
                            "original_model_label": committed_prediction.original_model_label,
                            "class_index": committed_prediction.class_index,
                            "confidence": committed_prediction.confidence,
                            "top_k": [
                                {
                                    "canonical_label": item.canonical_label,
                                    "original_model_label": item.original_model_label,
                                    "class_index": item.class_index,
                                    "confidence": item.confidence,
                                }
                                for item in predictions
                            ],
                        },
                    )
                    self.events.put(
                        (
                            "stable_prediction",
                            {
                                "attempt_id": self.current_attempt_id,
                                "prediction": committed_prediction,
                                "predictions": predictions,
                                "latency_ms": inference_ms,
                            },
                        )
                    )
                with self._commit_lock:
                    self._commit_sequence += 1
                    sequence = self._commit_sequence
                committed_at = time.perf_counter()
                candidate_started_at = self._candidate_started_at or committed_at
                self._latest_commit = (
                    sequence,
                    newly_committed,
                    candidate_started_at,
                    committed_at,
                )
                if self._candidate_started_at is not None:
                    self.performance.add(
                        "candidate_to_commit",
                        (committed_at - self._candidate_started_at) * 1000,
                    )
                self.performance.add(
                    "confirmation_to_subtitle_update",
                    (time.perf_counter() - committed_at) * 1000,
                )
                self._candidate_label = ""
                self._candidate_started_at = None
                self.events.put(("committed", (sequence, newly_committed)))
                self.events.put(("subtitle", self.subtitle_manager.subtitle))
        else:
            self._candidate_label = ""
            self._candidate_started_at = None
            stabilization_started = time.perf_counter()
            self.subtitle_manager.update(None, 0.0, hands_present)
            self.performance.add(
                "stabilization",
                (time.perf_counter() - stabilization_started) * 1000,
            )
            self.events.put(("prediction", ([], hands_present)))
        if self.mode == "testing" and getattr(self.args, "debug_testing", False):
            candidate = predictions[0] if predictions else None
            stable = self._stable_predictions[0] if self._stable_predictions else None
            self.events.put(
                (
                    "debug_state",
                    {
                        "session_id": self.session_id,
                        "attempt_id": self.current_attempt_id,
                        "target": self.target_label,
                        "attempt_state": (
                            "in_progress" if self.current_attempt_id is not None else "idle"
                        ),
                        "candidate": candidate.canonical_label if candidate else "none",
                        "stable": stable.canonical_label if stable else "none",
                        "confidence": candidate.confidence if candidate else 0.0,
                        "stabilization": self.subtitle_manager.state.name,
                        "latency_ms": self.performance.mean("model_inference"),
                    },
                )
            )

    def _draw_demo_overlay(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        fps: float,
        landmark_ms: float,
        classifier_ms: float,
    ) -> None:
        height, width = frame.shape[:2]
        if self.show_landmarks:
            draw_landmark_overlay(frame, landmarks)

        draw_filled_rect(frame, 0, 0, width, 36, BLACK, 0.6)
        draw_text(
            frame,
            f"FPS {fps:.0f}  |  LM {landmark_ms:.0f}ms  CLS {classifier_ms:.0f}ms",
            (10, 24),
            0.6,
            GRAY,
        )

        draw_filled_rect(frame, width - 160, 0, width, 36, BLACK, 0.6)
        state = self.subtitle_manager.state
        draw_text(
            frame,
            STATE_LABEL[state],
            (width - 150, 24),
            0.6,
            STATE_COLOR[state],
        )

        panel_y = 44
        draw_filled_rect(frame, 0, panel_y, width, panel_y + 80, BLACK, 0.5)
        if not self._hands_present:
            draw_text(frame, "No hands detected", (10, panel_y + 28), 0.7, ORANGE)
        elif self._display_predictions:
            best = self._display_predictions[0]
            color = GREEN if best.confidence >= self.args.thr else ORANGE
            draw_text(
                frame,
                f"{best.canonical_label}  {best.confidence * 100:.0f}%",
                (10, panel_y + 28),
                0.9,
                color,
            )
            for index, prediction in enumerate(self._display_predictions[1:], 1):
                draw_text(
                    frame,
                    f"  {index + 1}. {prediction.canonical_label} "
                    f"{prediction.confidence * 100:.0f}%",
                    (10, panel_y + 28 + 26 * index),
                    0.55,
                    GRAY,
                )

        if state == State.CANDIDATE:
            draw_text(
                frame,
                f"[{self.subtitle_manager.candidate_word}]",
                (10, panel_y + 68),
                0.6,
                YELLOW,
            )
            draw_progress_bar(
                frame,
                120,
                panel_y + 55,
                200,
                12,
                self.subtitle_manager.candidate_progress,
                GREEN,
            )

        subtitle_text = self.subtitle_manager.subtitle or "(subtitle will appear here)"
        subtitle_height = 90
        draw_filled_rect(
            frame, 0, height - subtitle_height, width, height, BLACK, 0.7
        )
        cv2.line(
            frame,
            (0, height - subtitle_height),
            (width, height - subtitle_height),
            (80, 80, 80),
            1,
        )
        draw_text(
            frame, "SUBTITLE:", (10, height - subtitle_height + 22), 0.5, GRAY, 1
        )
        lines = wrap_text(subtitle_text, 0.75, 2, width - 20)
        for line_index, line in enumerate(lines[-2:]):
            draw_text(
                frame,
                line,
                (10, height - subtitle_height + 48 + line_index * 30),
                0.75,
                WHITE,
            )
        draw_text(
            frame,
            "C=clear  L=landmarks  Q=quit",
            (width - 270, height - 8),
            0.45,
            GRAY,
            1,
        )

        if self._latest_commit and time.perf_counter() - self._latest_commit[3] < 0.5:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), GREEN, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            draw_text(
                frame,
                f"+ {self._latest_commit[0]}",
                (width // 2 - 80, height // 2),
                1.2,
                GREEN,
            )

    def _finish_attempt(self, frames: list[np.ndarray]) -> None:
        if self.current_attempt_id is None:
            self.events.put(("worker_error", "No active attempt can be completed."))
            return
        attempt_id = self.current_attempt_id
        if len(frames) < 8:
            self._finish_active_attempt(
                "invalid",
                error=f"Attempt contained only {len(frames)} usable frames.",
            )
            self.events.put(
                (
                    "attempt_result",
                    {
                        "attempt_id": attempt_id,
                        "status": "invalid",
                        "frames": len(frames),
                        "target_label": self.target_label,
                    },
                )
            )
            return
        if not self._stable_predictions:
            self._finish_active_attempt(
                "no_prediction",
                warning="No prediction reached the configured stabilization threshold.",
            )
            self.events.put(
                (
                    "attempt_result",
                    {
                        "attempt_id": attempt_id,
                        "status": "no_prediction",
                        "frames": len(frames),
                        "target_label": self.target_label,
                    },
                )
            )
            return
        predictions = self._stable_predictions
        best = predictions[0]
        self._finish_active_attempt(
            "completed",
            predicted_label=best.canonical_label,
            original_model_label=best.original_model_label,
            canonical_label_value=best.canonical_label,
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
            confidence=best.confidence,
            inference_latency_ms=(
                statistics.fmean(self._attempt_inference_latencies)
                if self._attempt_inference_latencies else None
            ),
        )
        self.events.put(
            (
                "attempt_result",
                {
                    "attempt_id": attempt_id,
                    "status": "completed",
                    "predictions": predictions,
                    "frames": len(frames),
                    "target_label": self.target_label,
                },
            )
        )

    def _finish_active_attempt(
        self,
        status: str,
        *,
        error: str | None = None,
        warning: str | None = None,
        predicted_label: str = "",
        original_model_label: str = "",
        canonical_label_value: str = "",
        class_index: int | None = None,
        top_k_predictions: list[dict] | None = None,
        confidence: float = 0.0,
        inference_latency_ms: float | None = None,
    ) -> None:
        attempt_id = self.current_attempt_id
        if attempt_id is None:
            return
        if self.store is None:
            raise RuntimeError("Testing attempt storage is unavailable")
        saved_status = self.store.finish_attempt(
            attempt_id,
            status=status,
            predicted_label=predicted_label,
            original_model_label=original_model_label,
            canonical_label_value=canonical_label_value,
            class_index=class_index,
            top_k_predictions=top_k_predictions,
            confidence=confidence,
            inference_latency_ms=inference_latency_ms,
            error=error,
            warning=warning,
        )
        self.current_attempt_id = None
        self.events.put(
            ("attempt_persisted", {"attempt_id": attempt_id, "status": saved_status})
        )


class ASLApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("ASL Learning Studio")
        self.root.geometry("1280x720")
        self.root.minsize(960, 540)
        self.root.configure(bg=BACKGROUND)

        self.model, self.vocab = self._load_model(Path(args.ckpt))
        self.model_version = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
        self.vocabulary_version = hashlib.sha256(
            "\n".join(self.vocab).encode("utf-8")
        ).hexdigest()
        self.session_id: str | None = None
        self.attempt_store = RecognitionAttemptStore(Path(args.attempt_db))
        manifest_path = Path(args.demo_manifest)
        self.catalog, self.missing_demos = load_catalog(self.vocab, manifest_path)
        self._missing_report_error = self._update_missing_report()

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.subtitle_manager = SubtitleManager(
            min_votes=args.min_votes,
            conf_threshold=args.thr,
            cooldown_sec=args.cooldown,
        )
        self.worker: RecognitionWorker | None = None
        self._worker_frame_version = 0
        self._pending_mode: str | None = None
        self._transitioning = False
        self._closing = False
        self._commit_floor = 0
        self._ignore_commits_until_clear_ack = False
        self._sound_enabled = tk.BooleanVar(value=True)
        self._current_mode = "demo"
        self._last_error = ""
        self._initial_target_error = ""
        self._video_capture: cv2.VideoCapture | None = None
        self._video_after_id: str | None = None
        self._video_playing = False
        self._video_last_tick = 0.0
        self._demo_watched = False
        self._demo_watch_duration = 0.0
        self._selected_label = ""
        self._ui_performance = PerformanceMonitor()
        self._speech_commit_times: dict[str, collections.deque[float]] = (
            collections.defaultdict(collections.deque)
        )
        self._last_performance_report = time.perf_counter()

        self.speech = SpeechService(
            lambda error: self.events.put(("speech_error", error)),
            on_started=lambda text, started_at: self.events.put(
                ("speech_started", (text, started_at))
            ),
            on_ready=lambda duration: self.events.put(
                ("speech_engine_ready", duration)
            ),
        )
        self._configure_style()
        self._build_interface()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        self.root.after(30, self._poll)
        initial_mode = "testing" if args.mode == "testing" else "demo"
        if initial_mode == "testing":
            self._select_initial_target()
        self._show_mode(initial_mode)
        status_messages = [
            message
            for message in (
                self._initial_target_error,
                self._missing_report_error,
            )
            if message
        ]
        if status_messages:
            self.status_var.set(" · ".join(status_messages))

    @staticmethod
    def _load_model(checkpoint_path: Path) -> tuple[SeqClassifier, list[str]]:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        torch.set_num_threads(min(2, torch.get_num_threads()))
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_args = checkpoint["args"]
        model = SeqClassifier(
            checkpoint["feature_dim"],
            len(checkpoint["vocab"]),
            checkpoint_args["hidden"],
            2,
            checkpoint_args["dropout"],
            checkpoint_args["kind"],
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, list(checkpoint["vocab"])

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), foreground=INK)
        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Preview.TFrame", background=PREVIEW_BG)
        style.configure("Title.TLabel", background=BACKGROUND, foreground=INK,
                        font=("Segoe UI", 19, "bold"))
        style.configure("Subheading.TLabel", background=CARD, foreground=INK,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Body.TLabel", background=CARD, foreground=INK)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED)
        style.configure("Status.TLabel", background=BACKGROUND, foreground=MUTED)
        style.configure("Accent.TButton", background=ACCENT, foreground="black",
                        padding=(16, 10), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_DARK),
                                                 ("disabled", "#555555")])
        style.configure("Secondary.TButton", background="#303030", foreground=INK,
                        padding=(13, 9))
        style.map("Secondary.TButton", background=[("active", "#444444")])
        style.configure("TButton", background="#303030", foreground=INK)
        style.map("TButton", background=[("active", "#444444")])
        style.configure(
            "TCombobox",
            fieldbackground="#202020",
            background="#303030",
            foreground=INK,
            arrowcolor=INK,
        )
        style.map("TCombobox", fieldbackground=[("readonly", "#202020")],
                  foreground=[("readonly", INK)])
        style.configure("TEntry", fieldbackground="#202020", foreground=INK)
        style.configure("TSpinbox", fieldbackground="#202020", foreground=INK)
        style.configure("TSeparator", background="#444444")
        style.configure("Mode.TButton", padding=(13, 8))
        style.configure("SelectedMode.TButton", background="#333333",
                        foreground=ACCENT, padding=(13, 8),
                        font=("Segoe UI", 10, "bold"))
        style.configure("TProgressbar", troughcolor="#333333",
                        background=ACCENT, thickness=10)

    def _card(self, parent: tk.Misc, **kwargs: Any) -> ttk.Frame:
        return ttk.Frame(parent, style="Card.TFrame", padding=18, **kwargs)

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, style="App.TFrame", padding=(18, 12, 18, 10))
        self.header = header
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        brand = ttk.Frame(header, style="App.TFrame")
        brand.grid(row=0, column=0, sticky="w")
        ttk.Label(brand, text="ASL Learning Studio", style="Title.TLabel").pack(
            side="left"
        )

        self.page_container = ttk.Frame(self.root, style="App.TFrame", padding=0)
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)
        self.demo_page = self._build_demo_page(self.page_container)
        self.testing_page = self._build_testing_page(self.page_container)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
            padding=(18, 6, 18, 8),
        )
        self.status_label.grid(row=2, column=0, sticky="ew")

        self.camera_index = tk.IntVar(value=self.args.camera)
        self.mode_switch_button = tk.Button(
            self.root,
            text="Testing Mode",
            command=self._switch_mode,
            relief="flat",
            borderwidth=0,
            bg="#292929",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.mode_switch_button.place(relx=1.0, x=-10, y=6, anchor="ne")
        self.sound_button = tk.Button(
            self.root,
            text="Sound: ON",
            command=self._toggle_sound,
            relief="flat",
            borderwidth=0,
            bg="#292929",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
            cursor="hand2",
        )
        self.sound_button.place(relx=1.0, x=-120, y=6, anchor="ne")

    def _build_demo_page(self, parent: tk.Misc) -> ttk.Frame:
        page = ttk.Frame(parent, style="App.TFrame")
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        self.demo_preview = VideoPanel(page, height=720)
        self.demo_preview.grid(row=0, column=0, sticky="nsew")
        return page

    def _build_testing_page(self, parent: tk.Misc) -> ttk.Frame:
        page = ttk.Frame(parent, style="App.TFrame")
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(1, weight=1)
        heading = ttk.Frame(page, style="App.TFrame")
        heading.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 13))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="Guided testing", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        history_button = ttk.Button(
            heading,
            text="History / verify",
            style="Secondary.TButton",
            command=self._show_attempt_history,
        )
        history_button.grid(row=0, column=2, padx=(10, 0))
        report_button = ttk.Button(
            heading,
            text="Testing report",
            style="Secondary.TButton",
            command=self._show_testing_report,
        )
        report_button.grid(row=0, column=3, padx=(8, 0))
        self.demo_count_label = ttk.Label(
            heading,
            text=f"{len(self.catalog)} of {len(self.vocab)} verified demos",
            style="Status.TLabel",
        )
        self.demo_count_label.grid(row=0, column=1, sticky="e")
        self.debug_testing_label = ttk.Label(
            page,
            text="",
            style="Muted.TLabel",
            wraplength=950,
            justify="left",
        )
        if getattr(self.args, "debug_testing", False):
            self.debug_testing_label.grid(
                row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
            )

        reference_card = self._card(page)
        reference_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        reference_card.columnconfigure(0, weight=1)
        reference_card.rowconfigure(2, weight=1)
        ttk.Label(
            reference_card, text="Target sign", style="Subheading.TLabel"
        ).grid(row=0, column=0, sticky="w")
        target_row = ttk.Frame(reference_card, style="Card.TFrame")
        target_row.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        target_row.columnconfigure(1, weight=1)
        ttk.Label(target_row, text="Choose a target", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 9)
        )
        self.target_var = tk.StringVar()
        self._target_values: dict[str, str] = {}
        self.target_selector = ttk.Combobox(
            target_row, textvariable=self.target_var, state="readonly"
        )
        self.target_selector.grid(row=0, column=1, sticky="ew")
        self.target_selector.bind("<<ComboboxSelected>>", self._target_changed)
        self.target_label_display = ttk.Label(
            reference_card, text="No target selected", style="Title.TLabel",
            font=("Segoe UI", 22, "bold"),
        )
        self.target_label_display.grid(row=2, column=0, sticky="nw", pady=(2, 10))
        self.reference_preview = VideoPanel(reference_card, height=300)
        self.reference_preview.grid(row=3, column=0, sticky="nsew")
        demo_controls = ttk.Frame(reference_card, style="Card.TFrame")
        demo_controls.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.replay_button = ttk.Button(
            demo_controls, text="Play demonstration", style="Secondary.TButton",
            command=self._toggle_demo_video,
        )
        self.replay_button.pack(side="left")
        self.start_attempt_button = ttk.Button(
            demo_controls, text="Start attempt", style="Accent.TButton",
            command=self._start_test_attempt,
        )
        self.start_attempt_button.pack(side="right")
        self.demo_video_status = ttk.Label(
            reference_card, text="Choose a target to view its demonstration.",
            style="Muted.TLabel", wraplength=460,
        )
        self.demo_video_status.grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.missing_report_button = ttk.Button(
            reference_card,
            text="Open missing-demo list",
            style="Secondary.TButton",
            command=self._open_missing_report,
        )
        self.missing_report_button.grid(row=6, column=0, sticky="w", pady=(8, 0))

        attempt_card = self._card(page)
        attempt_card.grid(row=1, column=1, sticky="nsew")
        attempt_card.columnconfigure(0, weight=1)
        attempt_card.rowconfigure(1, weight=1)
        camera_header = ttk.Frame(attempt_card, style="Card.TFrame")
        camera_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        camera_header.columnconfigure(0, weight=1)
        ttk.Label(camera_header, text="Your attempt", style="Subheading.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.test_camera_status = ttk.Label(
            camera_header, text="Camera off", style="Muted.TLabel"
        )
        self.test_camera_status.grid(row=0, column=1, sticky="e")
        self.test_stop_button = ttk.Button(
            camera_header,
            text="Stop camera",
            style="Secondary.TButton",
            command=self._stop_test_camera,
            state="disabled",
        )
        self.test_stop_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.test_preview = VideoPanel(attempt_card, height=390)
        self.test_preview.grid(row=1, column=0, sticky="nsew")
        self.test_attempt_button = ttk.Button(
            attempt_card, text="Camera starts when you begin",
            style="Accent.TButton", command=self._toggle_test_recording,
            state="disabled",
        )
        self.test_attempt_button.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.test_result = ttk.Label(
            attempt_card,
            text="Your prediction and confidence will appear here.",
            style="Body.TLabel",
            wraplength=430,
            justify="left",
            font=("Segoe UI", 12),
        )
        self.test_result.grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            attempt_card,
            text="Saved attempts remain unverified until reviewed independently.",
            style="Muted.TLabel",
            wraplength=430,
        ).grid(row=4, column=0, sticky="w", pady=(9, 0))

        self._populate_targets()
        return page

    def _populate_targets(self) -> None:
        values: list[str] = []
        for label in self.vocab:
            display = f"{canonical_label(label)}  ·  {label}"
            values.append(display)
            self._target_values[display] = label
        self.target_selector.configure(values=values)
        if not values:
            self.target_selector.configure(state="disabled")
            self.replay_button.configure(state="disabled")
            self.start_attempt_button.configure(state="disabled")
            self.target_label_display.configure(text="No target labels available")
            self.reference_preview.show_placeholder("The checkpoint vocabulary is empty.")
        else:
            self.target_selector.configure(state="readonly")

    def _select_initial_target(self) -> None:
        if self.args.target:
            wanted = self.args.target.strip().upper()
            if wanted not in self.vocab:
                self._initial_target_error = (
                    f"{wanted} is not in this model's vocabulary."
                )
                self.target_label_display.configure(text="Target not in vocabulary")
                self.demo_video_status.configure(text=self._initial_target_error)
                return
            if wanted in self.vocab:
                self._set_target_value(wanted)
                return
            self._initial_target_error = f"{wanted} is not in this model's vocabulary."
            self.target_label_display.configure(text="Target not in vocabulary")
            self.demo_video_status.configure(text=self._initial_target_error)
            return
        if self.vocab:
            self._set_target_value(self.vocab[0])

    def _set_target_value(self, label: str) -> None:
        for value, raw_label in self._target_values.items():
            if raw_label == label:
                self.target_var.set(value)
                self._selected_label = label
                return

    def _target_changed(self, _: tk.Event | None = None) -> None:
        label = self._target_values.get(self.target_var.get())
        if label is None:
            return
        if self.worker is not None:
            self.status_var.set("Stop the current attempt before changing its target.")
            self._set_target_value(self._selected_label)
            return
        self._selected_label = label
        if self.session_id is not None:
            self.attempt_store.record_event(
                self.session_id,
                "target_selected",
                details={"target_label": label},
            )
        self._load_target_video()

    def _load_target_video(self) -> None:
        self._stop_demo_video()
        self._demo_watched = False
        self._demo_watch_duration = 0.0
        if not self._selected_label:
            return
        self.target_label_display.configure(text=canonical_label(self._selected_label))
        demo = self.catalog.get(self._selected_label)
        if demo is None:
            reason = self.missing_demos.get(self._selected_label, "not catalogued")
            self.reference_preview.show_placeholder(
                "No verified reference video is available.\n"
                "You may still perform this target; it will be reported as no-reference."
            )
            self.demo_video_status.configure(
                text=f"Optional reference unavailable: {reason}"
            )
            self.replay_button.configure(state="disabled")
            self.start_attempt_button.configure(state="normal")
            self.missing_report_button.grid()
            return
        self.reference_preview.show_placeholder("Press Play demonstration to watch.")
        self.demo_video_status.configure(
            text=f"Verified reference · version {demo.version} · replay as often as needed."
        )
        self.replay_button.configure(state="normal", text="Play demonstration")
        self.start_attempt_button.configure(state="normal")
        self.missing_report_button.grid_remove()
        self._open_demo_video(demo)

    def _ensure_testing_session(self) -> None:
        if self.session_id is not None:
            return
        self.session_id = uuid.uuid4().hex
        self.attempt_store.start_session(
            self.session_id,
            model_version=self.model_version,
            vocabulary_version=self.vocabulary_version,
            configuration={
                "checkpoint": Path(self.args.ckpt).name,
                "confidence_threshold": self.args.thr,
                "minimum_votes": self.args.min_votes,
                "cooldown_seconds": self.args.cooldown,
                "preprocessing_version": C.PREPROCESSING_VERSION,
            },
        )

    def _create_test_attempt(self) -> int:
        self._ensure_testing_session()
        if self.session_id is None or not self._selected_label:
            raise RuntimeError("A testing session and target label are required")
        demo = self.catalog.get(self._selected_label)
        return self.attempt_store.begin_attempt(
            session_id=self.session_id,
            target_label=self._selected_label,
            demo_video_id=demo.video_id if demo else "",
            demo_video_version=demo.version if demo else "",
            demo_watched=self._demo_watched,
            demo_watch_duration=self._demo_watch_duration,
            model_version=self.model_version,
            preprocessing_version=C.PREPROCESSING_VERSION,
        )

    def _show_testing_report(self) -> None:
        try:
            sessions = self.attempt_store.sessions()
            session = None
            attempts = []
            if self.session_id is not None:
                session = self.attempt_store.get_session(self.session_id)
                attempts = self.attempt_store.attempts(self.session_id)
            if not attempts:
                for candidate in sessions:
                    candidate_attempts = self.attempt_store.attempts(
                        candidate["session_id"]
                    )
                    if candidate_attempts:
                        session = candidate
                        attempts = candidate_attempts
                        break
            text = format_report(build_report(attempts, session=session))
        except (OSError, sqlite3.Error, KeyError, ValueError) as exc:
            messagebox.showerror("Testing report", str(exc), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Testing Mode report")
        window.geometry("760x620")
        body = tk.Text(
            window,
            wrap="word",
            bg=CARD,
            fg=INK,
            insertbackground=INK,
            font=("Consolas", 10),
            padx=14,
            pady=12,
        )
        body.pack(fill="both", expand=True)
        body.insert("1.0", text)
        body.configure(state="disabled")

    def _show_attempt_history(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Testing history and verification")
        window.geometry("1080x680")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        table = ttk.Treeview(
            window,
            columns=(
                "session",
                "time",
                "target",
                "prediction",
                "status",
                "ground_truth",
                "confidence",
            ),
            show="headings",
            selectmode="browse",
        )
        for column, title, width in (
            ("session", "Session ID", 110),
            ("time", "Started", 160),
            ("target", "Target", 140),
            ("prediction", "Prediction", 150),
            ("status", "Status", 110),
            ("ground_truth", "Verified ground truth", 180),
            ("confidence", "Confidence", 90),
        ):
            table.heading(column, text=title)
            table.column(column, width=width, stretch=column in {"target", "prediction"})
        table.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        detail = tk.Text(
            window,
            height=14,
            wrap="word",
            bg=CARD,
            fg=INK,
            insertbackground=INK,
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        detail.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        window.rowconfigure(1, weight=1)
        controls = ttk.Frame(window, padding=12)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        def refresh(selected_id: int | None = None) -> None:
            for item in table.get_children():
                table.delete(item)
            for row in self.attempt_store.attempts():
                confidence = (
                    f"{row['confidence']:.1%}" if row["predicted_label"] else "—"
                )
                table.insert(
                    "",
                    "end",
                    iid=str(row["attempt_id"]),
                    values=(
                        row["session_id"],
                        row["started_at"] or row["created_at"],
                        row["target_label"],
                        row["canonical_label"] or row["predicted_label"] or "—",
                        row["status"],
                        row["ground_truth_label"] or "unverified",
                        confidence,
                    ),
                )
            if selected_id is not None and table.exists(str(selected_id)):
                table.selection_set(str(selected_id))
                show_detail()

        def show_detail(_: tk.Event | None = None) -> None:
            selected = table.selection()
            detail.configure(state="normal")
            detail.delete("1.0", "end")
            if not selected:
                detail.configure(state="disabled")
                return
            attempt_id = int(selected[0])
            row = self.attempt_store.get_attempt(attempt_id)
            top_k = json.loads(row["top_k_predictions"] or "[]")
            lines = [
                f"Session: {row['session_id']}    Attempt: {attempt_id}",
                f"Target: {row['target_label']}    Demo: "
                f"{row['demo_video_id'] or 'not available'} "
                f"{row['demo_video_version'] or ''}",
                f"Model: {row['model_version']}    Preprocessing: "
                f"{row['preprocessing_version']}",
                f"Started: {row['started_at']}    Ended: {row['ended_at']}    "
                f"Duration: {row['duration_seconds']}",
                f"Raw prediction: {row['original_model_label'] or 'none'} "
                f"(class {row['class_index']})",
                f"Canonical prediction: {row['canonical_label'] or 'none'} "
                f"({row['confidence']:.1%})",
                f"Top-K: {json.dumps(top_k, ensure_ascii=False)}",
                f"Ground truth: {row['ground_truth_label'] or 'unverified'}    "
                f"Reviewer: {row['verified_by'] or '—'}    "
                f"Verified: {row['verified_at'] or '—'}",
                f"Status: {row['status']}    Error: {row['error'] or '—'}    "
                f"Warning: {row['warning'] or '—'}",
                "",
                "EVENT TRACE",
            ]
            for event in self.attempt_store.attempt_events(attempt_id):
                lines.append(
                    f"{event['occurred_at']}  {event['event_type']}  "
                    f"attempt={event['attempt_id'] or 'session'}  "
                    f"{event['details_json']}"
                )
            lines.append("")
            lines.append("VERIFICATION HISTORY")
            for review in self.attempt_store.review_history(attempt_id):
                lines.append(
                    f"{review['reviewed_at']} {review['action']} by "
                    f"{review['reviewer']}: {review['previous_ground_truth_label']} -> "
                    f"{review['new_ground_truth_label']} "
                    f"({review['previous_correctness_status']} -> "
                    f"{review['new_correctness_status']}); {review['reason'] or 'no reason'}"
                )
            detail.insert("1.0", "\n".join(lines))
            detail.configure(state="disabled")

        def verify_selected() -> None:
            selected = table.selection()
            if not selected:
                messagebox.showinfo(
                    "Verify ground truth", "Select an attempt first.", parent=window
                )
                return
            ground_truth = simpledialog.askstring(
                "Verify ground truth",
                "Enter the independently observed actual sign (not the requested target):",
                parent=window,
            )
            if ground_truth is None:
                return
            reviewer = simpledialog.askstring(
                "Reviewer",
                "Reviewer identifier:",
                parent=window,
            )
            if reviewer is None:
                return
            attempt_id = int(selected[0])
            try:
                status = self.attempt_store.verify_ground_truth(
                    attempt_id, ground_truth, reviewer
                )
            except (KeyError, ValueError, sqlite3.Error) as exc:
                messagebox.showerror("Verification failed", str(exc), parent=window)
                return
            messagebox.showinfo(
                "Ground truth saved",
                f"Attempt #{attempt_id}: {status} against verified ground truth.",
                parent=window,
            )
            refresh(attempt_id)

        table.bind("<<TreeviewSelect>>", show_detail)
        ttk.Button(
            controls,
            text="Verify selected attempt",
            command=verify_selected,
        ).pack(side="left")
        ttk.Button(controls, text="Refresh", command=refresh).pack(side="left", padx=8)
        ttk.Button(controls, text="Close", command=window.destroy).pack(side="right")
        refresh()

    def _open_missing_report(self) -> None:
        report_path = C.MISSING_SIGN_DEMOS_REPORT
        self._missing_report_error = self._update_missing_report()
        if self._missing_report_error:
            self.status_var.set(self._missing_report_error)
            messagebox.showerror(
                "Missing-demo report",
                self._missing_report_error,
                parent=self.root,
            )
            return
        try:
            os.startfile(str(report_path))
        except OSError as exc:
            self._missing_report_error = (
                f"Could not open missing-demo report {report_path}: {exc}"
            )
            self.status_var.set(self._missing_report_error)
            messagebox.showerror(
                "Missing-demo report",
                self._missing_report_error,
                parent=self.root,
            )

    def _update_missing_report(self) -> str:
        try:
            write_missing_report(self.missing_demos)
        except OSError as exc:
            return f"Could not update missing-demo report: {exc}"
        return ""

    def _open_demo_video(self, demo: SignDemo) -> None:
        self._video_capture = cv2.VideoCapture(str(demo.path))
        if not self._video_capture.isOpened():
            self._video_capture.release()
            self._video_capture = None
            self.demo_video_status.configure(
                text="The verified demo could not be opened. Check the MP4 file."
            )
            self.replay_button.configure(state="disabled")
            self.start_attempt_button.configure(state="disabled")
            return
        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        delay = max(15, round(1000 / fps)) if np.isfinite(fps) and fps > 0 else 40
        self._video_delay_ms = delay
        self._video_playing = True
        self._video_last_tick = 0.0
        self.replay_button.configure(text="Pause demonstration")
        self._schedule_video_frame(0)

    def _schedule_video_frame(self, delay: int | None = None) -> None:
        if self._video_after_id is not None:
            self.root.after_cancel(self._video_after_id)
        self._video_after_id = self.root.after(
            self._video_delay_ms if delay is None else delay,
            self._play_video_frame,
        )

    def _play_video_frame(self) -> None:
        self._video_after_id = None
        if not self._video_playing or self._video_capture is None:
            return
        ok, frame = self._video_capture.read()
        if not ok:
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._video_capture.read()
        if not ok or frame is None:
            self.demo_video_status.configure(
                text="The demo has no readable frames. Check the MP4 before using it."
            )
            self._stop_demo_video()
            return
        now = time.perf_counter()
        if self._video_last_tick:
            self._demo_watch_duration += min(
                max(0.0, now - self._video_last_tick),
                self._video_delay_ms / 1000 * 1.5,
            )
        self._video_last_tick = now
        if not self._demo_watched and self.session_id is not None:
            self.attempt_store.record_event(
                self.session_id,
                "demo_watched",
                details={
                    "target_label": self._selected_label,
                    "demo_video_id": (
                        self.catalog[self._selected_label].video_id
                        if self._selected_label in self.catalog else None
                    ),
                },
            )
        self._demo_watched = True
        self.reference_preview.show_frame(frame)
        self._schedule_video_frame()

    def _toggle_demo_video(self) -> None:
        if self._video_playing:
            self._stop_demo_video()
            self.replay_button.configure(text="Replay demonstration")
            return
        if not self._selected_label:
            return
        self._open_demo_video(self.catalog[self._selected_label])

    def _stop_demo_video(self) -> None:
        self._video_playing = False
        self._video_last_tick = 0.0
        if self._video_after_id is not None:
            self.root.after_cancel(self._video_after_id)
            self._video_after_id = None
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def _bind_shortcuts(self) -> None:
        self.root.bind("c", lambda _: self._clear_subtitle())
        self.root.bind("l", lambda _: self._toggle_landmarks())
        self.root.bind("<Control-l>", lambda _: self._clear_subtitle())
        self.root.bind("<Escape>", lambda _: self._request_close())
        self.root.bind("q", lambda _: self._request_close())

    def _show_mode(self, mode: str) -> None:
        previous_mode = self._current_mode
        if mode == "testing" or previous_mode == "testing":
            self.subtitle_manager.clear()
        self._current_mode = mode
        if mode == "testing":
            self._ensure_testing_session()
        self._stop_demo_video()
        self.demo_page.grid_remove()
        self.testing_page.grid_remove()
        page = self.demo_page if mode == "demo" else self.testing_page
        page.grid(row=0, column=0, sticky="nsew")
        self.mode_switch_button.configure(
            text="Testing Mode" if mode == "demo" else "Demo Mode"
        )
        if mode == "demo":
            self.header.grid_remove()
            self.status_label.grid_remove()
            self.mode_switch_button.place_configure(y=40)
            self.sound_button.place_configure(y=40)
        else:
            self.header.grid()
            self.status_label.grid()
            self.mode_switch_button.place_configure(y=6)
            self.sound_button.place_configure(y=6)
        self.speech.cancel_pending()
        self._speech_commit_times.clear()
        if mode == "testing" and self._selected_label:
            self._load_target_video()
        if mode == "testing":
            self.demo_count_label.configure(
                text=f"{len(self.catalog)} of {len(self.vocab)} verified demos"
            )
        self.status_var.set(
            "C=clear subtitle · L=toggle landmarks · Q=quit"
            if mode == "demo"
            else (
                "Guided test · reference demos are optional; reviewed ground truth is required for accuracy."
                if self.vocab
                else "Guided testing is unavailable because the checkpoint vocabulary is empty."
            )
        )
        if mode == "testing" and self._missing_report_error:
            self.status_var.set(
                f"{self.status_var.get()} · {self._missing_report_error}"
            )
        if mode == "demo" and self.worker is None and not self._closing:
            self._last_error = ""
            self.demo_preview.show_placeholder("Connecting to camera…")
            self._start_worker(mode="demo", camera_index=self.args.camera)

    def _switch_mode(self) -> None:
        next_mode = "testing" if self._current_mode == "demo" else "demo"
        self._request_mode(next_mode)

    def _toggle_landmarks(self) -> None:
        if self.worker is not None and self.worker.mode == "demo":
            self.worker.command("toggle_landmarks")

    def _request_mode(self, mode: str) -> None:
        if mode == self._current_mode or self._transitioning:
            return
        if self.worker is not None:
            if self.worker.current_attempt_id is not None and not messagebox.askyesno(
                "Discard unfinished attempt?",
                "Switching modes will discard this unfinished attempt. "
                "It will be retained as cancelled. Continue?",
                parent=self.root,
            ):
                return
            if self.worker.current_attempt_id is not None:
                self._end_active_attempt("cancelled", "Mode changed before completion.")
            self._transitioning = True
            self._pending_mode = mode
            self.speech.cancel_pending()
            self._speech_commit_times.clear()
            self._stop_demo_video()
            self.worker.stop()
            self.status_var.set("Closing the camera before switching modes…")
            return
        self._show_mode(mode)

    def _toggle_demo_camera(self) -> None:
        if self.worker is not None:
            if self.worker.mode == "demo":
                self.worker.stop()
                self.status_var.set("Stopping camera…")
            return
        try:
            camera_index = int(self.camera_index.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Camera", "Enter a valid camera index.", parent=self.root)
            return
        self._start_worker(mode="demo", camera_index=camera_index)

    def _start_worker(
        self,
        *,
        mode: str,
        camera_index: int,
        demo: SignDemo | None = None,
        target_label: str | None = None,
        attempt_id: int | None = None,
    ) -> RecognitionWorker | None:
        if self.worker is not None:
            return None
        self._worker_frame_version = 0
        self.worker = RecognitionWorker(
            mode=mode,
            camera_index=camera_index,
            model=self.model,
            vocab=self.vocab,
            args=self.args,
            subtitle_manager=self.subtitle_manager,
            events=self.events,
            demo=demo,
            target_label=target_label,
            demo_watched=self._demo_watched,
            demo_watch_duration=self._demo_watch_duration,
            model_version=self.model_version,
            session_id=self.session_id or "",
            store=self.attempt_store,
            attempt_id=attempt_id,
        )
        self.worker.start()
        if mode == "demo":
            self.demo_preview.show_placeholder("Connecting to camera…")
        else:
            self.test_attempt_button.configure(
                text="Starting camera…", state="disabled"
            )
            self.test_stop_button.configure(state="normal")
            self.target_selector.configure(state="disabled")
            self.test_camera_status.configure(text="Connecting…")
            self.test_preview.show_placeholder("Connecting to camera…")
        return self.worker

    def _start_test_attempt(self) -> None:
        if self.worker is not None:
            self.status_var.set("Stop the current camera before starting another test.")
            return
        if not self._selected_label or self._selected_label not in self.vocab:
            self.status_var.set("Select a target label before testing.")
            return
        self._stop_demo_video()
        try:
            camera_index = int(self.camera_index.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Camera", "Enter a valid camera index.", parent=self.root)
            return
        self.test_result.configure(
            text="Attempt in progress. Perform one sign, then choose Finish attempt."
        )
        self.subtitle_manager.clear()
        try:
            attempt_id = self._create_test_attempt()
        except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
            messagebox.showerror("Testing attempt", str(exc), parent=self.root)
            return
        worker = self._start_worker(
            mode="testing",
            camera_index=camera_index,
            demo=self.catalog.get(self._selected_label),
            target_label=self._selected_label,
            attempt_id=attempt_id,
        )
        if worker is not None:
            worker.command("start_attempt", attempt_id)

    def _toggle_test_recording(self) -> None:
        if self.worker is None:
            self._start_test_attempt()
            return
        if self.worker.mode != "testing":
            return
        if self.worker.recording:
            self.worker.command("finish_attempt")
            self.test_attempt_button.configure(
                text="Finishing…", state="disabled"
            )
            self.test_camera_status.configure(text="Processing attempt…")
        else:
            self.subtitle_manager.clear()
            try:
                attempt_id = self._create_test_attempt()
            except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
                messagebox.showerror("Testing attempt", str(exc), parent=self.root)
                return
            self.worker.command("start_attempt", attempt_id)
            self.test_attempt_button.configure(text="Starting…", state="disabled")

    def _stop_test_camera(self) -> None:
        if self.worker is None or self.worker.mode != "testing":
            return
        if self.worker.current_attempt_id is not None and not messagebox.askyesno(
            "Discard unfinished attempt?",
            "Stopping now will retain this attempt as cancelled. Continue?",
            parent=self.root,
        ):
            return
        if self.worker.current_attempt_id is not None:
            self._end_active_attempt("cancelled", "Camera stopped before completion.")
        self.worker.stop()
        self.test_stop_button.configure(state="disabled")
        self.status_var.set("Stopping camera…")

    def _end_active_attempt(self, status: str, reason: str) -> None:
        if self.worker is None or self.worker.current_attempt_id is None:
            return
        attempt_id = self.worker.current_attempt_id
        saved_status = self.attempt_store.finish_attempt(
            attempt_id,
            status=status,
            error=reason,
        )
        self.worker.current_attempt_id = None
        self.status_var.set(f"Attempt #{attempt_id} saved as {saved_status}.")
        context = f": {reason}" if saved_status == status else ""
        self.test_result.configure(
            text=f"Attempt #{attempt_id} saved as {saved_status}{context}"
        )

    def _clear_subtitle(self) -> None:
        self.speech.cancel_pending()
        self._speech_commit_times.clear()
        if self.worker is not None and self.worker.mode == "demo":
            self._ignore_commits_until_clear_ack = True
            self.worker.command("clear_subtitle")
        else:
            self.subtitle_manager.clear()
            if self.worker is not None:
                self._commit_floor = self.worker.commit_sequence

    def _toggle_sound(self) -> None:
        self._sound_enabled.set(not self._sound_enabled.get())
        enabled = self._sound_enabled.get()
        self.speech.set_enabled(enabled)
        if not enabled:
            self._speech_commit_times.clear()
        self.sound_button.configure(text=f"Sound: {'ON' if enabled else 'OFF'}")
        self.status_var.set(
            "Speech enabled · only newly committed signs are spoken."
            if enabled
            else "Speech muted · recognition and subtitles continue normally."
        )

    def _poll(self) -> None:
        if self._closing:
            if self.worker is None or not self.worker.is_alive():
                self.root.destroy()
                return
            self.root.after(40, self._poll)
            return

        self._drain_events()
        if self.worker is not None:
            version, frame = self.worker.latest_frame(self._worker_frame_version)
            if frame is not None:
                self._worker_frame_version = version
                preview = (
                    self.demo_preview
                    if self.worker.mode == "demo"
                    else self.test_preview
                )
                preview.show_frame(frame)
                commit = self.worker.commit_info
                if commit is not None and commit[0] > self.worker._last_rendered_commit:
                    now = time.perf_counter()
                    self._ui_performance.add(
                        "commit_to_ui_preview",
                        (now - commit[3]) * 1000,
                    )
                    self._ui_performance.add(
                        "prediction_to_subtitle_preview",
                        (now - commit[2]) * 1000,
                    )
                    self.worker._last_rendered_commit = commit[0]
            if not self.worker.is_alive():
                stopped_mode = self.worker.mode
                self.worker = None
                if stopped_mode == "demo":
                    placeholder = self._last_error or "Camera is off"
                    self.demo_preview.show_placeholder(placeholder)
                else:
                    self.test_camera_status.configure(text="Camera off")
                    self.test_attempt_button.configure(
                        text="Camera starts when you begin", state="normal"
                    )
                    self.test_stop_button.configure(state="disabled")
                    self.target_selector.configure(
                        state="readonly" if self.vocab else "disabled"
                    )
                if self._transitioning:
                    self._transitioning = False
                    pending_mode = self._pending_mode
                    self._pending_mode = None
                    if pending_mode is not None:
                        self._show_mode(pending_mode)
            elif self.worker.mode == "testing" and self.worker.recording:
                self.test_attempt_button.configure(
                    text="Finish attempt", state="normal"
                )
                self.test_camera_status.configure(text="Recording")
        if self.args.debug_performance:
            now = time.perf_counter()
            if now - self._last_performance_report >= 5.0:
                details = (
                    self.worker.performance.report()
                    if self.worker is not None
                    else "camera worker stopped"
                )
                ui_details = self._ui_performance.report()
                print(
                    f"[performance] {details}"
                    + (f" | UI/TTS {ui_details}" if ui_details else ""),
                    flush=True,
                )
                self._last_performance_report = now
        self.root.after(30, self._poll)

    def _drain_events(self) -> None:
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                return
            if event == "camera_ready":
                if value == "demo":
                    self._last_error = ""
                else:
                    self.test_camera_status.configure(text="Camera active")
                    self.test_attempt_button.configure(
                        text="Finish attempt", state="normal"
                    )
                    self.status_var.set("Camera active · perform the target sign.")
            elif event in ("prediction", "subtitle"):
                pass
            elif event == "committed":
                sequence, label = value
                if self.worker is None or self.worker.mode != "demo":
                    self.events.task_done()
                    continue
                if (
                    self._transitioning
                    or self._ignore_commits_until_clear_ack
                    or sequence <= self._commit_floor
                ):
                    self.events.task_done()
                    continue
                key = spoken_text(label)
                self._speech_commit_times[key].append(time.perf_counter())
                if not self._sound_enabled.get() or not self.speech.enqueue(label):
                    self._speech_commit_times[key].pop()
            elif event == "subtitle_cleared":
                self._commit_floor = max(self._commit_floor, value)
                self._ignore_commits_until_clear_ack = False
            elif event == "attempt_started":
                self.test_attempt_button.configure(
                    text="Finish attempt", state="normal"
                )
                self.test_camera_status.configure(text="Recording")
                self.status_var.set(
                    f"Attempt #{value} started · finish when the sign is complete."
                )
            elif event == "recording_progress":
                self.test_camera_status.configure(text=f"Recording · {value} frames")
            elif event == "attempt_too_short":
                self.test_attempt_button.configure(text="Retry attempt", state="normal")
                self.test_camera_status.configure(text="Attempt too short")
                self.test_result.configure(
                    text=f"Only {value} frames were captured. Continue, then finish the attempt."
                )
            elif event == "attempt_result":
                result = value
                if result["status"] == "completed":
                    best = result["predictions"][0]
                    self.test_result.configure(
                        text=(
                            f"Stable prediction: {best.canonical_label}\n"
                            f"Confidence: {best.confidence:.0%}\n"
                            f"Saved attempt #{result['attempt_id']} · "
                            "ground truth is unverified"
                        )
                    )
                elif result["status"] == "no_prediction":
                    self.test_result.configure(
                        text=(
                            f"No stable prediction was confirmed for attempt "
                            f"#{result['attempt_id']}. It is saved separately and "
                            "excluded from accuracy."
                        )
                    )
                else:
                    self.test_result.configure(
                        text=(
                            f"Attempt #{result['attempt_id']} is {result['status']} "
                            f"({result['frames']} frames) and excluded from accuracy."
                        )
                    )
                self.test_attempt_button.configure(
                    text="Start another attempt", state="normal"
                )
                self.test_camera_status.configure(text="Attempt saved")
                self.status_var.set(
                    f"Attempt #{result['attempt_id']} saved as {result['status']}."
                )
            elif event == "stable_prediction":
                result = value
                prediction = result["prediction"]
                self.test_result.configure(
                    text=(
                        f"Stable prediction: {prediction.canonical_label} "
                        f"({prediction.confidence:.0%}) · attempt "
                        f"#{result['attempt_id']}"
                    )
                )
            elif event == "attempt_persisted":
                if getattr(self.args, "debug_testing", False):
                    self.debug_testing_label.configure(
                        text=(
                            f"Database save: attempt #{value['attempt_id']} "
                            f"{value['status']}"
                        )
                    )
                if value["status"] in {
                    "cancelled",
                    "interrupted",
                    "error",
                    "invalid",
                }:
                    self.test_result.configure(
                        text=(
                            f"Attempt #{value['attempt_id']} saved as "
                            f"{value['status']} and excluded from accuracy."
                        )
                    )
                    self.status_var.set(
                        f"Attempt #{value['attempt_id']} saved as {value['status']}."
                    )
            elif event == "debug_state":
                if getattr(self.args, "debug_testing", False):
                    self.debug_testing_label.configure(
                        text=(
                            f"Session {value['session_id']} · attempt "
                            f"{value['attempt_id'] or '—'} · target {value['target']} · "
                            f"state {value['attempt_state']} · "
                            f"candidate {value['candidate']} · stable {value['stable']} · "
                            f"confidence {value['confidence']:.1%} · "
                            f"stabilization {value['stabilization']} · "
                            f"inference {value['latency_ms']:.1f} ms"
                        )
                    )
            elif event == "worker_error":
                self._last_error = value
                self.status_var.set(value)
                if self.worker is not None and self.worker.mode == "testing":
                    self.test_camera_status.configure(text="Camera unavailable")
                    self.test_result.configure(text=value)
            elif event == "speech_engine_ready":
                self._ui_performance.add("speech_engine_startup", value)
            elif event == "speech_error":
                self._speech_commit_times.clear()
                self.status_var.set(
                    f"Speech unavailable; recognition continues. {value}"
                )
            elif event == "speech_started":
                text, started_at = value
                pending = self._speech_commit_times.get(text)
                if pending:
                    committed_at = pending.popleft()
                    self._ui_performance.add(
                        "commit_to_speech_start",
                        (started_at - committed_at) * 1000,
                    )
                    if not pending:
                        self._speech_commit_times.pop(text, None)
            elif event == "worker_stopped":
                if value == "demo":
                    self.demo_preview.show_placeholder(
                        self._last_error or "Camera is off"
                    )
                else:
                    self.test_preview.show_placeholder("Camera is off")
            self.events.task_done()

    def _request_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_demo_video()
        self.speech.close()
        if self.session_id is not None:
            try:
                session_status = "completed"
                if self.worker is not None and self.worker.current_attempt_id is not None:
                    self._end_active_attempt(
                        "interrupted", "Application closed before attempt completion."
                    )
                    session_status = "interrupted"
                self.attempt_store.end_session(self.session_id, session_status)
            except (KeyError, ValueError, sqlite3.Error) as exc:
                messagebox.showerror("Testing session", str(exc), parent=self.root)
        if self.worker is not None:
            self.worker.stop()
        else:
            self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASL Learning Studio: free practice and guided sign testing"
    )
    parser.add_argument("--ckpt", default=str(C.MODELS / "gru_seed3.pt"))
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=("demo", "recognition", "testing"), default="demo")
    parser.add_argument("--target", default="", help="initial testing target label")
    parser.add_argument("--min-votes", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument(
        "--debug-performance",
        action="store_true",
        help="print rolling pipeline timing metrics every five seconds",
    )
    parser.add_argument(
        "--debug-testing",
        action="store_true",
        help="show Testing Mode attempt state and persistence diagnostics",
    )
    parser.add_argument("--demo-manifest", default=str(C.SIGN_DEMO_MANIFEST))
    parser.add_argument("--attempt-db", default=str(C.RECOGNITION_ATTEMPTS_DB))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        ASLApplication(args).root.mainloop()
    except (FileNotFoundError, KeyError, RuntimeError, sqlite3.Error) as exc:
        messagebox.showerror("ASL Learning Studio", str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
