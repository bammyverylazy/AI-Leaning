import argparse
import collections
import queue
import sys
import threading
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from application import ASLApplication, CameraCapture, PerformanceMonitor, RecognitionWorker, build_parser
from dataset import build_features, preprocess_sequence
from attempt_store import RecognitionAttemptStore
from self_recorded_store import SelfRecordedSampleStore
from subtitle import SubtitleManager


class RecognitionWorkerTests(unittest.TestCase):
    def test_camera_stop_releases_a_blocked_read(self):
        class BlockingCapture:
            def __init__(self):
                self.read_started = threading.Event()
                self.released = threading.Event()

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                self.read_started.set()
                self.released.wait(3)
                return False, None

            def release(self):
                self.released.set()

        capture = BlockingCapture()
        events = queue.Queue()
        stop_event = threading.Event()
        camera = CameraCapture(
            0,
            queue.Queue(maxsize=1),
            stop_event,
            events,
            PerformanceMonitor(),
            "demo",
        )
        with patch("application.cv2.VideoCapture", return_value=capture):
            camera.start()
            self.assertTrue(capture.read_started.wait(1))
            camera.stop()
            camera.join(1)

        self.assertFalse(camera.is_alive())
        self.assertTrue(capture.released.is_set())
        self.assertTrue(stop_event.is_set())
        self.assertEqual(events.get_nowait(), ("camera_ready", "demo"))

    def test_camera_worker_can_restart_after_open_failure(self):
        class ClosedCapture:
            def isOpened(self):
                return False

            def release(self):
                pass

        class BlockingCapture:
            def __init__(self):
                self.released = threading.Event()

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                self.released.wait(3)
                return False, None

            def release(self):
                self.released.set()

        events = queue.Queue()
        failed = CameraCapture(
            0, queue.Queue(maxsize=1), threading.Event(), events,
            PerformanceMonitor(), "demo"
        )
        with patch(
            "application.cv2.VideoCapture",
            side_effect=[ClosedCapture(), ClosedCapture()],
        ):
            failed.start()
            failed.join(1)
        self.assertFalse(failed.is_alive())
        self.assertEqual(events.get_nowait()[0], "worker_error")

        events = queue.Queue()
        capture = BlockingCapture()
        retry = CameraCapture(
            0, queue.Queue(maxsize=1), threading.Event(), events,
            PerformanceMonitor(), "demo"
        )
        with patch("application.cv2.VideoCapture", return_value=capture):
            retry.start()
            self.assertEqual(events.get(timeout=1), ("camera_ready", "demo"))
            retry.stop()
            retry.join(1)
        self.assertFalse(retry.is_alive())
        self.assertTrue(capture.released.is_set())

    def test_collection_finish_saves_candidate_and_cancel_discards_frames(self):
        events = queue.Queue()
        worker = RecognitionWorker(
            mode="collection",
            camera_index=0,
            model=lambda _features: torch.tensor([[1.0]]),
            vocab=["SIGN"],
            args=argparse.Namespace(thr=0.45, debug_testing=False),
            subtitle_manager=SubtitleManager(),
            events=events,
            model_version="test-model",
            session_id="",
        )
        landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        frames = [landmarks.copy() for _ in range(12)]
        frame_buffer = collections.deque(
            (landmarks.copy() for _ in range(4)), maxlen=C.SEQ_LEN
        )
        worker.collection_sample_id = "finished-sample"
        worker.command("start_collection")
        worker._process_commands(frames, frame_buffer)
        self.assertTrue(worker.recording)
        self.assertEqual(events.get_nowait()[0], "collection_started")
        frames.extend(landmarks.copy() for _ in range(12))

        worker.command("finish_collection")
        worker._process_commands(frames, frame_buffer)
        event, result = events.get_nowait()
        self.assertEqual(event, "collection_ready")
        self.assertEqual(result["sample_id"], "finished-sample")
        self.assertEqual(result["original_frame_count"], 12)
        self.assertEqual(result["raw_landmarks"].shape, (12, C.N_POINTS, 3))
        self.assertFalse(worker.recording)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "cancelled.avi"
            video_path.write_bytes(b"unfinished recording")
            worker.collection_video_path = video_path
            worker.collection_sample_id = "cancelled-sample"
            worker.command("start_collection")
            worker._process_commands(frames, frame_buffer)
            self.assertTrue(worker.recording)
            frames.extend(landmarks.copy() for _ in range(12))
            worker.command("cancel_collection")
            worker._process_commands(frames, frame_buffer)
            self.assertFalse(worker.recording)
            self.assertEqual(frames, [])
            self.assertFalse(video_path.exists())
            self.assertEqual(events.get_nowait()[0], "collection_started")
            self.assertEqual(events.get_nowait(), ("collection_cancelled", None))
            self.assertTrue(events.empty())

    def test_repeated_demo_testing_worker_lifecycle_is_clean(self):
        class FakeCapture:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                if self.released:
                    return False, None
                time.sleep(0.002)
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class FakeLandmarker:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def detect_for_video(self, *_args):
                return None

        active_captures = 0
        max_active_captures = 0
        capture_lock = threading.Lock()
        captures = []

        def create_capture(*_args):
            nonlocal active_captures, max_active_captures
            capture = FakeCapture()
            with capture_lock:
                active_captures += 1
                max_active_captures = max(max_active_captures, active_captures)
            original_release = capture.release

            def release():
                nonlocal active_captures
                if not capture.released:
                    with capture_lock:
                        active_captures -= 1
                original_release()

            capture.release = release
            captures.append(capture)
            return capture

        landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        landmarks[C.LH, :] = (0.4, 0.5, 0.0)
        landmarks[C.RH, :] = (0.6, 0.5, 0.0)
        landmarks[C.L_SH, :] = (0.35, 0.4, 0.0)
        landmarks[C.R_SH, :] = (0.65, 0.4, 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = RecognitionAttemptStore(Path(temp_dir) / "attempts.sqlite3")
            session_id = store.start_session(
                "transition-session",
                model_version="test-model",
                vocabulary_version="test-vocab",
                configuration={},
            )
            with (
                patch("application.cv2.VideoCapture", side_effect=create_capture),
                patch("application.make_landmarker", return_value=FakeLandmarker()),
                patch("application.result_to_array", return_value=landmarks),
            ):
                for cycle in range(5):
                    for mode in ("demo", "testing", "demo"):
                        events = queue.Queue()
                        attempt_id = None
                        if mode == "testing":
                            attempt_id = store.begin_attempt(
                                session_id=session_id,
                                target_label="SIGN_A",
                                model_version="test-model",
                                preprocessing_version="test-preprocessing",
                            )
                        worker = RecognitionWorker(
                            mode=mode,
                            camera_index=0,
                            model=lambda _features: torch.tensor([[2.0, 0.0]]),
                            vocab=["SIGN_A", "SIGN_B"],
                            args=argparse.Namespace(
                                thr=0.45,
                                debug_testing=False,
                            ),
                            subtitle_manager=SubtitleManager(
                                min_votes=1, idle_frames=1, cooldown_sec=0
                            ),
                            events=events,
                            model_version="test-model",
                            session_id=session_id,
                            store=store if mode == "testing" else None,
                            attempt_id=attempt_id,
                        )
                        worker.start()
                        if attempt_id is not None:
                            worker.command("start_attempt", attempt_id)
                        expected_event = "committed" if mode == "demo" else "prediction"
                        deadline = time.monotonic() + 3
                        saw_prediction = False
                        while time.monotonic() < deadline and not saw_prediction:
                            try:
                                event, _ = events.get(timeout=0.05)
                            except queue.Empty:
                                continue
                            saw_prediction = event == expected_event
                        worker.stop()
                        worker.join(2)
                        self.assertFalse(worker.is_alive())
                        self.assertTrue(saw_prediction, f"{mode} did not predict")
                        frame_version, frame = worker.latest_frame(0)
                        self.assertGreater(frame_version, 0)
                        self.assertIsNotNone(frame)
                        self.assertTrue(worker.capture.stop_event.is_set())
                        self.assertTrue(captures[-1].released)
                        self.assertEqual(active_captures, 0)
            store.end_session(session_id)

        self.assertEqual(len(captures), 15)
        self.assertEqual(max_active_captures, 1)
        self.assertFalse(
            any(
                thread.is_alive()
                and thread.name in {"asl-camera-capture", "asl-recognition"}
                for thread in threading.enumerate()
            )
        )

    def test_live_hand_activity_requires_complete_hand_landmarks(self):
        events = queue.Queue()
        manager = MagicMock()
        worker = RecognitionWorker(
            mode="demo",
            camera_index=0,
            model=lambda _features: torch.tensor([[1.0]]),
            vocab=["SIGN"],
            args=argparse.Namespace(),
            subtitle_manager=manager,
            events=events,
            model_version="test-model",
            session_id="test-session",
        )
        sequence = collections.deque(maxlen=C.SEQ_LEN)

        landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        landmarks[C.LH.start, 0] = 0.5  # one point is not a complete hand
        worker._recognize_live(sequence, landmarks)
        self.assertFalse(worker._hands_present)
        self.assertFalse(manager.update.call_args.args[2])

        landmarks[C.LH, 0] = 0.5  # all 21 left-hand x coordinates are present
        worker._recognize_live(sequence, landmarks)
        self.assertTrue(worker._hands_present)
        self.assertTrue(manager.update.call_args.args[2])

        landmarks[C.LH.start, 0] = np.nan
        landmarks[C.RH, 0] = 0.5  # a complete right hand is sufficient
        worker._recognize_live(sequence, landmarks)
        self.assertTrue(worker._hands_present)
        self.assertTrue(manager.update.call_args.args[2])

    def test_live_tensor_matches_offline_features_for_same_window(self):
        class CapturingModel:
            def __init__(self):
                self.input = None

            def __call__(self, features):
                self.input = features.detach().clone()
                return torch.tensor([[2.0, 1.0]], dtype=torch.float32)

        rng = np.random.default_rng(20261105)
        window = rng.normal(0.0, 0.25, size=(C.SEQ_LEN, C.N_POINTS, 3)).astype(
            np.float32
        )
        window[:, C.L_SH, :] = np.array([0.35, 0.4, 0.02], dtype=np.float32)
        window[:, C.R_SH, :] = np.array([0.65, 0.4, 0.02], dtype=np.float32)
        window[:, C.LH, 0] = rng.uniform(0.2, 0.8, size=(C.SEQ_LEN, C.N_HAND))
        window[:, C.RH, 0] = rng.uniform(0.2, 0.8, size=(C.SEQ_LEN, C.N_HAND))
        frames = collections.deque(
            (frame.copy() for frame in window), maxlen=C.SEQ_LEN
        )
        capture_model = CapturingModel()
        worker = RecognitionWorker(
            mode="demo",
            camera_index=0,
            model=capture_model,
            vocab=["SIGN_A", "SIGN_B"],
            args=argparse.Namespace(),
            subtitle_manager=SubtitleManager(),
            events=queue.Queue(),
            model_version="test-model",
            session_id="test-session",
        )

        worker._recognize_live(frames, frames[-1])

        offline_pos, offline_present = preprocess_sequence(window)
        offline_features = build_features(offline_pos, offline_present)
        expected = torch.from_numpy(offline_features)[None]
        self.assertIsNotNone(capture_model.input)
        self.assertEqual(tuple(capture_model.input.shape), (1, C.SEQ_LEN, 590))
        self.assertEqual(capture_model.input.dtype, torch.float32)
        self.assertEqual(expected.dtype, torch.float32)
        self.assertTrue(torch.equal(capture_model.input, expected))

    def test_worker_emits_speech_eligible_event_only_after_subtitle_commit(self):
        events = queue.Queue()
        manager = SubtitleManager(min_votes=2, idle_frames=1, cooldown_sec=0)
        worker = RecognitionWorker(
            mode="demo",
            camera_index=0,
            model=lambda _features: torch.tensor([[4.0, 0.0]]),
            vocab=["WHATFOR1", "CHRISTMAS1"],
            args=argparse.Namespace(),
            subtitle_manager=manager,
            events=events,
            model_version="test-model",
            session_id="test-session",
        )
        sequence = collections.deque(
            [np.zeros((C.N_POINTS, 3), dtype=np.float32) for _ in range(C.SEQ_LEN)],
            maxlen=C.SEQ_LEN,
        )
        landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        landmarks[C.LH, 0] = 0.5

        with (
            patch(
                "application.preprocess_sequence",
                return_value=(np.zeros((C.SEQ_LEN, C.N_POINTS, 3)), np.zeros((C.SEQ_LEN, 2))),
            ),
            patch(
                "application.build_features",
                return_value=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
            ),
            patch(
                "application.save_attempt_sequence",
                return_value="data/evaluation/sequences/1.npz",
            ),
        ):
            worker._recognize_live(sequence, landmarks)
            first_events = self._drain(events)
            self.assertFalse(any(name == "committed" for name, _ in first_events))

            worker._recognize_live(sequence, landmarks)
            committed_events = [
                value for name, value in self._drain(events) if name == "committed"
            ]
            self.assertEqual(len(committed_events), 1)
            self.assertEqual(committed_events[0][1], "WHAT FOR")
            self.assertEqual(manager.subtitle, "WHAT FOR")

            worker._recognize_live(sequence, landmarks)
            self.assertFalse(
                any(name == "committed" for name, _ in self._drain(events))
            )

    def test_testing_worker_saves_only_live_stabilized_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RecognitionAttemptStore(Path(temp_dir) / "attempts.sqlite3")
            session_id = store.start_session(
                "session",
                model_version="model",
                vocabulary_version="vocab",
                configuration={},
            )
            attempt_id = store.begin_attempt(
                session_id=session_id,
                target_label="AXE1",
                model_version="model",
                preprocessing_version="pre-v1",
            )
            events = queue.Queue()
            manager = SubtitleManager(min_votes=2, idle_frames=1, cooldown_sec=0)
            worker = RecognitionWorker(
                mode="testing",
                camera_index=0,
                model=lambda _features: torch.tensor([[4.0, 0.0]]),
                vocab=["WHATFOR1", "CHRISTMAS1"],
                args=argparse.Namespace(thr=0.45, debug_testing=False),
                subtitle_manager=manager,
                events=events,
                target_label="AXE1",
                model_version="model",
                session_id=session_id,
                store=store,
                attempt_id=attempt_id,
            )
            sequence = collections.deque(
                [
                    np.zeros((C.N_POINTS, 3), dtype=np.float32)
                    for _ in range(C.SEQ_LEN)
                ],
                maxlen=C.SEQ_LEN,
            )
            landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
            landmarks[C.LH, 0] = 0.5
            recorded_frames = [landmarks]
            worker.command("start_attempt", attempt_id)
            worker._process_commands(recorded_frames, sequence)
            self.assertEqual(recorded_frames, [])
            self.assertEqual(len(sequence), 0)
            self.assertTrue(worker.recording)
            sequence.extend(
                np.zeros((C.N_POINTS, 3), dtype=np.float32)
                for _ in range(C.SEQ_LEN)
            )

            with (
                patch(
                    "application.preprocess_sequence",
                    return_value=(
                        np.zeros((C.SEQ_LEN, C.N_POINTS, 3)),
                        np.zeros((C.SEQ_LEN, 2)),
                    ),
                ),
                patch(
                    "application.build_features",
                    return_value=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
                ),
            ):
                worker._recognize_live(sequence, landmarks)
                worker._recognize_live(sequence, landmarks)
                worker._finish_attempt([landmarks] * C.SEQ_LEN)

            saved = store.get_attempt(attempt_id)
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["canonical_label"], "WHAT FOR")
            self.assertEqual(saved["original_model_label"], "WHATFOR1")
            self.assertEqual(saved["class_index"], 0)
            self.assertEqual(
                saved["sequence_path"], "data/evaluation/sequences/1.npz"
            )
            self.assertEqual(saved["sequence_length"], C.SEQ_LEN)
            self.assertEqual(saved["feature_dimension"], 590)
            self.assertEqual(saved["frame_count"], C.SEQ_LEN)
            trace = store.attempt_events(attempt_id)
            self.assertIn(
                "stable_prediction_confirmed",
                [row["event_type"] for row in trace],
            )

    @staticmethod
    def _drain(events):
        collected = []
        while not events.empty():
            collected.append(events.get_nowait())
            events.task_done()
        return collected


class ApplicationArgumentsTests(unittest.TestCase):
    def test_missing_saved_sequence_remains_rejectable_from_review_queue(self):
        app = ASLApplication.__new__(ASLApplication)
        app.collection_review_queue = MagicMock()
        app.collection_review_queue.selection.return_value = ("pending-1",)
        app._collection_preview_result = None
        app._collection_camera_state = "off"
        app._collection_review_after_id = None
        app._collection_review_frame = None
        app._collection_review_landmarks = None
        app._collection_pending_id = None
        app._collection_last_accepted_id = ""
        app._collection_recording_state = "idle"
        app.recording_store = MagicMock()
        app.recording_store.get.return_value = {
            "sample_id": "pending-1",
            "status": "pending_review",
            "target_label": "AXE1",
            "ground_truth_label": None,
            "signer_id": "S001",
            "session_id": "session-1",
            "recorded_at": "2025-01-01T10:00:00+00:00",
            "camera_source": "opencv-index:0",
            "camera_angle": "front",
            "distance": "normal",
            "lighting": "normal",
            "signing_speed": "normal",
            "dominant_hand": "UNKNOWN",
            "recording_status": "PENDING_REVIEW",
            "verification_status": "UNVERIFIED",
            "quality_status": "PASSED",
            "rejection_reason": None,
            "sequence_path": "missing.npz",
        }
        app.recording_store.resolve_artifact_path.return_value = Path("missing.npz")
        app.collection_status_var = MagicMock()
        app.collection_camera_status_var = MagicMock()
        app.collection_preview = MagicMock()
        app.collection_sample_detail = MagicMock()
        app.root = MagicMock()
        app._update_collection_buttons = MagicMock()

        with patch(
            "application.load_attempt_sequence",
            side_effect=FileNotFoundError("missing sequence"),
        ):
            app._select_collection_review_sample()

        self.assertEqual(app._collection_pending_id, "pending-1")
        self.assertEqual(app._collection_recording_state, "review")
        app.collection_preview.show_placeholder.assert_called_once()
        app._update_collection_buttons.assert_called_once()

    def test_collection_controls_are_visible_and_connected_at_desktop_sizes(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        try:
            root.grid_columnconfigure(0, weight=1)
            root.grid_rowconfigure(1, weight=1)
            header = tk.Frame(root, height=55)
            header.grid(row=0, column=0, sticky="ew")
            header.grid_propagate(False)
            parent = tk.Frame(root)
            parent.grid(row=1, column=0, sticky="nsew")
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(0, weight=1)
            footer = tk.Frame(root, height=32)
            footer.grid(row=2, column=0, sticky="ew")
            footer.grid_propagate(False)

            app = ASLApplication.__new__(ASLApplication)
            app.root = root
            app.vocab = ["AXE1"]
            app.catalog = {}
            app.recording_store = MagicMock()
            app.recording_store.signer_ids.return_value = []
            app._collection_pending_id = None
            app._collection_preview_result = None
            app._collection_last_accepted_id = ""
            app._collection_camera_state = "off"
            app._collection_recording_state = "idle"
            app._collection_countdown_remaining = 0
            app._collection_session_id = None
            app._toggle_collection_camera = MagicMock()
            app._start_collection_countdown = MagicMock()
            app._finish_collection_recording = MagicMock()
            app._cancel_collection_recording = MagicMock()
            app._configure_style()
            page = app._build_collection_page(parent)
            page.grid(row=0, column=0, sticky="nsew")

            for width, height in ((1366, 768), (1920, 1080)):
                root.geometry(f"{width}x{height}")
                root.update()
                canvas = app.collection_controls_canvas
                self.assertGreater(canvas.winfo_height(), 100)
                self.assertEqual(
                    app.collection_preview.image.cget("text"), "Camera is off"
                )
                self.assertTrue(app.collection_review_queue.winfo_ismapped())
                for button in (
                    app.collection_camera_button,
                    app.collection_record_button,
                    app.collection_finish_button,
                    app.collection_cancel_button,
                ):
                    self.assertTrue(button.winfo_ismapped())
                    top = button.winfo_rooty() - canvas.winfo_rooty()
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(
                        top + button.winfo_height(), canvas.winfo_height()
                    )
                    self.assertGreater(button.winfo_width(), 40)
                if width == 1366:
                    before_scroll = canvas.yview()[0]
                    app.collection_camera_button.event_generate(
                        "<MouseWheel>", delta=-120
                    )
                    root.update()
                    self.assertGreater(canvas.yview()[0], before_scroll)
                    canvas.yview_moveto(0)

            app.collection_camera_button.invoke()
            self.assertTrue(app._toggle_collection_camera.called)
            for button in (
                app.collection_record_button,
                app.collection_finish_button,
                app.collection_cancel_button,
            ):
                button.configure(state="normal")
                button.invoke()
            self.assertTrue(app._start_collection_countdown.called)
            self.assertTrue(app._finish_collection_recording.called)
            self.assertTrue(app._cancel_collection_recording.called)
        finally:
            root.destroy()

    def test_collection_control_states_follow_camera_and_recording_lifecycle(self):
        app = ASLApplication.__new__(ASLApplication)
        app._collection_camera_state = "off"
        app._collection_recording_state = "idle"
        app._collection_countdown_remaining = 3
        app._collection_pending_id = None
        app._collection_preview_result = None
        app._collection_last_accepted_id = ""
        app._collection_session_id = "session-1"
        app.collection_session_button = MagicMock()
        app.collection_camera_button = MagicMock()
        app.collection_record_button = MagicMock()
        app.collection_finish_button = MagicMock()
        app.collection_cancel_button = MagicMock()
        app.collection_save_button = MagicMock()
        app.collection_discard_preview_button = MagicMock()
        app.collection_accept_button = MagicMock()
        app.collection_reject_button = MagicMock()
        app.collection_verify_button = MagicMock()

        app._update_collection_buttons()
        app.collection_camera_button.configure.assert_called_with(
            text="Start camera", state="normal"
        )
        app.collection_record_button.configure.assert_called_with(
            text="Record sample", state="disabled"
        )
        app.collection_finish_button.configure.assert_called_with(state="disabled")
        app.collection_cancel_button.configure.assert_called_with(state="disabled")
        app.collection_save_button.configure.assert_called_with(state="disabled")
        app.collection_discard_preview_button.configure.assert_called_with(
            state="disabled"
        )

        for recording_state, expected_finish, expected_cancel in (
            ("countdown", "disabled", "normal"),
            ("recording", "normal", "normal"),
            ("saving", "disabled", "disabled"),
            ("review", "disabled", "disabled"),
        ):
            app._collection_camera_state = "ready"
            app._collection_recording_state = recording_state
            app._update_collection_buttons()
            self.assertEqual(
                app.collection_finish_button.configure.call_args.kwargs["state"],
                expected_finish,
            )
            self.assertEqual(
                app.collection_cancel_button.configure.call_args.kwargs["state"],
                expected_cancel,
            )
            self.assertEqual(
                app.collection_record_button.configure.call_args.kwargs["state"],
                "disabled",
            )

        app._collection_recording_state = "idle"
        app._update_collection_buttons()
        self.assertEqual(
            app.collection_record_button.configure.call_args.kwargs["state"],
            "normal",
        )
        app._collection_pending_id = "pending"
        app._update_collection_buttons()
        self.assertEqual(
            app.collection_record_button.configure.call_args.kwargs["state"],
            "disabled",
        )
        app._collection_pending_id = None
        app._collection_last_accepted_id = "accepted-sample"
        app._collection_recording_state = "idle"
        app._update_collection_buttons()
        self.assertEqual(
            app.collection_verify_button.configure.call_args.kwargs["state"],
            "normal",
        )
        app._collection_last_accepted_id = ""
        app._collection_camera_state = "error"
        app._update_collection_buttons()
        app.collection_camera_button.configure.assert_called_with(
            text="Retry camera", state="normal"
        )
        app._collection_camera_state = "ready"
        app._collection_recording_state = "preview"
        app._collection_preview_result = {"sample_id": "preview"}
        app._update_collection_buttons()
        app.collection_save_button.configure.assert_called_with(state="normal")
        app.collection_discard_preview_button.configure.assert_called_with(
            state="normal"
        )
        self.assertEqual(
            app.collection_session_button.configure.call_args.kwargs["state"],
            "disabled",
        )

    def test_collection_session_requires_anonymous_signer_and_generates_stable_id(self):
        app = ASLApplication.__new__(ASLApplication)
        app._collection_pending_id = None
        app._collection_preview_result = None
        app._collection_session_id = None
        app.recording_store = MagicMock()
        app.recording_store.validate_signer_id.side_effect = (
            SelfRecordedSampleStore.validate_signer_id
        )
        app.recording_store.signer_ids.return_value = ["S001"]
        app.recording_store.new_session_id.side_effect = [
            "session-first",
            "session-second",
        ]
        app.collection_signer_selector = MagicMock()
        app.collection_signer_var = MagicMock()
        app.collection_signer_var.get.return_value = ""
        app.collection_session_var = MagicMock()
        app.collection_status_var = MagicMock()
        app._update_collection_buttons = MagicMock()

        app._start_collection_session()
        app.collection_session_var.set.assert_not_called()
        self.assertIsNone(app._collection_session_id)

        app.collection_signer_var.get.return_value = "S001"
        app._start_collection_session()
        self.assertEqual(app._collection_session_id, "session-first")
        app._start_collection_session()
        self.assertEqual(app._collection_session_id, "session-second")

    def test_discarding_collection_preview_creates_no_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "preview.avi"
            video.write_bytes(b"temporary video")
            app = ASLApplication.__new__(ASLApplication)
            app._collection_preview_result = {"raw_video_path": str(video)}
            app._collection_review_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            app._collection_sample_id = "unsaved"
            app._collection_recording_state = "preview"
            app._collection_camera_state = "ready"
            app._collection_pending_id = None
            app._collection_session_id = "session-1"
            app.collection_camera_status_var = MagicMock()
            app.collection_status_var = MagicMock()
            app.collection_preview = MagicMock()
            app.collection_session_button = MagicMock()
            app.collection_camera_button = MagicMock()
            app.collection_record_button = MagicMock()
            app.collection_finish_button = MagicMock()
            app.collection_cancel_button = MagicMock()
            app.collection_save_button = MagicMock()
            app.collection_discard_preview_button = MagicMock()
            app.collection_accept_button = MagicMock()
            app.collection_reject_button = MagicMock()
            app.collection_verify_button = MagicMock()
            app.collection_target_selector = MagicMock()
            app.collection_metadata_widgets = []
            app._set_collection_metadata_enabled = MagicMock()
            app.recording_store = MagicMock()

            app._discard_collection_preview()

            self.assertFalse(video.exists())
            self.assertIsNone(app._collection_preview_result)
            self.assertIsNone(app._collection_review_frame)
            self.assertEqual(app._collection_sample_id, "")
            app.recording_store.create_pending.assert_not_called()
            app.collection_status_var.set.assert_called_once_with(
                "Preview discarded. No candidate or sample record was created."
            )

    def test_collection_preview_save_creates_unverified_pending_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "self_recorded"
            with patch.object(C, "SELF_RECORDING", root), patch.object(
                C, "SELF_RECORDING_DB", root / "samples.sqlite3"
            ):
                app = ASLApplication.__new__(ASLApplication)
                app._collection_sample_id = "pending-review"
                app._collection_pending_id = None
                app._collection_preview_result = None
                app._collection_camera_state = "ready"
                app._collection_recording_state = "saving"
                app._collection_countdown_remaining = 0
                app._collection_metadata = {
                    "target_label": "AXE1",
                    "signer_id": "S001",
                    "session_id": "session-123",
                    "dominant_hand": "RIGHT",
                    "camera_index": 0,
                    "camera_source": "opencv-index:0",
                    "camera_angle": "front",
                    "distance": "normal",
                    "lighting": "normal",
                    "signing_speed": "normal",
                    "notes": "studio check",
                    "recorded_at": "2025-01-01T12:00:00+00:00",
                }
                app.recording_store = SelfRecordedSampleStore()
                app.model_version = "test-model"
                app.collection_status_var = MagicMock()
                app.collection_camera_status_var = MagicMock()
                app.collection_camera_button = MagicMock()
                app.collection_record_button = MagicMock()
                app.collection_finish_button = MagicMock()
                app.collection_cancel_button = MagicMock()
                app.collection_save_button = MagicMock()
                app.collection_discard_preview_button = MagicMock()
                app.collection_target_selector = MagicMock()
                app.collection_metadata_widgets = []
                app.collection_accept_button = MagicMock()
                app.collection_reject_button = MagicMock()
                app.collection_verify_button = MagicMock()
                app.collection_sample_detail = MagicMock()
                app._refresh_collection_dashboard = MagicMock()
                app._update_collection_buttons = MagicMock()
                app._set_collection_metadata_enabled = MagicMock()

                raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
                raw[:, C.LH, :] = (0.4, 0.5, 0.0)
                raw[:, C.RH, :] = (0.6, 0.5, 0.0)
                app._on_collection_ready(
                    {
                        "sample_id": "pending-review",
                        "raw_landmarks": raw,
                        "duration_seconds": 1.0,
                        "original_frame_count": len(raw),
                        "raw_video_path": "",
                    }
                )

                self.assertEqual(app.recording_store.samples(), [])
                self.assertEqual(app._collection_recording_state, "preview")
                app._save_collection_candidate()
                saved = app.recording_store.get("pending-review")
                self.assertEqual(saved["status"], "pending_review")
                self.assertIsNone(saved["ground_truth_label"])
                self.assertEqual(saved["target_label"], "AXE1")
                self.assertEqual(saved["signer_id"], "S001")
                self.assertEqual(saved["session_id"], "session-123")
                self.assertEqual(saved["dominant_hand"], "RIGHT")
                self.assertEqual(saved["recorded_at"], "2025-01-01T12:00:00+00:00")
                self.assertEqual(saved["verification_status"], "UNVERIFIED")
                self.assertEqual(saved["feature_dimension"], 590)
                self.assertEqual(app._collection_pending_id, "pending-review")
                self.assertEqual(app._collection_recording_state, "review")
                app.collection_accept_button.configure.assert_called_with(
                    state="normal"
                )
                app.collection_verify_button.configure.assert_called_with(
                    state="disabled"
                )

    def test_collection_storage_failure_does_not_create_review_candidate(self):
        app = ASLApplication.__new__(ASLApplication)
        app._collection_sample_id = "failed-save"
        app._collection_pending_id = None
        app._collection_preview_result = None
        app._collection_camera_state = "ready"
        app._collection_recording_state = "preview"
        app._collection_countdown_remaining = 0
        app._collection_metadata = {
            "target_label": "AXE1",
            "signer_id": "S001",
            "session_id": "session-123",
            "dominant_hand": "UNKNOWN",
            "camera_index": 0,
            "camera_source": "opencv-index:0",
            "camera_angle": "front",
            "distance": "normal",
            "lighting": "normal",
            "signing_speed": "normal",
            "notes": "",
            "recorded_at": "2025-01-01T12:00:00+00:00",
        }
        app.recording_store = MagicMock()
        app.recording_store.create_pending.side_effect = OSError("disk full")
        app.model_version = "test-model"
        app.collection_status_var = MagicMock()
        app.collection_camera_button = MagicMock()
        app.collection_record_button = MagicMock()
        app.collection_finish_button = MagicMock()
        app.collection_cancel_button = MagicMock()
        app.collection_save_button = MagicMock()
        app.collection_discard_preview_button = MagicMock()
        app.collection_target_selector = MagicMock()
        app.collection_metadata_widgets = []
        app.collection_accept_button = MagicMock()
        app.collection_reject_button = MagicMock()
        app.collection_verify_button = MagicMock()
        app.collection_camera_status_var = MagicMock()
        app.collection_sample_detail = MagicMock()
        app._update_collection_buttons = MagicMock()
        app._set_collection_metadata_enabled = MagicMock()
        app._refresh_collection_dashboard = MagicMock()

        raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
        raw[:, C.LH, :] = (0.4, 0.5, 0.0)
        raw[:, C.RH, :] = (0.6, 0.5, 0.0)
        app._on_collection_ready(
            {
                "sample_id": "failed-save",
                "raw_landmarks": raw,
                "duration_seconds": 1.0,
                "original_frame_count": len(raw),
                "raw_video_path": "",
            }
        )

        self.assertEqual(app.recording_store.create_pending.call_count, 0)
        app._save_collection_candidate()
        app.recording_store.create_pending.assert_called_once()
        self.assertIsNone(app._collection_pending_id)
        self.assertEqual(app._collection_sample_id, "failed-save")
        self.assertEqual(app._collection_recording_state, "preview")
        status = app.collection_status_var.set.call_args.args[0]
        self.assertIn("disk full", status)
        self.assertIn("not saved", status)

    def test_collection_camera_open_failure_reports_error_and_offers_retry(self):
        app = ASLApplication.__new__(ASLApplication)
        app.events = queue.Queue()
        app.events.put(("worker_error", "camera could not be opened"))
        app.worker = MagicMock()
        app.worker.mode = "collection"
        app.worker.latest_frame.return_value = (0, None)
        app.worker.is_alive.return_value = False
        app.args = argparse.Namespace(debug_performance=False)
        app.root = MagicMock()
        app.status_var = MagicMock()
        app.collection_camera_status_var = MagicMock()
        app.collection_status_var = MagicMock()
        app.collection_preview = MagicMock()
        app.collection_camera_button = MagicMock()
        app.collection_record_button = MagicMock()
        app.collection_finish_button = MagicMock()
        app.collection_cancel_button = MagicMock()
        app.collection_save_button = MagicMock()
        app.collection_discard_preview_button = MagicMock()
        app.collection_session_button = MagicMock()
        app.collection_accept_button = MagicMock()
        app.collection_reject_button = MagicMock()
        app.collection_verify_button = MagicMock()
        app.collection_target_selector = MagicMock()
        app.collection_metadata_widgets = []
        app._collection_camera_state = "starting"
        app._collection_recording_state = "idle"
        app._collection_countdown_after = None
        app._collection_pending_id = None
        app._collection_preview_result = None
        app._collection_session_id = None
        app._worker_frame_version = 0
        app._last_error = ""
        app._closing = False
        app._transitioning = False
        app._pending_mode = None

        app._poll()

        app.collection_camera_status_var.set.assert_called_with(
            "Camera unavailable: camera could not be opened"
        )
        app.collection_camera_button.configure.assert_called_with(
            text="Retry camera", state="normal"
        )
        app.collection_record_button.configure.assert_called_with(
            text="Record sample", state="disabled"
        )

    def test_parser_defaults_to_demo_and_accepts_testing_entry_mode(self):
        parser = build_parser()

        self.assertEqual(parser.parse_args([]).mode, "demo")
        self.assertEqual(
            parser.parse_args(["--mode", "testing", "--target", "AXE1"]).target,
            "AXE1",
        )
        self.assertTrue(parser.parse_args(["--debug-performance"]).debug_performance)
        self.assertTrue(parser.parse_args(["--debug-testing"]).debug_testing)

    def test_testing_targets_are_not_filtered_by_missing_references(self):
        app = ASLApplication.__new__(ASLApplication)
        app.vocab = ["AXE1", "DOG1"]
        app.catalog = {}
        app.target_selector = MagicMock()
        app.target_var = MagicMock()
        app._target_values = {}

        app._populate_targets()

        configured = next(
            call.kwargs["values"]
            for call in app.target_selector.configure.call_args_list
            if "values" in call.kwargs
        )
        self.assertEqual(len(configured), 2)
        self.assertIn("AXE (variant 1)", configured[0])

    def test_switching_from_demo_selects_a_testing_target(self):
        app = ASLApplication.__new__(ASLApplication)
        app._current_mode = "demo"
        app._selected_label = ""
        app.vocab = ["AXE1", "DOG1"]
        app.catalog = {}
        app.args = argparse.Namespace(target="")
        app._target_values = {"AXE · AXE1": "AXE1"}
        app.target_var = MagicMock()
        app.subtitle_manager = MagicMock()
        app._ensure_testing_session = MagicMock()
        app._stop_demo_video = MagicMock()
        app._load_target_video = MagicMock()
        app._speech_commit_times = collections.defaultdict(collections.deque)
        app.speech = MagicMock()
        app.demo_page = MagicMock()
        app.testing_page = MagicMock()
        app.collection_page = MagicMock()
        app.mode_switch_button = MagicMock()
        app.sound_button = MagicMock()
        app.header = MagicMock()
        app.status_label = MagicMock()
        app.demo_retry_button = MagicMock()
        app.demo_count_label = MagicMock()
        app.status_var = MagicMock()
        app.worker = None
        app._missing_report_error = ""
        app._closing = False

        app._show_mode("testing")

        self.assertEqual(app._selected_label, "AXE1")
        app.target_var.set.assert_called_once_with("AXE · AXE1")
        app._load_target_video.assert_called_once_with()

    def test_attempt_can_start_when_selected_label_has_no_demo(self):
        app = ASLApplication.__new__(ASLApplication)
        app._stop_demo_video = MagicMock()
        app._demo_watched = True
        app._demo_watch_duration = 5.0
        app._selected_label = "AXE1"
        app.catalog = {}
        app.missing_demos = {"AXE1": "demo not verified"}
        app.target_label_display = MagicMock()
        app.reference_preview = MagicMock()
        app.demo_video_status = MagicMock()
        app.replay_button = MagicMock()
        app.start_attempt_button = MagicMock()
        app.missing_report_button = MagicMock()

        app._load_target_video()

        app.start_attempt_button.configure.assert_called_with(state="normal")
        app.replay_button.configure.assert_called_with(state="disabled")
        app.reference_preview.show_placeholder.assert_called_once()

    def test_target_selector_stays_available_after_camera_stops_without_demos(self):
        app = ASLApplication.__new__(ASLApplication)
        app._closing = False
        app._drain_events = MagicMock()
        app._worker_frame_version = 0
        app.worker = MagicMock()
        app.worker.latest_frame.return_value = (0, None)
        app.worker.is_alive.return_value = False
        app.worker.mode = "testing"
        app.test_camera_status = MagicMock()
        app.test_attempt_button = MagicMock()
        app.test_stop_button = MagicMock()
        app.target_selector = MagicMock()
        app.catalog = {}
        app.vocab = ["AXE1"]
        app._transitioning = False
        app.args = argparse.Namespace(debug_performance=False)
        app.root = MagicMock()

        app._poll()

        app.target_selector.configure.assert_called_once_with(state="readonly")

    def test_missing_report_permission_error_does_not_block_testing(self):
        app = ASLApplication.__new__(ASLApplication)
        app.missing_demos = {"AXE1": "demo not verified"}

        with patch(
            "application.write_missing_report",
            side_effect=PermissionError("report is open"),
        ):
            error = app._update_missing_report()

        self.assertIn("Could not update missing-demo report", error)
        self.assertIn("report is open", error)


if __name__ == "__main__":
    unittest.main()
