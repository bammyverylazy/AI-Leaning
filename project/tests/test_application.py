import argparse
import collections
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from application import ASLApplication, RecognitionWorker, build_parser
from attempt_store import RecognitionAttemptStore
from subtitle import SubtitleManager


class RecognitionWorkerTests(unittest.TestCase):
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
        landmarks[C.LH.start, 0] = 0.5

        with (
            patch(
                "application.preprocess_sequence",
                return_value=(np.zeros((C.SEQ_LEN, C.N_POINTS, 3)), np.zeros((C.SEQ_LEN, 2))),
            ),
            patch(
                "application.build_features",
                return_value=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
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
            landmarks[C.LH.start, 0] = 0.5
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
