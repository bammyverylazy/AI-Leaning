import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attempt_store import RecognitionAttemptStore
from testing_metrics import build_report, format_report


class TestingMetricsTests(unittest.TestCase):
    def test_verified_top1_is_exactly_eighty_percent_of_ten(self):
        attempts = []
        for index in range(10):
            truth = f"SIGN{index}"
            predicted = truth if index < 8 else f"WRONG{index}"
            attempts.append(
                {
                    "attempt_id": index + 1,
                    "status": "completed",
                    "correctness_status": "correct" if index < 8 else "incorrect",
                    "target_label": truth,
                    "predicted_label": predicted,
                    "canonical_label": predicted,
                    "ground_truth_label": truth,
                    "demo_video_id": "demo-1",
                    "top_k_predictions": json.dumps(
                        [
                            {
                                "canonical_label": label,
                                "original_model_label": label,
                                "class_index": rank,
                                "confidence": 0.8 - rank * 0.1,
                            }
                            for rank, label in enumerate(
                                [
                                    predicted,
                                    f"OTHER{index}A",
                                    truth if index >= 8 else f"OTHER{index}B",
                                ]
                            )
                        ]
                    ),
                }
            )

        report = build_report(attempts)

        self.assertEqual(report["verified_top1"]["correct"], 8)
        self.assertEqual(report["verified_top1"]["total"], 10)
        self.assertEqual(report["verified_top1"]["accuracy"], 0.8)
        self.assertEqual(report["counts"]["verified"], 10)
        self.assertEqual(report["verified_top_k"][3]["accuracy"], 1.0)
        self.assertIsNone(report["verified_top_k"][5]["accuracy"])
        self.assertEqual(len(report["common_confusions"]), 2)
        self.assertEqual(report["confusion_matrix"]["sample_count"], 10)
        self.assertIn("8 / 10 attempts", format_report(report))
        self.assertIn("not verified accuracy", format_report(report))

    def test_unverified_invalid_cancelled_and_missing_prediction_are_excluded(self):
        attempts = [
            {
                "status": "completed",
                "correctness_status": "unverified",
                "target_label": "AXE1",
                "predicted_label": "AXE1",
                "canonical_label": "AXE1",
            },
            {
                "status": "invalid",
                "correctness_status": "unverified",
                "target_label": "AXE1",
                "predicted_label": "",
                "canonical_label": "",
            },
            {
                "status": "cancelled",
                "correctness_status": "unverified",
                "target_label": "AXE1",
                "predicted_label": "",
                "canonical_label": "",
            },
            {
                "status": "no_prediction",
                "correctness_status": "unverified",
                "target_label": "AXE1",
                "predicted_label": "",
                "canonical_label": "",
            },
            {
                "status": "completed",
                "correctness_status": "incorrect",
                "target_label": "AXE1",
                "predicted_label": "",
                "canonical_label": "",
                "ground_truth_label": "AXE1",
            },
        ]

        report = build_report(attempts)

        self.assertEqual(report["counts"]["total"], 5)
        self.assertEqual(report["counts"]["valid"], 1)
        self.assertEqual(report["counts"]["unverified"], 1)
        self.assertEqual(report["counts"]["invalid"], 1)
        self.assertEqual(report["counts"]["cancelled"], 1)
        self.assertEqual(report["counts"]["no_prediction"], 1)
        self.assertEqual(report["counts"]["missing_prediction"], 1)
        self.assertIsNone(report["verified_top1"]["accuracy"])
        self.assertEqual(report["target_agreement"]["accuracy"], 1.0)
        self.assertIsNone(report["confusion_matrix"])

    def test_canonical_aliases_are_merged_but_distinct_variants_are_not(self):
        report = build_report(
            [
                {
                    "attempt_id": 1,
                    "status": "completed",
                    "correctness_status": "correct",
                    "target_label": "WHATFOR1",
                    "predicted_label": "WHAT FOR",
                    "canonical_label": "WHAT FOR",
                    "ground_truth_label": "WHATFOR2",
                    "top_k_predictions": "[]",
                },
                {
                    "attempt_id": 2,
                    "status": "completed",
                    "correctness_status": "incorrect",
                    "target_label": "CHRISTMAS2",
                    "predicted_label": "CHRISTMAS (variant 1)",
                    "canonical_label": "CHRISTMAS (variant 1)",
                    "ground_truth_label": "CHRISTMAS2",
                    "top_k_predictions": "[]",
                },
            ]
        )

        self.assertEqual(report["verified_top1"]["correct"], 1)
        self.assertEqual(report["verified_top1"]["total"], 2)
        self.assertEqual(report["target_agreement"]["accuracy"], 0.5)
        self.assertIn("WHATFOR1/2/3", report["label_policy"])
        self.assertIsNone(report["verified_top_k"][3]["accuracy"])

    def test_empty_report_has_explicit_unavailable_metrics(self):
        report = build_report([])

        self.assertEqual(report["counts"]["total"], 0)
        self.assertIsNone(report["target_agreement"]["accuracy"])
        self.assertIsNone(report["verified_top1"]["accuracy"])
        self.assertIn("0 / 0", format_report(report))

    def test_confidence_buckets_use_only_verified_attempts_and_report_coverage(self):
        attempts = [
            {
                "status": "completed",
                "correctness_status": "correct",
                "target_label": "A",
                "predicted_label": "A",
                "canonical_label": "A",
                "ground_truth_label": "A",
                "confidence": 0.9,
                "top_k_predictions": "[]",
            },
            {
                "status": "completed",
                "correctness_status": "incorrect",
                "target_label": "A",
                "predicted_label": "B",
                "canonical_label": "B",
                "ground_truth_label": "A",
                "confidence": 0.6,
                "top_k_predictions": "[]",
            },
            {
                "status": "completed",
                "correctness_status": "unverified",
                "target_label": "A",
                "predicted_label": "A",
                "canonical_label": "A",
                "confidence": 0.1,
                "top_k_predictions": "[]",
            },
            {
                "status": "no_prediction",
                "correctness_status": "unverified",
                "target_label": "A",
                "predicted_label": "",
                "canonical_label": "",
            },
        ]

        report = build_report(attempts)

        self.assertEqual(report["confidence_analysis"]["high (>=75%)"]["total"], 1)
        self.assertEqual(report["confidence_analysis"]["medium (50-<75%)"]["total"], 1)
        self.assertEqual(report["confidence_analysis"]["low (<50%)"]["total"], 0)
        self.assertEqual(report["coverage"]["eligible_attempts"], 4)
        self.assertEqual(report["coverage"]["answered"], 3)
        self.assertEqual(report["coverage"]["rejected"], 1)
        self.assertEqual(report["coverage"]["coverage"], 0.75)
        self.assertEqual(report["coverage"]["answered_verified_total"], 2)

    def test_persist_verify_and_report_pipeline_keeps_target_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RecognitionAttemptStore(Path(temp_dir) / "attempts.sqlite3")
            session_id = store.start_session(
                "session",
                model_version="model-sha",
                vocabulary_version="vocab-sha",
                configuration={"threshold": 0.45},
            )
            attempt_id = store.begin_attempt(
                session_id=session_id,
                target_label="AXE1",
                model_version="model-sha",
                preprocessing_version="pre-v1",
            )
            store.finish_attempt(
                attempt_id,
                status="completed",
                predicted_label="DOG (variant 1)",
                original_model_label="DOG1",
                canonical_label_value="DOG (variant 1)",
                class_index=4,
                confidence=0.72,
                top_k_predictions=[
                    {
                        "canonical_label": "DOG (variant 1)",
                        "original_model_label": "DOG1",
                        "class_index": 4,
                        "confidence": 0.72,
                    },
                    {
                        "canonical_label": "AXE (variant 1)",
                        "original_model_label": "AXE1",
                        "class_index": 3,
                        "confidence": 0.2,
                    },
                    {
                        "canonical_label": "BEE (variant 1)",
                        "original_model_label": "BEE1",
                        "class_index": 2,
                        "confidence": 0.08,
                    },
                ],
            )
            store.verify_ground_truth(attempt_id, "DOG1", "independent-reviewer")

            report = build_report(
                store.attempts(session_id),
                session=store.get_session(session_id),
            )

        self.assertEqual(report["target_agreement"]["accuracy"], 0.0)
        self.assertEqual(report["verified_top1"]["accuracy"], 1.0)
        self.assertEqual(report["verified_top1"]["total"], 1)
        self.assertEqual(report["counts"]["verified"], 1)
        self.assertEqual(report["counts"]["unverified"], 0)
        self.assertEqual(report["session"]["model_version"], "model-sha")


if __name__ == "__main__":
    unittest.main()
