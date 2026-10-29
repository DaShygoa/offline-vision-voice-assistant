"""Foreground rejection and orientation regressions; no model or camera needed.

Run: python tools/test_identification.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app import AssistantApp
from vision.depth import DepthRegion
from vision.memory import VisualMemory


class ForegroundFaceTests(unittest.TestCase):
    def setUp(self):
        self.app = object.__new__(AssistantApp)
        self.frame = np.full((160, 220, 3), (100, 145, 190), np.uint8)
        self.mask = np.zeros((160, 220), np.uint8)
        cv2.ellipse(self.mask, (110, 80), (90, 55), 0, 0, 360, 1, -1)
        self.frame[self.mask > 0] = (65, 65, 65)
        self.region = DepthRegion((0, 0, 220, 160), 0.9, 0.1, 0.4, mask=self.mask)
        self.cleaned = np.full_like(self.frame, 65)

    def test_face_outside_object_does_not_reject_gray_object(self):
        self.assertTrue(self.app._is_face_candidate(self.frame))
        self.assertFalse(self.app._depth_candidate_is_face(
            self.frame, self.cleaned, self.region.box, self.region))

    def test_face_inside_foreground_is_still_rejected(self):
        self.frame[self.mask > 0] = (100, 145, 190)
        self.assertTrue(self.app._depth_candidate_is_face(
            self.frame, self.cleaned, self.region.box, self.region))

    def test_mask_holes_cannot_conceal_raw_face(self):
        self.frame[self.mask > 0] = (100, 145, 190)
        self.mask[45:115, 65:155] = 0
        self.assertTrue(self.app._depth_candidate_is_face(
            self.frame, self.cleaned, self.region.box, self.region))

    def test_missing_mask_keeps_raw_face_protection(self):
        self.region.mask = None
        self.assertTrue(self.app._depth_candidate_is_face(
            self.frame, self.cleaned, self.region.box, self.region))

    def test_empty_mask_keeps_raw_face_protection(self):
        self.region.mask[:] = 0
        self.assertTrue(self.app._depth_candidate_is_face(
            self.frame, self.cleaned, self.region.box, self.region))


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="orientation_test_")
        self.addCleanup(self.tmp.cleanup)
        self.app = object.__new__(AssistantApp)
        self.app.memory = VisualMemory({
            "db_path": os.path.join(self.tmp.name, "bank.db"),
            "samples_dir": os.path.join(self.tmp.name, "samples"),
            "embeddings_path": os.path.join(self.tmp.name, "embeds.npz"),
            "embed_dim": 4,
            "center_features": False,
            "match_threshold": 0.65,
            "margin_threshold": 0.1,
        })
        self.addCleanup(self.app.memory._conn.close)
        self.app.memory.add_object("mouse", None, [np.array([1, 0, 0, 0])], [])
        self.app.memory.add_object("phone", None, [np.array([0, 1, 0, 0])], [])
        self.upright = np.zeros((30, 50, 3), np.uint8)
        self.upright[0, 0] = 255
        self.calls = 0
        self.app._embed_crop = self.embed

    def embed(self, crop):
        self.calls += 1
        if not crop.any():
            return np.array([0, 0, 1, 0], np.float32)
        if crop[0, 0, 0] == 255:
            return np.array([1, 0, 0, 0], np.float32)
        # An unfamiliar orientation can score above the absolute threshold,
        # but cannot be called sure when another object's score is too close.
        return np.array([0.72, 0.69, 0, 0], np.float32)

    def test_recognizes_all_four_quarter_turns(self):
        for turns in range(4):
            with self.subTest(turns=turns):
                emb, match = self.app._match_crop(np.rot90(self.upright, turns))
                self.assertEqual(match.user_name, "mouse")
                self.assertGreaterEqual(match.score, self.app.memory.threshold)
                np.testing.assert_array_equal(emb, np.array([1, 0, 0, 0]))

    def test_confident_original_view_does_not_run_extra_inference(self):
        self.app._match_crop(self.upright)
        self.assertEqual(self.calls, 1)

    def test_ambiguous_original_view_does_not_bypass_margin(self):
        _, match = self.app._match_crop(np.rot90(self.upright))
        self.assertGreater(self.calls, 1)
        self.assertEqual(match.score, 1.0)

    def test_unknown_stays_below_threshold_after_all_turns(self):
        _, match = self.app._match_crop(np.zeros_like(self.upright))
        self.assertLess(match.score, self.app.memory.threshold)
        self.assertEqual(self.calls, 4)

    def test_extraction_failures_return_no_match(self):
        self.app._embed_crop = lambda crop: None
        self.assertEqual(self.app._match_crop(self.upright), (None, None))

    def test_diagonal_foreground_is_straightened_without_losing_ends(self):
        mask = np.zeros((180, 220), np.uint8)
        points = cv2.boxPoints(((110, 90), (150, 50), -30)).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 1)
        crop = np.zeros((180, 220, 3), np.uint8)
        crop[mask > 0] = (30, 160, 230)
        straight = self.app._deskew_crop(crop, mask)
        self.assertIsNotNone(straight)
        self.assertGreater(straight.shape[1], straight.shape[0] * 2)
        self.assertGreater(np.count_nonzero(straight[:, :, 1] > 100), 0.9 * np.count_nonzero(mask))

    def test_square_or_missing_mask_does_not_guess_an_angle(self):
        crop = np.zeros((100, 100, 3), np.uint8)
        self.assertIsNone(self.app._deskew_crop(crop, None))
        self.assertIsNone(self.app._deskew_crop(crop, np.ones((100, 100), np.uint8)))


class PausedRestartTests(unittest.TestCase):
    def test_resume_loads_cold_voice_models_before_starting_worker(self):
        app = object.__new__(AssistantApp)
        events = []
        app._resource_lock = threading.RLock()
        app._mic_enabled = False
        app._threads = []
        app.use_voice = False
        app.stats = SimpleNamespace(mic_enabled=False)
        app.asr = SimpleNamespace(ready=False, load=lambda: events.append("asr"))
        app.vad = SimpleNamespace(load=lambda: events.append("vad"))
        worker = Mock()
        worker.start.side_effect = lambda: events.append("worker")
        with patch("core.app.threading.Thread", return_value=worker):
            self.assertTrue(app.set_mic_enabled(True))
        self.assertEqual(events, ["asr", "vad", "worker"])
        self.assertTrue(app.use_voice)


class UnknownResponseTests(unittest.TestCase):
    def test_below_threshold_does_not_name_a_library_object(self):
        app = object.__new__(AssistantApp)
        app.memory = SimpleNamespace(threshold=0.5)
        app.stats = SimpleNamespace(last_similarity=0.0)
        reply = app._describe_unknown(SimpleNamespace(user_name="手机", score=0.42))
        self.assertNotIn("手机", reply)
        self.assertIn("42%", reply)
        self.assertIn("50%", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
