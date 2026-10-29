"""Camera enrollment regressions; no camera, microphone, or model weights needed.

Run: python tools/test_enrollment.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app import AssistantApp
from vision.depth import DepthRegion


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Camera:
    def __init__(self, frame):
        self.frame = frame
        self.frame_id = 0
        self.frozen = False

    def latest(self):
        if not self.frozen:
            self.frame_id += 1
        return self.frame, self.frame_id

    def stop(self):
        pass


class SlowDepth:
    """Model the CPU timing that left only 2/3 samples in a 2.5s window."""

    ready = True

    def __init__(self, clock):
        self.clock = clock

    def estimate_depth(self, frame):
        self.clock.sleep(1.5)
        return np.zeros(frame.shape[:2], np.float32)

    def find_top_regions(self, frame, depth, k):
        self.clock.sleep(0.25)
        return [DepthRegion((40, 40, 200, 200), 1.0, 0.0, 0.33)]

    def crop_region(self, frame, region):
        self.clock.sleep(0.3)
        x1, y1, x2, y2 = region.box
        return frame[y1:y2, x1:x2].copy(), region.box


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="enrollment_test_")
        self.addCleanup(self.tmp.cleanup)
        cfg = {
            "memory": {
                "db_path": os.path.join(self.tmp.name, "memory.db"),
                "samples_dir": os.path.join(self.tmp.name, "samples"),
                "embeddings_path": os.path.join(self.tmp.name, "embeddings.npz"),
                "min_samples": 3,
                "max_samples": 10,
                "sample_duration_sec": 2.5,
                "sample_timeout_sec": 15.0,
            },
            "depth": {"enabled": False},
            "yolo": {"enabled": False},
            "tts": {"enabled": False},
            "debug": {"enabled": False},
        }
        self.app = AssistantApp(cfg, headless=True, use_voice=False)
        self.addCleanup(self.app.memory._conn.close)
        self.app._running = True
        self.app.resnet = SimpleNamespace(ready=True)
        self.frame = np.random.default_rng(1).integers(40, 210, (240, 320, 3), dtype=np.uint8)
        self.app.camera = Camera(self.frame)
        self.app._is_face_candidate = lambda crop: False
        self.clock = Clock()
        self.app._embed_crop = self.embed
        self.time_patch = patch("core.app.time", self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def embed(self, crop):
        self.clock.sleep(0.05)
        # Same scene/embedding across distinct camera frames must still satisfy
        # the minimum, even if the user holds the object completely still.
        return np.ones(512, np.float32) / np.sqrt(512)

    def remember(self):
        reply = self.app._do_remember("手机")
        self.assertFalse(self.app._overlay.learning)
        self.assertIsNone(self.app._overlay.match_box)
        return reply

    def test_slow_cpu_collects_minimum_and_persists_images_and_features(self):
        self.app.depth_est = SlowDepth(self.clock)
        reply = self.remember()
        self.assertIn("已经记住手机", reply)
        self.assertGreater(self.clock.now - 100, 2.5)
        self.assertLess(self.clock.now - 100, 15)
        self.assertEqual(self.app.memory.stats()["samples"], 3)
        paths = self.app.memory.sample_images("手机")
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(Path(path).is_file())
            self.assertTrue(Path(path + ".npy").is_file())

    def test_old_configs_get_extended_timeout_by_default(self):
        del self.app.cfg["memory"]["sample_timeout_sec"]
        self.app.depth_est = SlowDepth(self.clock)
        self.assertIn("已经记住手机", self.remember())

    def test_fast_static_scene_keeps_normal_window_and_minimum(self):
        self.assertIn("已经记住手机", self.remember())
        self.assertEqual(self.app.memory.stats()["samples"], 3)
        self.assertLess(self.clock.now - 100, 2.6)

    def test_processing_timeout_is_not_reported_as_bad_lighting(self):
        self.app.depth_est = SlowDepth(self.clock)
        self.app.cfg["memory"]["sample_timeout_sec"] = 2.5
        reply = self.remember()
        self.assertIn("采样处理超时", reply)
        self.assertNotIn("光线", reply)
        self.assertEqual(self.app.memory.stats()["objects"], 0)
        # A synchronous inference already in flight may finish after the limit.
        self.assertLess(self.clock.now - 100, 4.5)

    def test_dark_frames_fail_with_quality_reason_and_bounded_wait(self):
        self.app.camera.frame = np.zeros_like(self.frame)
        reply = self.remember()
        self.assertIn("画面质量不足", reply)
        self.assertLess(self.clock.now - 100, 15.1)
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_frozen_camera_does_not_enroll_same_frame_three_times(self):
        self.app.camera.frozen = True
        reply = self.remember()
        self.assertIn("1/3", reply)
        self.assertIn("摄像头", reply)
        self.assertLess(self.clock.now - 100, 15.1)
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_disconnected_camera_times_out_without_writing(self):
        self.app.camera.frame = None
        self.assertIn("摄像头", self.remember())
        self.assertLess(self.clock.now - 100, 15.1)
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_embedding_failures_have_their_own_reason(self):
        self.app._embed_crop = lambda crop: None
        self.assertIn("特征提取失败", self.remember())
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_depth_localization_failure_does_not_enroll_background(self):
        self.app.depth_est = SlowDepth(self.clock)
        self.app.depth_est.find_top_regions = lambda *args, **kwargs: []
        self.assertIn("没有定位到完整物体", self.remember())
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_every_sample_uses_its_own_depth_frame(self):
        self.app.depth_est = SlowDepth(self.clock)
        seen = []
        estimate = self.app.depth_est.estimate_depth

        def record_frame(frame):
            seen.append(self.app.camera.frame_id)
            return estimate(frame)

        self.app.depth_est.estimate_depth = record_frame
        self.assertIn("已经记住手机", self.remember())
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3)

    def test_remember_clears_previous_identification_box(self):
        self.app._overlay.highlight = [(np.zeros(4), "old", 0.5)]
        self.remember()
        self.assertEqual(self.app._overlay.highlight, [])

    def test_pause_cancels_even_after_minimum_collected(self):
        count = 0

        def embed_and_pause(crop):
            nonlocal count
            count += 1
            if count == 3:
                self.app.set_recognition_enabled(False)
            return self.embed(crop)

        self.app._embed_crop = embed_and_pause
        self.assertIn("录入已取消", self.remember())
        self.assertEqual(self.app.memory.stats()["objects"], 0)

    def test_exceptions_clear_learning_overlay(self):
        def fail(crop):
            raise RuntimeError("test inference failure")

        self.app._embed_crop = fail
        with self.assertRaises(RuntimeError):
            self.app._do_remember("手机")
        self.assertFalse(self.app._overlay.learning)
        self.assertIsNone(self.app._overlay.match_box)

    def test_invalid_sample_limits_fail_before_sampling(self):
        self.app.cfg["memory"]["max_samples"] = 2
        self.assertIn("配置错误", self.remember())
        self.assertEqual(self.clock.now, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
