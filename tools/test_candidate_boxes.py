"""Candidate-box visibility regressions without camera or model weights."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.app import AssistantApp
from vision.visualizer import OverlayState
from web import server

class CandidateBoxTests(unittest.TestCase):
    def setUp(self):
        self.app = object.__new__(AssistantApp)
        self.app._overlay = OverlayState()
        self.app._recognition_enabled = True
        import threading
        self.app._recognition_lock = threading.RLock()
        self.frame = np.zeros((120, 200, 3), np.uint8)
        self.app.camera = SimpleNamespace(latest=lambda: (self.frame, 1))
        self.app.resnet = SimpleNamespace(ready=True)
        self.app.stats = SimpleNamespace(last_intent='')
        self.app.debug_cap = Mock()
        self.app.depth_gate = 0.3
        self.app._update_depth_vis = Mock()
        self.app.cropper = SimpleNamespace(quality_ok=lambda crop: False)

    def test_filtered_depth_candidates_are_published_even_without_match(self):
        regions = [SimpleNamespace(box=(10, 10, 60, 70), mean_depth=0.1, score=1, area_frac=0.2, mask=None),
                   SimpleNamespace(box=(90, 20, 150, 100), mean_depth=0.9, score=1, area_frac=0.2, mask=None)]
        self.app.depth_est = SimpleNamespace(ready=True, estimate_depth=lambda frame: frame,
            find_top_regions=lambda frame, depth, k: regions,
            crop_region=lambda frame, region: (frame[10:70,10:60], region.box))
        reply = self.app._do_identify()
        self.assertIn('没看到', reply)
        expires, width, height, boxes = self.app._overlay.candidate_snapshot
        self.assertGreater(expires, time.time())
        self.assertEqual((width, height), (200, 120))
        self.assertEqual(boxes, (((10, 10, 60, 70), None, None, 'quality_fail'),
                                 ((90, 20, 150, 100), None, None, 'too_far')))

    def test_stale_boxes_cleared_when_camera_unavailable(self):
        self.app._show_candidate_snapshot(self.frame, [{'box': (1, 2, 3, 4)}])
        self.app.camera.latest = lambda: (None, None)
        self.assertIn('没有画面', self.app._do_identify())
        self.assertEqual(self.app._overlay.candidate_snapshot[3], ())

    def test_center_fallback_published_when_crop_is_rejected(self):
        self.app.depth_est = None
        self.app.cropper.center_roi = lambda frame: (20, 20, 100, 100)
        self.app.cropper.crop_box = lambda frame, box: frame[20:100,20:100]
        self.assertIn('看不清', self.app._do_identify())
        self.assertEqual(self.app._overlay.candidate_snapshot[3], (((20, 20, 100, 100), None, None, 'quality_fail'),))

    def test_below_threshold_candidate_keeps_name_and_score_but_result_unknown(self):
        match = SimpleNamespace(user_name='手机', score=0.42)
        region = SimpleNamespace(box=(10, 10, 60, 70), mean_depth=0.1, score=1, area_frac=0.2, mask=None)
        self.app.depth_est = SimpleNamespace(ready=True, estimate_depth=lambda frame: frame,
            find_top_regions=lambda frame, depth, k: [region],
            crop_region=lambda frame, region: (frame[10:70, 10:60], region.box))
        self.app.cropper.quality_ok = lambda crop: True
        self.app._depth_candidate_is_face = lambda *args: False
        self.app._match_crop = lambda *args: (np.ones(4), match)
        self.app.memory = SimpleNamespace(top_k=lambda *args, **kwargs: [match], threshold=0.7, margin_threshold=0.1)
        self.app._flash_match = Mock()
        reply = self.app._do_identify()
        self.assertIn('未知物体', reply)
        self.assertEqual(self.app._overlay.candidate_snapshot[3], (((10, 10, 60, 70), '手机', 0.42, 'matched'),))

    def test_center_fallback_publishes_match(self):
        match = SimpleNamespace(user_name='水杯', score=0.42)
        self.app.depth_est = None
        self.app.cropper.center_roi = lambda frame: (20, 20, 100, 100)
        self.app.cropper.crop_box = lambda frame, box: frame[20:100, 20:100]
        self.app.cropper.quality_ok = lambda crop: True
        self.app._match_crop = lambda *args: (np.ones(4), match)
        self.app.memory = SimpleNamespace(top_k=lambda *args, **kwargs: [match], threshold=0.7, margin_threshold=0.1)
        self.app._flash_match = Mock()
        self.assertIn('未知物体', self.app._do_identify())
        self.assertEqual(self.app._overlay.candidate_snapshot[3], (((20, 20, 100, 100), '水杯', 0.42, 'matched'),))

    def test_status_only_exposes_unexpired_candidates_while_camera_is_active(self):
        self.app._show_candidate_snapshot(self.frame, [{'box': (10, 20, 30, 40), 'name': '手机', 'score': 0.42, 'reason': 'matched'}])
        stats = SimpleNamespace(ui_fps=0, mode='', last_intent='', last_reply='',
            last_similarity=0, mem_objects=0, mem_info='', det_info='', asr_latency_ms=0)
        fake = SimpleNamespace(stats=stats, _overlay=self.app._overlay, current_devices={},
            _last_subtitle='', _subtitle_until=0, resnet=None, depth_est=None,
            use_voice=False, recognition_enabled=True, camera_enabled=True, mic_enabled=False)
        with patch.object(server, '_assistant', fake):
            client = server.app.test_client()
            result = client.get('/status').get_json()['candidate_boxes']
            self.assertEqual(result, {'width': 200, 'height': 120, 'boxes': [
                {'box': [10, 20, 30, 40], 'name': '手机', 'score': 0.42, 'reason': 'matched'}]})
            fake.camera_enabled = False
            self.assertIsNone(client.get('/status').get_json()['candidate_boxes'])
            fake.camera_enabled = True
            self.app._overlay.candidate_snapshot = (time.time() - 1, 200, 120, ((10, 20, 30, 40),))
            self.assertIsNone(client.get('/status').get_json()['candidate_boxes'])

if __name__ == '__main__':
    unittest.main()
