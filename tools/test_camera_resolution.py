"""Camera resolution negotiation and stream geometry regressions."""
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from vision.camera import Camera
from web import server
from web.server import _resize_for_stream

class ResolutionTests(unittest.TestCase):
    def test_stream_preserves_landscape_portrait_and_small_frames(self):
        for height, width, expected in [(1080,1920,(540,960)), (960,1280,(540,720)),
                                         (1920,1080,(540,304)), (480,640,(480,640))]:
            with self.subTest(size=(width,height)):
                frame=np.zeros((height,width,3),np.uint8)
                output,sx,sy=_resize_for_stream(frame,960,540)
                self.assertEqual(output.shape[:2],expected)
                self.assertAlmostEqual(sx,sy,delta=1/min(width,height))
                self.assertEqual(output.shape[1],round(width*sx))
                output[0,0]=255
                self.assertFalse(frame[0,0].any())

    def test_auto_mode_does_not_request_a_resolution(self):
        cap=Mock(); cap.isOpened.return_value=True
        cap.get.side_effect=lambda key: {cv2.CAP_PROP_FRAME_WIDTH:1280,cv2.CAP_PROP_FRAME_HEIGHT:960}.get(key,0)
        with patch('vision.camera.cv2.VideoCapture',return_value=cap):
            camera=Camera(); camera.open()
        calls=[call.args[0] for call in cap.set.call_args_list]
        self.assertNotIn(cv2.CAP_PROP_FRAME_WIDTH,calls)
        self.assertNotIn(cv2.CAP_PROP_FRAME_HEIGHT,calls)
        self.assertEqual((camera.width,camera.height),(1280,960))
        camera.stop()

    def test_requested_resolution_is_not_replaced_by_previous_device_actual_size(self):
        caps=[]
        for width,height in [(640,480),(1280,720)]:
            cap=Mock(); cap.isOpened.return_value=True
            cap.get.side_effect=lambda key,w=width,h=height: {cv2.CAP_PROP_FRAME_WIDTH:w,cv2.CAP_PROP_FRAME_HEIGHT:h}.get(key,0)
            caps.append(cap)
        with patch('vision.camera.cv2.VideoCapture',side_effect=caps):
            camera=Camera(width=1920,height=1080)
            camera.open(); camera.stop(); camera.index=1; camera.open()
        for cap in caps:
            cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH,1920)
            cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        self.assertEqual((camera.width,camera.height),(1280,720))
        camera.stop()

    def test_frame_size_reports_actual_frame_not_driver_setting(self):
        camera=Camera(width=1920,height=1080)
        self.assertEqual(camera.frame_size,(0,0))
        camera._frame=np.zeros((480,640,3),np.uint8)
        self.assertEqual(camera.frame_size,(640,480))
        camera.stop()
        self.assertEqual(camera.frame_size,(0,0))

    def test_live_stream_scales_final_box_with_four_by_three_frame(self):
        frame = np.zeros((600, 800, 3), np.uint8)
        overlay = SimpleNamespace(highlight=[(np.array([100,100,300,300]), 'item', 0.8)], highlight_size=(800,600))
        assistant = SimpleNamespace(_running=True, camera_enabled=True,
            camera=SimpleNamespace(latest=lambda: (frame, 1)), stats=SimpleNamespace(ui_fps=0),
            _overlay=overlay, _flash_until=time.time()+10, _last_subtitle='', _subtitle_until=0, depth_est=None)
        encoded = []
        def capture(image, quality):
            encoded.append(image.copy())
            assistant._running = False
            return b'jpeg'
        with patch.object(server, '_assistant', assistant), patch.object(server, '_encode_jpeg', side_effect=capture):
            server._frame_loop()
        self.assertEqual(encoded[0].shape[:2], (540,720))
        np.testing.assert_array_equal(encoded[0][90,90], [80,220,80])
        self.assertFalse(frame.any())

if __name__=='__main__': unittest.main()
