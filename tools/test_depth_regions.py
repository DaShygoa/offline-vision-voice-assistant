"""Depth localization regressions without ONNX weights or a camera.

Run: python tools/test_depth_regions.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.depth import DepthEstimator, DepthRegion, _iou


class SurfaceRegionTests(unittest.TestCase):
    def setUp(self):
        self.est = DepthEstimator({"min_area_frac": 0.005, "max_area_frac": 0.45})
        self.frame = np.full((240, 320, 3), 190, np.uint8)
        self.depth = np.full((240, 320), 0.9, np.float32)
        # A flat packet in front of the body, with a connected skin-coloured
        # hand. The wall at the left edge is even nearer than the packet.
        self.target = (105, 65, 195, 150)
        self.depth[150:, 70:270] = 0.6
        self.frame[150:, 70:270] = (70, 50, 35)
        self.depth[:, :25] = 0.01
        self.frame[65:150, 105:195] = (235, 238, 240)
        self.depth[65:150, 105:195] = 0.1
        self.depth[85:165, 195:245] = 0.13
        self.frame[85:165, 195:245] = (100, 145, 190)

    def test_filled_packet_survives_nearer_border_wall(self):
        regions = self.est.find_top_regions(self.frame, self.depth, k=4)
        self.assertTrue(regions)
        self.assertTrue(any(_iou(region.box, self.target) > 0.75 for region in regions))
        for region in regions:
            self.assertLess(region.box[3] - region.box[1], 0.95 * 240)

    def test_whole_surface_survives_object_plus_hand_proposal(self):
        regions = self.est._surface_regions(self.frame, self.depth)
        target = max(regions, key=lambda region: _iou(region.box, self.target))
        self.assertGreater(_iou(target.box, self.target), 0.75)
        self.assertEqual(target.mask[100, 145], 1)
        self.assertEqual(target.mask[100, 220], 0)

    def test_no_object_in_uniform_depth(self):
        regions = self.est.find_top_regions(self.frame, np.full_like(self.depth, 0.5))
        self.assertEqual(regions, [])

    def test_skin_separated_packet_keeps_pink_printing_inside_hull(self):
        frame = np.full((120, 120, 3), 220, np.uint8)
        frame[45:75, 45:75] = (170, 180, 220)  # skin-like pink packaging
        mask = np.zeros((120, 120), np.uint8)
        mask[20:100, 20:100] = 1
        mask[45:75, 45:75] = 0
        region = DepthRegion((20, 20, 100, 100), 0.9, 0.1, 0.4,
                             mask=mask, skin_separated=True)
        crop, box = self.est.crop_region(frame, region)
        np.testing.assert_array_equal(crop[60 - box[1], 60 - box[0]], frame[60, 60])

    def test_regular_packet_keeps_pink_printing_inside_hull(self):
        frame = np.full((120, 120, 3), 220, np.uint8)
        frame[45:75, 45:75] = (170, 180, 220)
        mask = np.zeros((120, 120), np.uint8)
        mask[20:100, 20:100] = 1
        region = DepthRegion((20, 20, 100, 100), 0.9, 0.1, 0.4, mask=mask)
        crop, box = self.est.crop_region(frame, region)
        np.testing.assert_array_equal(crop[60 - box[1], 60 - box[0]], frame[60, 60])

    def test_sparse_person_sized_edge_chain_is_rejected(self):
        region = DepthRegion((30, 50, 310, 230), 0.9, 0.1, 0.03)
        self.assertFalse(self.est._plausible_region(region, 320, 240))

    def test_thin_packet_edge_cannot_be_a_whole_object(self):
        region = DepthRegion((100, 65, 195, 77), 0.9, 0.1, 0.01)
        self.assertFalse(self.est._plausible_region(region, 320, 240))

    def test_masks_and_boxes_remain_in_original_frame_coordinates(self):
        frame = np.repeat(np.repeat(self.frame, 4, axis=0), 4, axis=1)
        depth = np.repeat(np.repeat(self.depth, 4, axis=0), 4, axis=1)
        target_box = tuple(value * 4 for value in self.target)
        regions = self.est.find_top_regions(frame, depth, k=4)
        best = max(regions, key=lambda region: _iou(region.box, target_box))
        self.assertGreater(_iou(best.box, target_box), 0.75)
        self.assertEqual(best.mask.shape, depth.shape)
        self.assertEqual(best.mask[400, 580], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
