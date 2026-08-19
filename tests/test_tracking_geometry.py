import unittest

import numpy as np

from unigaze_personalization.tracking_geometry import (
    ScreenGeometry,
    compensate_origin,
    denormalize_gaze,
    intersect_screen,
)


class TrackingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.screen = ScreenGeometry(width_mm=500, height_mm=300, camera_to_top_mm=0, plane_z_mm=0)

    def test_ray_intersects_screen(self):
        hit = intersect_screen([10, 20, 600], [0, 0, -1], self.screen)
        np.testing.assert_allclose(hit, [10, 20, 0])

    def test_lateral_origin_motion_changes_screen_hit(self):
        corrected = compensate_origin([0, 0], [0, 150, 600], [50, 150, 600], self.screen)
        self.assertAlmostEqual(corrected[0], 0.2, places=6)
        self.assertAlmostEqual(corrected[1], 0.0, places=6)

    def test_depth_motion_preserves_visual_ray(self):
        corrected = compensate_origin([0.4, 0], [0, 150, 600], [0, 150, 500], self.screen)
        self.assertAlmostEqual(corrected[0], 0.4 * 500 / 600, places=6)

    def test_identity_denormalization_points_toward_screen(self):
        direction = denormalize_gaze([0, 0], np.eye(3))
        np.testing.assert_allclose(direction, [0, 0, -1])


if __name__ == "__main__":
    unittest.main()
