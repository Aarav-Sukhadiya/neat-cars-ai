import unittest
import pygame
from src.render.camera import Camera

class TestCamera(unittest.TestCase):
    def setUp(self):
        # Screen: 800x600, Track: 2000x2000
        self.camera = Camera(800, 600, 2000, 2000)

    def test_camera_initialization(self):
        self.assertEqual(self.camera.width, 800)
        self.assertEqual(self.camera.height, 600)
        self.assertEqual(self.camera.offset_x, 0)
        self.assertEqual(self.camera.offset_y, 0)

    def test_camera_clamping_top_left(self):
        # Target near top left, should clamp to 0,0
        self.camera.update((100, 100))
        self.assertEqual(self.camera.offset_x, 0)
        self.assertEqual(self.camera.offset_y, 0)

    def test_camera_centering(self):
        # Target in middle, offset should center the target on screen
        self.camera.update((1000, 1000))
        # Screen center is 400, 300. Offset should be 1000-400, 1000-300
        self.assertEqual(self.camera.offset_x, 600)
        self.assertEqual(self.camera.offset_y, 700)

    def test_camera_clamping_bottom_right(self):
        # Target near bottom right, should clamp to max offsets
        self.camera.update((1900, 1900))
        # Max offset x = 2000 - 800 = 1200
        # Max offset y = 2000 - 600 = 1400
        self.assertEqual(self.camera.offset_x, 1200)
        self.assertEqual(self.camera.offset_y, 1400)

    def test_apply_translation(self):
        self.camera.update((1000, 1000)) # Offset (600, 700)
        # World pos (1000, 1000) should translate to screen center (400, 300)
        screen_pos = self.camera.apply((1000, 1000))
        self.assertEqual(screen_pos, (400, 300))
        
        # World pos (600, 700) should translate to (0, 0)
        screen_pos2 = self.camera.apply((600, 700))
        self.assertEqual(screen_pos2, (0, 0))

if __name__ == '__main__':
    unittest.main()
