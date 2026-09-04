import unittest
import pygame
from src.render.track import Track

class TestTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    def test_track_generation(self):
        # Even if we don't load the real image, Track class should initialize safely
        track = Track(800, 600)
        
        self.assertEqual(track.width, 800)
        self.assertEqual(track.height, 600)
        
        # Test drawing borders on a surface
        surf = pygame.Surface((800, 600))
        track.draw((100, 100), (255, 255, 255))
        
        # We can't easily test visual pixel colors without heavy mocks, 
        # but we can ensure the methods run without crashing.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
