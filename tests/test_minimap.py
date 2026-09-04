import unittest
from src.render.minimap import Minimap

class TestMinimap(unittest.TestCase):
    def setUp(self):
        # Minimap size 200x200, Track size 2000x2000
        self.minimap = Minimap(200, 200, 2000, 2000)

    def test_minimap_initialization(self):
        self.assertEqual(self.minimap.scale_x, 200 / 2000.0)
        self.assertEqual(self.minimap.scale_y, 200 / 2000.0)

    def test_world_to_minimap_translation(self):
        # Top left
        self.assertEqual(self.minimap.world_to_minimap((0, 0)), (0, 0))
        
        # Center
        self.assertEqual(self.minimap.world_to_minimap((1000, 1000)), (100, 100))
        
        # Bottom right
        self.assertEqual(self.minimap.world_to_minimap((2000, 2000)), (200, 200))

if __name__ == '__main__':
    unittest.main()
