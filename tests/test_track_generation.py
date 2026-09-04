import unittest
import json
import math
import os

class TestTrackGeneration(unittest.TestCase):
    def check_track_gaps(self, track_file):
        """Helper to ensure checkpoints aren't missing massive chunks (regression test)"""
        path = f"data/tracks/{track_file}"
        if not os.path.exists(path):
            self.skipTest(f"Track {track_file} not found locally.")
            
        with open(path, "r") as f:
            cps = json.load(f)
            
        max_dist = 0
        for i in range(len(cps)):
            c1 = cps[i]
            c2 = cps[(i + 1) % len(cps)]
            
            mid1_x = (c1[0] + c1[2]) / 2
            mid1_y = (c1[1] + c1[3]) / 2
            mid2_x = (c2[0] + c2[2]) / 2
            mid2_y = (c2[1] + c2[3]) / 2
            
            dist = math.hypot(mid2_x - mid1_x, mid2_y - mid1_y)
            if dist > max_dist:
                max_dist = dist
                
        # Maximum allowed gap is 800 pixels. If it's 2000+, it means cars will mathematically starve.
        self.assertLess(max_dist, 800, f"{track_file} has a massive gap of {max_dist} pixels!")

    def test_super_hard_gaps(self):
        self.check_track_gaps("super_hard.json")
        
    def test_intersection_loop_gaps(self):
        self.check_track_gaps("intersection_loop.json")

if __name__ == '__main__':
    unittest.main()
