import unittest
import numpy as np
from src.ai.vector_physics import VectorizedPhysics
from src.render.car import Car
from src.render.track import Track

class TestVectorPhysics(unittest.TestCase):
    def setUp(self):
        # Create dummy track and cars
        self.track = Track(1000, 1000)
        self.cars = [Car([500, 500], self.track), Car([500, 500], self.track)]
        self.vp = VectorizedPhysics(self.cars, 1000, 1000, [])
        
    def test_initialization(self):
        """Test if tensors correctly extract initial state from car objects"""
        self.assertEqual(len(self.vp.x), 2)
        self.assertEqual(self.vp.x[0], 500)
        self.assertEqual(self.vp.y[0], 500)
        self.assertTrue(self.vp.alive[0])
        self.assertEqual(self.vp.speed[0], 10.0)

    def test_kinematics(self):
        """Test if steering and accelerating update the tensors correctly"""
        # choices: 0=Left, 1=Right, 2=Accel, 3=Brake
        # Car 0: Accelerate, Car 1: Turn Left
        choices = np.array([2, 0], dtype=np.int32)
        
        # Track pixels dummy
        track_pixels = np.zeros((1000, 1000), dtype=np.uint8)
        
        initial_angle = self.vp.angle[1]
        
        self.vp.step(choices, track_pixels)
        
        # Car 0 should have increased speed
        self.assertAlmostEqual(self.vp.speed[0], 10.0 + Car.SPEED_INCREMENT)
        # Car 1 should have turned left (angle + increment)
        self.assertAlmostEqual(self.vp.angle[1], initial_angle + Car.ANGLE_INCREMENT)
        
    def test_sync_to_cars(self):
        """Test if the vector tensor state correctly pushes back to Python objects"""
        self.vp.x[0] = 777.0
        self.vp.speed[0] = 55.0
        self.vp.alive[0] = False
        
        self.vp.sync_to_cars(np.array([0, 1]))
        
        self.assertEqual(self.cars[0].position[0], 777.0)
        self.assertEqual(self.cars[0].speed, 55.0)
        self.assertFalse(self.cars[0].alive)

if __name__ == '__main__':
    unittest.main()
