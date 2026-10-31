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

    def test_sync_copies_distance_and_frames(self):
        """Regression: driven_distance/frames_alive were never synced (avg speed stat was always 0)."""
        track_pixels = np.zeros((1000, 1000), dtype=np.uint8)
        for _ in range(3):
            self.vp.step(np.array([2, 2], dtype=np.int32), track_pixels)
        self.vp.sync_to_cars(np.array([0, 1]))
        self.assertEqual(self.cars[0].frames_alive, 3)
        self.assertAlmostEqual(self.cars[0].driven_distance, float(self.vp.driven_distance[0]))
        self.assertGreater(self.cars[0].driven_distance, 0)

    def test_leaving_the_map_is_a_crash(self):
        """Regression: off-map corners were clamped onto the edge pixel, so cars could escape."""
        track = Track(100, 100)
        cars = [Car([60, 35], track)]           # centre (75, 50), heading right, 10 px/frame
        vp = VectorizedPhysics(cars, 100, 100, [])
        open_track = np.zeros((100, 100), dtype=np.uint8)   # no walls at all
        for _ in range(3):
            vp.step(np.array([2], dtype=np.int32), open_track)
        self.assertFalse(vp.alive[0])

    def test_no_checkpoint_fallback_fitness(self):
        """Without checkpoints, fitness used to stay 0 forever; now it rewards distance."""
        track_pixels = np.zeros((1000, 1000), dtype=np.uint8)
        self.vp.step(np.array([2, 2], dtype=np.int32), track_pixels)
        self.vp.update_fitness()
        first = self.vp.max_fitness.copy()
        self.vp.step(np.array([2, 2], dtype=np.int32), track_pixels)
        self.vp.update_fitness()
        self.assertTrue(np.all(first > 0))
        self.assertTrue(np.all(self.vp.max_fitness > first))

    def test_fallback_matches_scalar_car_reward(self):
        self.vp.driven_distance[:] = [123.0, 456.0]
        self.vp.speed[:] = [12.0, 15.0]
        self.vp.speed_penalty[:] = [0, 2]
        vec = self.vp._fallback_reward()
        for i, car in enumerate(self.cars):
            car.driven_distance = float(self.vp.driven_distance[i])
            car.speed = float(self.vp.speed[i])
            car.speed_penalty = float(self.vp.speed_penalty[i])
            self.assertAlmostEqual(float(vec[i]), car.get_reward(), places=4)

    def test_checkpoint_crossing_and_starvation(self):
        track = Track(1000, 1000)
        cars = [Car([485, 485], track)]                      # centre (500, 500), heading right
        cps = [[520, 400, 520, 600], [900, 400, 900, 600]]
        vp = VectorizedPhysics(cars, 1000, 1000, cps)
        open_track = np.zeros((1000, 1000), dtype=np.uint8)
        vp.step(np.array([0], dtype=np.int32), open_track)   # turn left, still crosses x=520
        vp.step(np.array([2], dtype=np.int32), open_track)
        self.assertEqual(vp.cp_index[0], 1)
        self.assertGreaterEqual(vp.cp_fitness[0], 1000.0)
        # Spin in place long enough to starve.
        for _ in range(400):
            vp.step(np.array([0], dtype=np.int32), open_track)
        self.assertFalse(vp.alive[0])


if __name__ == '__main__':
    unittest.main()
