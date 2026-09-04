import unittest
import pygame
import math
import os
from src.render.car import Car
from src.render.track import Track

class MockTrack:
    def __init__(self, width=1000, height=1000):
        self.width = width
        self.height = height

class TestCarPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        # Create a dummy surface for track
        cls.track_surf = pygame.Surface((1000, 1000))
        cls.track_surf.fill((0, 0, 0)) # Fill black
        cls.mock_track = MockTrack()

    def test_initialization(self):
        car = Car([100, 100], self.mock_track, checkpoints=[])
        self.assertEqual(car.position, [100, 100])
        self.assertEqual(car.speed, Car.DEFAULT_SPEED)
        self.assertEqual(car.angle, Car.DEFAULT_ANGLE)
        self.assertTrue(car.alive)
        self.assertEqual(car.checkpoint_fitness, 0.0)

    def test_acceleration_and_braking(self):
        car = Car([100, 100], self.mock_track)
        initial_speed = car.speed
        
        car.accelerate()
        self.assertEqual(car.speed, initial_speed + Car.SPEED_INCREMENT)
        
        car.brake()
        self.assertEqual(car.speed, initial_speed)
        
        # Test braking limit
        while car.speed > Car.MINIMUM_SPEED:
            car.brake()
        car.brake() # One more time should trigger the penalty branch
        self.assertEqual(car.speed, Car.MINIMUM_SPEED)
        self.assertGreater(car.speed_penalty, 0)

    def test_turning(self):
        car = Car([100, 100], self.mock_track)
        initial_angle = car.angle
        
        car.turn_left()
        self.assertEqual(car.angle, initial_angle + Car.ANGLE_INCREMENT)
        
        car.turn_right()
        self.assertEqual(car.angle, initial_angle)

    def test_corner_updates(self):
        car = Car([100, 100], self.mock_track)
        car.update_center()
        car.refresh_corners_positions()
        
        self.assertEqual(len(car.corners), 4)
        # Check if they form a box approximately CAR_SIZE_X by CAR_SIZE_Y
        width = math.hypot(car.corners[0][0] - car.corners[1][0], car.corners[0][1] - car.corners[1][1])
        self.assertGreater(width, 10)

    def test_checkpoint_intersection_logic(self):
        # A checkpoint line from (150, 50) to (150, 150)
        checkpoints = [[150, 50, 150, 150]]
        car = Car([100, 100], self.mock_track, checkpoints=checkpoints)
        car.speed = 40 # Moves +20 in X axis
        car.angle = 0
        
        # Pre-intersection
        self.assertEqual(car.current_checkpoint_index, 0)
        
        # Step physics (moves from 140 to 160, crossing the X=150 checkpoint)
        car.update_physics(self.track_surf)
        
        # Should have crossed checkpoint 0
        self.assertEqual(car.current_checkpoint_index, 1)
        self.assertGreater(car.checkpoint_fitness, 0.0)
        
    def test_get_reward_tracks_peak_fitness(self):
        checkpoints = [[200, 100, 200, 200], [400, 100, 400, 200]]
        car = Car([100, 150], self.mock_track, checkpoints=checkpoints)
        car.update_center()
        
        # Initial reward is negative distance to first checkpoint
        rew1 = car.get_reward()
        self.assertLess(rew1, 0)
        
        # Move closer artificially
        car.position = [150, 150]
        car.update_center()
        rew2 = car.get_reward()
        self.assertGreater(rew2, rew1)
        
        # Jump past checkpoint to spike dist_to_next
        car.checkpoint_fitness = 1000.0
        car.current_checkpoint_index = 1
        car.position = [210, 150] # Very far from checkpoint 2 (X=400)
        car.update_center()
        
        rew3 = car.get_reward()
        self.assertEqual(rew3, car.max_fitness_achieved)
        
if __name__ == '__main__':
    unittest.main()
