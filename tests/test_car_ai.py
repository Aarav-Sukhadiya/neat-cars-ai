import unittest
import neat
import os
from src.ai.car_ai import CarAI
from src.render.track import Track

class TestCarAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need a neat configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'neat_config.ini')
        cls.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        # Create a population to get some genomes
        p = neat.Population(cls.config)
        cls.genomes = list(p.population.items())
        cls.track = Track(1000, 1000)

    def test_car_ai_initialization(self):
        car_ai = CarAI(self.genomes, self.config, [100, 100], self.track)
        self.assertEqual(len(car_ai.cars), len(self.genomes))
        self.assertEqual(len(car_ai.genomes), len(self.genomes))
        self.assertEqual(car_ai.remaining_cars, len(self.genomes))

    def test_survival_tracking(self):
        car_ai = CarAI(self.genomes, self.config, [100, 100], self.track)
        
        # Kill one car manually
        car_ai.cars[0].alive = False
        
        # Since CarAI remaining_cars is tracked manually or computed in `compute`,
        # let's call compute on a dummy surface (black)
        import pygame
        surf = pygame.Surface((1000, 1000))
        surf.fill((0, 0, 0))
        
        car_ai.compute(surf)
        
        # We expect remaining cars to decrease or cars list to shrink depending on implementation
        # Actually CarAI doesn't remove cars from list, but counts remaining
        self.assertEqual(car_ai.remaining_cars, len(self.genomes) - 1)

if __name__ == '__main__':
    unittest.main()
