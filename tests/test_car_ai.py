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
        if hasattr(car_ai, 'vector_physics'): car_ai.vector_physics.alive[0] = False
        
        # Since CarAI remaining_cars is tracked manually or computed in `compute`,
        # let's call compute on a dummy surface (black)
        import pygame
        surf = pygame.Surface((1000, 1000))
        surf.fill((0, 0, 0))
        
        car_ai.compute(surf)
        
        # We expect remaining cars to decrease or cars list to shrink depending on implementation
        # Actually CarAI doesn't remove cars from list, but counts remaining
        self.assertEqual(car_ai.remaining_cars, len(self.genomes) - 1)

    def test_best_car_and_genome_fitness_track_physics(self):
        import pygame
        import numpy as np
        surf = pygame.Surface((1000, 1000))
        surf.fill((0, 0, 0))
        car_ai = CarAI(self.genomes, self.config, [400, 400], self.track)
        for _ in range(5):
            car_ai.compute(surf)
        vp = car_ai.vector_physics
        alive = np.flatnonzero(vp.alive)
        for i in alive:
            self.assertEqual(self.genomes[i][1].fitness, float(vp.max_fitness[i]))
        self.assertEqual(car_ai.best_fitness, float(vp.max_fitness[car_ai.best_index]))
        self.assertIs(car_ai.best_nn, car_ai.nns[car_ai.best_index])
        sensors = car_ai.get_sensors(int(alive[0]))
        self.assertEqual(len(sensors), 7)
        self.assertEqual(len(sensors[0][0]), 2)


if __name__ == '__main__':
    unittest.main()
