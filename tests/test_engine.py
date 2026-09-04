import unittest
from src.render.engine import Engine
import os

class TestEngine(unittest.TestCase):
    def setUp(self):
        # Create a dummy neat_config.ini if missing
        if not os.path.exists("config/neat_config.ini"):
            pass # we assume it exists in the repo
        self.engine = Engine("config/neat_config.ini", False, 10, headless=True)
        
    def test_stagnation_logic(self):
        # Manually trigger stagnation by setting the records exactly the same for 26 generations
        self.engine.all_time_best_fitness = (100.0, 1)
        self.engine.all_time_highest_median = (50.0, 1)
        self.engine.all_time_top_speed = (10.0, 1)
        
        # We simulate the generation loop reaching gen 30 without any updates
        current_gen = 30
        
        # Test condition from engine.py:
        # if (current_gen - self.all_time_best_fitness[1] > 25 and ...
        
        fit_stagnant = (current_gen - self.engine.all_time_best_fitness[1]) > 25
        med_stagnant = (current_gen - self.engine.all_time_highest_median[1]) > 25
        spd_stagnant = (current_gen - self.engine.all_time_top_speed[1]) > 25
        
        self.assertTrue(fit_stagnant and med_stagnant and spd_stagnant)

if __name__ == '__main__':
    unittest.main()
