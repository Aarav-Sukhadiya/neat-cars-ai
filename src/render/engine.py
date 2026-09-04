import pygame
import neat
import time
import numpy as np
import pickle
import copy
from typing import Tuple, List
from src.ai.car_ai import CarAI
from src.render.car import Car
from src.render.colors import Color
from src.render.track import Track
from src.render.stats_panel import StatsPanel

class Engine:
    WIDTH = 1900
    HEIGHT = 1080
    TRACK_WIDTH = 1500
    TRACK_HEIGHT = 1080
    SIDEBAR_WIDTH = 400
    FPS = 60
    DEFAULT_FONT = "comicsansms"

    def __init__(self, neat_config_path: str, debug: bool, max_simulations: int, headless: bool = False):
        self.neat_config_path = neat_config_path
        self.debug = debug
        self.max_simulations = max_simulations
        self.HEADLESS = headless
        self.title = "Neat Cars"
        
        self.all_time_best_fitness = (0.0, 0)
        self.all_time_best_median = (0.0, 0)
        self.all_time_top_speed = (0.0, 0)
        self.prev_gen_stats = None
        
        # In headless mode we do not initialize a display window!
        if not self.HEADLESS:
            pygame.init()
            pygame.display.set_caption(self.title)
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            pygame.init()
            # os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = None

        self.clock = pygame.time.Clock()
        
        # Track setup
        self.track = Track(self.TRACK_WIDTH, self.TRACK_HEIGHT)
        try:
            # Drop .convert() so it works without a display context
            track_img = pygame.image.load("assets/track.png")
            self.track.surface.blit(track_img, (0, 0))
        except Exception as e:
            print(f"Warning: Could not load track.png: {e}")

        self.car = Car([0, 0], self.track)
        
        # Only initialize stats panel if not headless
        self.stats_panel = None if self.HEADLESS else StatsPanel(self.SIDEBAR_WIDTH, self.HEIGHT)
        
        self.state = "ai_running"
        self.decided_car_pos = [200 - Car.CAR_SIZE_X / 2, 950 - Car.CAR_SIZE_Y / 2]
        self.instructions = ["Training in progress..."]
        self.instruction_index = 0
        
        # Load Hall of Fame
        self.hall_of_fame = []
        try:
            with open("data/hall_of_fame.pkl", "rb") as f:
                self.hall_of_fame = pickle.load(f)
            print(f"Loaded {len(self.hall_of_fame)} elite cars from Hall of Fame!")
        except (FileNotFoundError, EOFError):
            print("No Hall of Fame found. Starting fresh.")

    def handle_events(self):
        # We still pump events so PyGame doesn't freeze the OS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            if not self.HEADLESS:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.track.adjust_brush_size(1)
                    elif event.button == 5:
                        self.track.adjust_brush_size(-1)
        return True
    
    def handle_drawing_track(self):
        pass

    def handle_placing_start_point(self):
        pass

    def draw(self):
        if self.HEADLESS:
            return
        # Original draw logic
        self.screen.fill((15, 15, 20))
        self.screen.blit(self.track.get_surface(), (0, 0))
        if self.state == "placing_start_point" or self.state == "ai_running":
            self.screen.blit(self.car.sprite, self.car.position)
        self.stats_panel.draw(
            self.screen, x_offset=self.TRACK_WIDTH, y_offset=0,
            generation=0, total_cars=0, alive=0,
            time_left=0.0, time_limit=CarAI.MAX_FRAMES / 60.0,
            best_fitness=0.0, median_fitness=0.0,
            max_speed=0.0, best_net_nodes=0, best_net_conns=0
        )
        pygame.display.set_caption(f"{self.title} - {self.instructions[self.instruction_index]}")
        pygame.display.update()

    def start_ai(self):
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.neat_config_path
        )

        self.population = neat.Population(config)
        
        # Initial node_indexer fix for disk-loaded HoF genomes to prevent mutation crashes
        if self.hall_of_fame:
            max_node_id = -1
            for fit, elite_genome in self.hall_of_fame:
                if elite_genome.nodes:
                    max_node_id = max(max_node_id, max(elite_genome.nodes.keys()))
            if max_node_id >= 0:
                from itertools import count
                config.genome_config.node_indexer = count(max_node_id + 1)

        if self.debug and not self.HEADLESS:
            self.population.add_reporter(neat.StdOutReporter(True))
            self.population.add_reporter(neat.StatisticsReporter())
            self.population.add_reporter(neat.Checkpointer(5, filename_prefix="neat-checkpoint-"))

        class StagnationTermination(Exception):
            pass
        self.StagnationTermination = StagnationTermination
        
        try:
            winner = self.population.run(self.run_simulation, self.max_simulations)
            
            with open("data/best_car_brain.pkl", "wb") as f:
                pickle.dump(winner, f)
            print(f"\nSaved best genome to 'best_car_brain.pkl'!")
        except StagnationTermination:
            print("\n[!] Training terminated: All records have stagnated for more than 25 generations.")
            if self.hall_of_fame:
                winner = self.hall_of_fame[0][1]
                with open("data/best_car_brain.pkl", "wb") as f:
                    pickle.dump(winner, f)
                print(f"Saved all-time best genome to 'best_car_brain.pkl'!")
            if self.HEADLESS:
                import sys
                sys.exit(0)

    def run_simulation(self, genomes: List[neat.DefaultGenome], config: neat.Config) -> None:
        # Inject Hall of Fame into the current generation
        if self.hall_of_fame:
            pop_keys = list(self.population.population.keys())
            for i, (fit, elite_genome) in enumerate(self.hall_of_fame):
                if i < len(pop_keys):
                    k = pop_keys[i]
                    cloned = copy.deepcopy(elite_genome)
                    cloned.key = k
                    self.population.population[k] = cloned
            # Re-speciate since we swapped genomes
            self.population.species.speciate(config, self.population.population, self.population.generation)
            genomes = list(self.population.population.items())

        car_ai = CarAI(genomes, config, self.decided_car_pos, self.track)
        total_cars = len(car_ai.cars)
        frame_count = 0
        gen_top_speed = 0.0
        gen_best_median = 0.0
        
        # Track the individual max speed reached by every single car
        car_max_speeds = {id(car): 0.0 for car in car_ai.cars}

        import os
        last_print_time = 0.0

        while True:
            if not self.handle_events():
                return

            car_ai.compute(self.track.get_surface())
            frame_count += 1

            frames_left = max(0, CarAI.MAX_FRAMES - frame_count)

            if car_ai.remaining_cars == 0 or frames_left <= 0:
                break

            # Metrics
            max_speed = 0.0
            max_fit = -1
            best_car = None

            for car, (_, genome) in zip(car_ai.cars, car_ai.genomes):
                if car.alive:
                    if car.speed > max_speed:
                        max_speed = car.speed
                    if car.speed > car_max_speeds[id(car)]:
                        car_max_speeds[id(car)] = car.speed
                    if genome.fitness is not None and genome.fitness > max_fit:
                        max_fit = genome.fitness
                        best_car = car

            fitnesses = [g.fitness for _, g in car_ai.genomes if g.fitness is not None]
            median_fitness = float(np.median(fitnesses)) if fitnesses else 0.0
                
            best_net_nodes = len(car_ai.best_nn.nodes) if car_ai.best_nn else 0
            best_net_conns = len(car_ai.best_nn.connections) if car_ai.best_nn else 0

            # Update generation trackers
            if max_speed > gen_top_speed:
                gen_top_speed = max_speed
            if median_fitness > gen_best_median:
                gen_best_median = median_fitness

            # HEADLESS TERMINAL RENDERER
            if self.HEADLESS:
                current_time = time.time()
                if current_time - last_print_time > 0.066:  # ~15 FPS cap for terminal
                    last_print_time = current_time
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("==================================================")
                    print(f" NEAT CARS (HEADLESS MODE) - GENERATION {car_ai.TOTAL_GENERATIONS}")
                    print("==================================================")
                    print(f" Alive Cars    : {car_ai.remaining_cars} / {total_cars}")
                    print(f" Time Left     : {frames_left/60.0:.1f}s / {CarAI.MAX_FRAMES/60.0:.1f}s (Virtual)")
                    print(f" Best Fitness  : {car_ai.best_fitness:,.1f}")
                    print(f" Median Fitness: {median_fitness:,.1f}")
                    print(f" Top Speed     : {max_speed:.1f} px/s")
                    print(f" Brain Size    : {best_net_nodes} Nodes / {best_net_conns} Conns")
                    print("==================================================")
                    if self.prev_gen_stats:
                        print(f" PREVIOUS GENERATION (Gen {self.prev_gen_stats['gen']})")
                        print("==================================================")
                        print(f" Best Fitness  : {self.prev_gen_stats['best_fitness']:,.1f}")
                        print(f" Peak Median   : {self.prev_gen_stats['median_fitness']:,.1f}")
                        print(f" Top Speed     : {self.prev_gen_stats['top_speed']:.1f} px/s")
                        print("==================================================")
                    print(f" ALL-TIME RECORDS (Across Generations)")
                    print("==================================================")
                    print(f" Highest Fitness      : {self.all_time_best_fitness[0]:,.1f} (Gen {self.all_time_best_fitness[1]})")
                    print(f" Highest Median Score : {self.all_time_best_median[0]:,.1f} (Gen {self.all_time_best_median[1]})")
                    print(f" Highest Top Speed    : {self.all_time_top_speed[0]:.1f} px/s (Gen {self.all_time_top_speed[1]})")
                    print("==================================================")
            else:
                self.screen.fill((15, 15, 20))
                self.screen.blit(self.track.get_surface(), (0, 0))
                
                for car in car_ai.cars:
                    if car.alive:
                        car.draw(self.screen)

                if best_car:
                    for sensor in best_car.sensors:
                        pygame.draw.line(self.screen, Color.GREEN, best_car.center, sensor[0], 2)
                        pygame.draw.circle(self.screen, Color.RED, sensor[0], 4)

                self.stats_panel.draw(
                    self.screen, 
                    x_offset=self.TRACK_WIDTH, y_offset=0,
                    generation=car_ai.TOTAL_GENERATIONS,
                    total_cars=total_cars,
                    alive=car_ai.remaining_cars,
                    time_left=frames_left / 60.0,
                    time_limit=CarAI.MAX_FRAMES / 60.0,
                    best_fitness=car_ai.best_fitness,
                    median_fitness=median_fitness,
                    max_speed=max_speed,
                    best_net_nodes=best_net_nodes,
                    best_net_conns=best_net_conns
                )

                if car_ai.best_nn:
                    car_ai.best_nn.draw(self.screen)

                pygame.display.update()
                # self.clock.tick(self.FPS) # Uncomment to cap FPS in GUI mode

            frame_count += 1

        # The ALL-TIME Median Record uses the entire pack's peak performance (all 256 cars).
        # This prevents the median from being falsely skewed if only 2 elite cars survive.
        if gen_best_median > self.all_time_best_median[0]:
            self.all_time_best_median = (gen_best_median, car_ai.TOTAL_GENERATIONS)

        # However, Top Speed and Highest Fitness STILL mandate that the car survives the full timer.
        survivors = [(car, g) for car, (_, g) in zip(car_ai.cars, car_ai.genomes) if car.alive]
        
        best_survivor_fit = 0.0
        best_survivor_speed = 0.0
        
        if survivors:
            best_survivor_fit = max([g.fitness for c, g in survivors if g.fitness is not None] + [0.0])
            best_survivor_speed = max([car_max_speeds[id(c)] for c, g in survivors] + [0.0])

            if best_survivor_fit > self.all_time_best_fitness[0]:
                self.all_time_best_fitness = (best_survivor_fit, car_ai.TOTAL_GENERATIONS)
            if best_survivor_speed > self.all_time_top_speed[0]:
                self.all_time_top_speed = (best_survivor_speed, car_ai.TOTAL_GENERATIONS)

        # Save final generation stats to display next round
        # As requested, only show the best fitness and top speed of ALIVE cars from that generation.
        self.prev_gen_stats = {
            'gen': car_ai.TOTAL_GENERATIONS,
            'best_fitness': best_survivor_fit,
            'median_fitness': gen_best_median,
            'top_speed': best_survivor_speed
        }

        # GENERATION ENDED: Update Hall of Fame
        for _, g in car_ai.genomes:
            if g.fitness is not None:
                self.hall_of_fame.append((g.fitness, copy.deepcopy(g)))
                
        unique_hof = {}
        for fit, g in sorted(self.hall_of_fame, key=lambda x: x[0], reverse=True):
            brain_hash = str([(c.key, round(c.weight, 2)) for c in g.connections.values() if c.enabled])
            if brain_hash not in unique_hof:
                unique_hof[brain_hash] = (fit, g)
            if len(unique_hof) >= 16:
                break
                
        self.hall_of_fame = list(unique_hof.values())
        
        try:
            with open("data/hall_of_fame.pkl", "wb") as f:
                pickle.dump(self.hall_of_fame, f)
        except Exception as e:
            pass

        # CUSTOM STAGNATION TERMINATION
        # If the Highest Fitness, Median Score, and Top Speed were all set more than 25 generations ago
        gen_diff_fitness = car_ai.TOTAL_GENERATIONS - self.all_time_best_fitness[1]
        gen_diff_median = car_ai.TOTAL_GENERATIONS - self.all_time_best_median[1]
        gen_diff_speed = car_ai.TOTAL_GENERATIONS - self.all_time_top_speed[1]
        
        if gen_diff_fitness > 25 and gen_diff_median > 25 and gen_diff_speed > 25:
            if hasattr(self, 'StagnationTermination'):
                raise self.StagnationTermination()

    def run(self):
        while True:
            if not self.handle_events():
                break

            if self.state == "ai_running":
                self.start_ai()
                break

            self.draw()

        pygame.quit()