"""
Training engine: loads the track, runs NEAT, and renders (GUI) or reports (headless).

Flow::

    Engine.run()
      -> start_ai()                    builds the NEAT population, runs it
           -> population.run(run_simulation, max_simulations)
                -> run_simulation()    one generation: simulate, render, record stats,
                                       update the Hall of Fame, check termination
"""

import copy
import json
import os
import pickle
import sys
import time
from itertools import count
from typing import List, Optional

import neat
import numpy as np
import pygame

from src.ai.car_ai import CarAI
from src.render.camera import Camera
from src.render.car import Car
from src.render.colors import Color
from src.render.minimap import Minimap
from src.render.stats_panel import StatsPanel
from src.render.track import Track

TRACK_CONFIG_PATH = "config/track_config.json"
HALL_OF_FAME_PATH = "data/hall_of_fame.pkl"
BEST_BRAIN_PATH = "data/best_car_brain.pkl"
NEAT_CHECKPOINT_PREFIX = "data/neat_checkpoints/neat-checkpoint-"

HALL_OF_FAME_SIZE = 16
STAGNATION_GENERATIONS = 50
SIM_FPS = 60.0                      # simulated frames per (virtual) second
HEADLESS_PRINT_INTERVAL = 0.066     # seconds between terminal redraws (~15 FPS)
_ANSI_CLEAR = "\033[H\033[2J\033[3J"


class StagnationTermination(Exception):
    """Raised when no all-time record improved for STAGNATION_GENERATIONS generations."""


class SuccessTermination(Exception):
    """Raised when a generation's survival rate reaches --target-survival."""


def _genome_signature(genome) -> str:
    """Identity of a brain for Hall-of-Fame de-duplication (enabled weights, 2 d.p.)."""
    return str([(c.key, round(c.weight, 2)) for c in genome.connections.values() if c.enabled])


class Engine:
    WIDTH = 1900
    HEIGHT = 1080
    SIDEBAR_WIDTH = 400
    FPS = 60

    def __init__(self, neat_config_path: str, debug: bool, max_simulations: int, headless: bool = False,
                 track_id: Optional[int] = None, infinite: bool = False, target_survival: float = 0.0):
        """
        Args:
            neat_config_path: Path to the neat-python config file.
            debug: Attach neat-python reporters (stdout stats + checkpoints) in GUI mode.
            max_simulations: Maximum number of generations.
            headless: Run without a window; progress is printed to the terminal.
            track_id: Index into ``tracks`` of config/track_config.json
                      (None = use its ``current_track``).
            infinite: No generation limit and no stagnation termination.
            target_survival: Stop once a generation's survival rate (%) reaches this (0 = off).
        """
        pygame.init()
        if not headless:
            # Desktop size (display.Info() reports the *window* size once one exists).
            try:
                desk_w, desk_h = pygame.display.get_desktop_sizes()[0]
            except (AttributeError, IndexError, pygame.error):
                info = pygame.display.Info()
                desk_w, desk_h = info.current_w, info.current_h
            if desk_w <= 0 or desk_h <= 0:
                desk_w, desk_h = 1920, 1080
            Engine.WIDTH = max(800, min(1920, int(desk_w * 0.95)))
            Engine.HEIGHT = max(600, min(1080, int(desk_h * 0.95)))
            Engine.SIDEBAR_WIDTH = min(400, int(Engine.WIDTH * 0.25))
        else:
            Engine.WIDTH = 1920
            Engine.HEIGHT = 1080
            Engine.SIDEBAR_WIDTH = 400

        self.neat_config_path = neat_config_path
        self.debug = debug
        self.max_simulations = max_simulations if not infinite else 999999999
        self.HEADLESS = headless
        self.infinite = infinite
        self.target_survival = target_survival
        self.title = "Neat Cars"

        # All-time records: (value, generation)
        self.all_time_best_fitness = (0.0, 0)
        self.all_time_best_median = (0.0, 0)
        self.all_time_survival_rate = (0.0, 0)
        self.all_time_avg_speed = (0.0, 0)
        self.prev_gen_stats = None

        self._load_track_config(track_id)

        if not self.HEADLESS:
            pygame.display.set_caption(self.title)
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.screen = None

        self.clock = pygame.time.Clock()
        self._load_track()

        self.checkpoint_font = pygame.font.SysFont(None, 24)
        self._checkpoint_labels = {}

        self.stats_panel = None if self.HEADLESS else StatsPanel(self.SIDEBAR_WIDTH, self.HEIGHT)
        CarAI.NN_POSITION = (self.WIDTH - self.SIDEBAR_WIDTH + 30, 270)

        self.hall_of_fame = []
        try:
            with open(HALL_OF_FAME_PATH, "rb") as f:
                self.hall_of_fame = pickle.load(f)
            print(f"Loaded {len(self.hall_of_fame)} elite cars from Hall of Fame!")
        except (FileNotFoundError, EOFError):
            print("No Hall of Fame found. Starting fresh.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_track_config(self, track_id: Optional[int]) -> None:
        """Resolve image / mask / checkpoints / start pose from config/track_config.json."""
        self.track_image_path = "assets/track.png"
        self.track_mask_path = self.track_image_path
        self.checkpoints_path = "data/checkpoints.json"
        self.decided_car_pos = [200 - Car.CAR_SIZE_X / 2, 950 - Car.CAR_SIZE_Y / 2]

        if not os.path.exists(TRACK_CONFIG_PATH):
            return
        with open(TRACK_CONFIG_PATH, "r") as f:
            tconfig = json.load(f)

        tracks_dict = tconfig.get("tracks", {})
        track_keys = list(tracks_dict.keys())
        curr = tconfig.get("current_track", "")

        if track_id is not None:
            if not 0 <= track_id < len(track_keys):
                raise ValueError(f"Invalid track id {track_id}; choose 0-{len(track_keys) - 1}.")
            curr = track_keys[track_id]

        if curr not in tracks_dict:
            return
        track_data = tracks_dict[curr]
        self.track_image_path = track_data.get("image", self.track_image_path)
        self.track_mask_path = track_data.get("mask", self.track_image_path)
        self.checkpoints_path = track_data.get("checkpoints", self.checkpoints_path)
        if "start_pos" in track_data:
            # start_pos is the car's centre; the physics uses its top-left corner.
            self.decided_car_pos = [
                track_data["start_pos"][0] - Car.CAR_SIZE_X / 2,
                track_data["start_pos"][1] - Car.CAR_SIZE_Y / 2
            ]
        if "start_angle" in track_data:
            Car.DEFAULT_ANGLE = track_data["start_angle"]
        if "max_frames" in track_data:
            CarAI.MAX_FRAMES = track_data["max_frames"]

    def _load_track(self) -> None:
        """Load the visual image and collision mask, and build camera + minimap."""
        try:
            track_img = pygame.image.load(self.track_image_path)
            track_mask_img = pygame.image.load(self.track_mask_path)
        except (pygame.error, FileNotFoundError) as e:
            # Training on a blank (all-wall) track would kill every car on frame 1
            # forever, so fail loudly instead.
            hint = "" if pygame.image.get_extended() else \
                " (this pygame build has no PNG support; reinstall pygame/pygame-ce)"
            raise RuntimeError(f"Could not load track image '{self.track_image_path}' / "
                               f"mask '{self.track_mask_path}': {e}{hint}") from e

        track_w, track_h = track_img.get_size()
        self.track = Track(track_w, track_h)
        self.track.surface.blit(track_img, (0, 0))
        self.track.mask_surface.blit(track_mask_img, (0, 0))

        self.camera = Camera(self.WIDTH - self.SIDEBAR_WIDTH, self.HEIGHT, track_w, track_h)
        self.minimap = Minimap(self.SIDEBAR_WIDTH - 40, (self.SIDEBAR_WIDTH - 40) * (track_h / track_w),
                               track_w, track_h)
        self.minimap.set_track_surface(self.track.surface)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def handle_events(self) -> bool:
        """Pump events so the OS doesn't consider the window frozen; quit on close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        return True

    def _save_best(self, genome) -> None:
        with open(BEST_BRAIN_PATH, "wb") as f:
            pickle.dump(genome, f)
        print(f"Saved best genome to '{BEST_BRAIN_PATH}'!")

    def start_ai(self) -> None:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.neat_config_path
        )

        self.population = neat.Population(config)

        # Genomes loaded from disk may use node ids beyond the fresh population's counter;
        # advance it so mutations never create duplicate node ids.
        if self.hall_of_fame:
            max_node_id = max((max(g.nodes) for _, g in self.hall_of_fame if g.nodes), default=-1)
            if max_node_id >= 0:
                config.genome_config.node_indexer = count(max_node_id + 1)

        if self.debug and not self.HEADLESS:
            os.makedirs(os.path.dirname(NEAT_CHECKPOINT_PREFIX), exist_ok=True)
            self.population.add_reporter(neat.StdOutReporter(True))
            self.population.add_reporter(neat.StatisticsReporter())
            self.population.add_reporter(neat.Checkpointer(5, filename_prefix=NEAT_CHECKPOINT_PREFIX))

        # Kept as attributes for backwards compatibility.
        self.StagnationTermination = StagnationTermination
        self.SuccessTermination = SuccessTermination

        try:
            winner = self.population.run(self.run_simulation, self.max_simulations)
            self._save_best(winner)
        except (SuccessTermination, StagnationTermination) as stop:
            if isinstance(stop, SuccessTermination):
                print(f"\n[🏆] TARGET REACHED: Map Solved! Survival rate reached or exceeded {self.target_survival}%.")
            else:
                print(f"\n[!] Training terminated: All records have stagnated for more than "
                      f"{STAGNATION_GENERATIONS} generations.")
            if self.hall_of_fame:
                self._save_best(self.hall_of_fame[0][1])
            if self.HEADLESS:
                sys.exit(0)

    def _inject_hall_of_fame(self, config):
        """On generation 0, replace the first genomes with Hall-of-Fame clones.

        Only done once: re-injecting every generation (hyper-elitism) destroys diversity,
        and NEAT's own ``elitism`` already preserves the best genomes between generations.
        """
        pop_keys = list(self.population.population.keys())
        for k, (_, elite_genome) in zip(pop_keys, self.hall_of_fame):
            cloned = copy.deepcopy(elite_genome)
            cloned.key = k
            self.population.population[k] = cloned
        self.population.species.speciate(config, self.population.population, self.population.generation)
        return list(self.population.population.items())

    # pragma: no cover
    def run_simulation(self, genomes: List, config: neat.Config) -> None:
        """NEAT fitness function: simulate one generation and record its statistics."""
        if self.hall_of_fame and self.population.generation == 0:
            genomes = self._inject_hall_of_fame(config)

        car_ai = CarAI(genomes, config, self.decided_car_pos, self.track, checkpoints_path=self.checkpoints_path)
        vp = car_ai.vector_physics
        total_cars = len(car_ai.cars)
        frame_count = 0
        gen_top_speed = 0.0
        gen_best_median = 0.0
        last_print_time = 0.0

        while True:
            if not self.HEADLESS and car_ai.cars:
                # Camera follows the fittest alive car (first on ties).
                alive_idx = np.flatnonzero(vp.alive)
                lead = int(alive_idx[np.argmax(vp.max_fitness[alive_idx])]) if len(alive_idx) else 0
                self.camera.update(car_ai.cars[lead].center)

            if not self.handle_events():
                return

            car_ai.compute(self.track.mask_surface)
            frame_count += 1

            frames_left = max(0, CarAI.MAX_FRAMES - frame_count)

            if car_ai.remaining_cars == 0 or frames_left <= 0:
                break

            # ---- Per-frame metrics (vectorised) ----
            alive_idx = np.flatnonzero(vp.alive)
            max_speed = max(0.0, float(vp.speed[alive_idx].max())) if len(alive_idx) else 0.0
            best_car_idx = int(alive_idx[np.argmax(vp.max_fitness[alive_idx])]) if len(alive_idx) else None
            max_speed_sec = max_speed * SIM_FPS

            # Each genome's fitness equals its car's max_fitness (frozen once it dies).
            median_fitness = float(np.median(vp.max_fitness.astype(np.float64))) if total_cars else 0.0

            best_net_nodes = len(car_ai.best_nn.nodes) if car_ai.best_nn else 0
            best_net_conns = len(car_ai.best_nn.connections) if car_ai.best_nn else 0

            gen_top_speed = max(gen_top_speed, max_speed)
            gen_best_median = max(gen_best_median, median_fitness)

            if self.HEADLESS:
                current_time = time.time()
                if current_time - last_print_time > HEADLESS_PRINT_INTERVAL:
                    last_print_time = current_time
                    self._print_headless(car_ai, total_cars, frames_left, median_fitness,
                                         max_speed_sec, best_net_nodes, best_net_conns)
            else:
                self._render_frame(car_ai, best_car_idx, total_cars, frames_left, median_fitness,
                                   max_speed, best_net_nodes, best_net_conns)

        self._end_generation(car_ai, gen_best_median)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _print_headless(self, car_ai, total_cars, frames_left, median_fitness, max_speed_sec,
                        best_net_nodes, best_net_conns) -> None:
        bar = "=================================================="
        lines = [
            bar,
            f" NEAT CARS (HEADLESS MODE) - GENERATION {car_ai.TOTAL_GENERATIONS}",
            bar,
            f" Alive Cars    : {car_ai.remaining_cars} / {total_cars}",
            f" Time Left     : {frames_left / SIM_FPS:.1f}s / {CarAI.MAX_FRAMES / SIM_FPS:.1f}s (Virtual)",
            f" Best Fitness  : {car_ai.best_fitness:,.1f}",
            f" Median Fitness: {median_fitness:,.1f}",
            f" Survival Rate : {(car_ai.remaining_cars / total_cars * 100.0):.1f}%",
            f" Live Top Speed: {max_speed_sec:,.1f} px/s",
            f" Brain Size    : {best_net_nodes} Nodes / {best_net_conns} Conns",
            bar,
        ]
        if self.prev_gen_stats:
            p = self.prev_gen_stats
            lines += [
                f" PREVIOUS GENERATION (Gen {p['gen']})",
                bar,
                f" Best Fitness  : {p['best_fitness']:,.1f}",
                f" Peak Median   : {p['median_fitness']:,.1f}",
                f" Survival Rate : {p['survival_rate']:.1f}%",
                f" Avg Speed     : {p['avg_speed']:,.1f} px/s",
                bar,
            ]
        lines += [
            " ALL-TIME RECORDS (Across Generations)",
            bar,
            f" Highest Fitness      : {self.all_time_best_fitness[0]:,.1f} (Gen {self.all_time_best_fitness[1]})",
            f" Highest Median Score : {self.all_time_best_median[0]:,.1f} (Gen {self.all_time_best_median[1]})",
            f" Highest Survival Rate: {self.all_time_survival_rate[0]:.1f} % (Gen {self.all_time_survival_rate[1]})",
            f" Highest Avg Speed    : {self.all_time_avg_speed[0]:,.1f} px/s (Gen {self.all_time_avg_speed[1]})",
            bar,
        ]
        clear = _ANSI_CLEAR if os.name != "nt" else ""
        if os.name == "nt":
            os.system("cls")
        sys.stdout.write(clear + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def _checkpoint_label(self, i: int) -> pygame.Surface:
        label = self._checkpoint_labels.get(i)
        if label is None:
            label = self.checkpoint_font.render(str(i), True, (255, 255, 255))
            self._checkpoint_labels[i] = label
        return label

    def _render_frame(self, car_ai, best_car_idx, total_cars, frames_left, median_fitness,
                      max_speed, best_net_nodes, best_net_conns) -> None:  # pragma: no cover
        screen = self.screen
        cam = self.camera
        screen.fill((15, 15, 20))
        screen.blit(self.track.get_surface(), (-cam.offset_x, -cam.offset_y))

        # Checkpoints: start line green, others orange, numbered.
        for i, cp in enumerate(car_ai.checkpoints):
            p1 = cam.apply((cp[0], cp[1]))
            p2 = cam.apply((cp[2], cp[3]))
            color = (255, 165, 0) if i > 0 else (0, 255, 0)
            pygame.draw.line(screen, color, p1, p2, 2)
            screen.blit(self._checkpoint_label(i), ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))

        # Cars (skip those entirely outside the viewport).
        margin = Car.CAR_SIZE_X
        left, top = cam.offset_x - margin, cam.offset_y - margin
        right, bottom = cam.offset_x + cam.width + margin, cam.offset_y + cam.height + margin
        for car in car_ai.cars:
            if car.alive and left <= car.center[0] <= right and top <= car.center[1] <= bottom:
                car.draw(screen, cam)

        if best_car_idx is not None:
            best_car = car_ai.cars[best_car_idx]
            shifted_center = cam.apply(best_car.center)
            for hit, _ in car_ai.get_sensors(best_car_idx):
                shifted_sensor = cam.apply(hit)
                pygame.draw.line(screen, Color.GREEN, shifted_center, shifted_sensor, 2)
                pygame.draw.circle(screen, Color.RED, shifted_sensor, 4)

        cam_rect = pygame.Rect(cam.offset_x, cam.offset_y, cam.width, cam.height)
        self.minimap.draw(screen, car_ai.cars, cam_rect, (20, 20))

        self.stats_panel.draw(
            screen,
            x_offset=self.WIDTH - self.SIDEBAR_WIDTH, y_offset=0,
            generation=car_ai.TOTAL_GENERATIONS,
            total_cars=total_cars,
            alive=car_ai.remaining_cars,
            time_left=frames_left / SIM_FPS,
            time_limit=CarAI.MAX_FRAMES / SIM_FPS,
            best_fitness=car_ai.best_fitness,
            median_fitness=median_fitness,
            max_speed=max_speed,
            best_net_nodes=best_net_nodes,
            best_net_conns=best_net_conns
        )

        if car_ai.best_nn:
            car_ai.best_nn.draw(screen)

        pygame.display.update()

    # ------------------------------------------------------------------
    # End of generation: records, Hall of Fame, termination
    # ------------------------------------------------------------------

    def _end_generation(self, car_ai, gen_best_median: float) -> None:
        gen = car_ai.TOTAL_GENERATIONS
        vp = car_ai.vector_physics
        n_cars = len(car_ai.cars)

        survivors = [g for car, (_, g) in zip(car_ai.cars, car_ai.genomes) if car.alive]
        survival_rate = (len(survivors) / n_cars) * 100.0 if n_cars else 0.0

        # Median record uses the whole pack (so 2 lucky survivors can't skew it).
        if gen_best_median > self.all_time_best_median[0]:
            self.all_time_best_median = (gen_best_median, gen)

        # Average speed of the generation's fittest car (dead or alive).
        fits = [g.fitness if g.fitness is not None else -9999 for _, g in car_ai.genomes]
        best_i = int(np.argmax(fits)) if fits else 0
        best_avg_speed = float(vp.driven_distance[best_i]) / max(1, int(vp.frames_alive[best_i])) * SIM_FPS
        if best_avg_speed > self.all_time_avg_speed[0]:
            self.all_time_avg_speed = (best_avg_speed, gen)

        # Best fitness and survival records require surviving the full timer.
        best_survivor_fit = 0.0
        if survivors:
            best_survivor_fit = max([g.fitness for g in survivors if g.fitness is not None] + [0.0])
            if best_survivor_fit > self.all_time_best_fitness[0]:
                self.all_time_best_fitness = (best_survivor_fit, gen)
            if survival_rate > self.all_time_survival_rate[0]:
                self.all_time_survival_rate = (survival_rate, gen)

        self.prev_gen_stats = {
            'gen': gen,
            'best_fitness': best_survivor_fit,
            'median_fitness': gen_best_median,
            'survival_rate': survival_rate,
            'avg_speed': best_avg_speed
        }

        self._update_hall_of_fame(car_ai.genomes)

        if self.target_survival > 0 and survival_rate >= self.target_survival:
            raise SuccessTermination()

        stale = all(gen - record[1] > STAGNATION_GENERATIONS for record in
                    (self.all_time_best_fitness, self.all_time_best_median, self.all_time_survival_rate))
        if not self.infinite and stale:
            raise StagnationTermination()

    def _update_hall_of_fame(self, genomes) -> None:
        """Keep the top HALL_OF_FAME_SIZE distinct brains ever seen, and persist them.

        Candidates are sorted by fitness (stable: older entries win ties) and only the
        genomes that actually make the cut are deep-copied.
        """
        candidates = list(self.hall_of_fame)
        new_ids = set()
        for _, g in genomes:
            if g.fitness is not None:
                candidates.append((g.fitness, g))
                new_ids.add(id(g))

        unique_hof = {}
        for fit, g in sorted(candidates, key=lambda x: x[0], reverse=True):
            signature = _genome_signature(g)
            if signature not in unique_hof:
                unique_hof[signature] = (fit, g)
            if len(unique_hof) >= HALL_OF_FAME_SIZE:
                break

        old_ids = {id(g) for _, g in self.hall_of_fame}
        self.hall_of_fame = [
            (fit, copy.deepcopy(g) if id(g) in new_ids and id(g) not in old_ids else g)
            for fit, g in unique_hof.values()
        ]

        try:
            os.makedirs(os.path.dirname(HALL_OF_FAME_PATH), exist_ok=True)
            with open(HALL_OF_FAME_PATH, "wb") as f:
                pickle.dump(self.hall_of_fame, f)
        except OSError as e:
            print(f"Warning: could not save Hall of Fame: {e}")

    # pragma: no cover
    def run(self) -> None:
        self.handle_events()
        self.start_ai()
        pygame.quit()
