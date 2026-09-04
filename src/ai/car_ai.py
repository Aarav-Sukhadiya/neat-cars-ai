# ------------------ IMPORTS ------------------


import neat
import pygame
import numpy as np
from src.render.car import Car, Action
from src.ai.vector_physics import VectorizedPhysics
from src.render.neural_network.nn import NN
from src.render.track import Track

# GPU acceleration modules (gracefully degrade if unavailable)
try:
    from src.gpu.gpu_inference import GPUBatchInference, DEVICE
    _GPU_INFERENCE = True
except ImportError:
    _GPU_INFERENCE = False

try:
    from src.gpu.gpu_raycast import GPURaycast
    _GPU_RAYCAST = True
except ImportError:
    _GPU_RAYCAST = False


# ------------------ CLASSES ------------------

class CarAI:

    TOTAL_GENERATIONS = 0
    MAX_FRAMES = 1200  # 1200 frames = exactly 20 simulated seconds at 60 FPS

    def __init__(self, genomes: neat.DefaultGenome, config: neat.Config, start_position: list, track: Track, checkpoints_path: str = 'data/checkpoints.json'):
        CarAI.TOTAL_GENERATIONS += 1

        self.genomes = genomes
        self.config = config

        self.cars = []
        self.nets = []
        self.best_fitness = -float('inf')
        self.best_nn = None
        self.best_input = None

        # Build cars and neat nets in one pass.
        #
        # KEY OPTIMISATION: NN visualisation objects (NN class) are expensive —
        # creating 2000 of them costs ~115ms per generation. We defer creation
        # to on-demand: _nn_cache[i] is built only the first time car i becomes
        # the best-fitness car and needs to be shown on screen.
        self._nn_cache: dict = {}

        import os
        import json
        self.checkpoints = []
        if os.path.exists(checkpoints_path):
            with open(checkpoints_path, 'r') as f:
                self.checkpoints = json.load(f)

        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.nets.append(net)
            genome.fitness = 0
            self.cars.append(Car(start_position, track, checkpoints=self.checkpoints))

        # nns kept as a list of None; populated lazily via _get_nn()
        self.nns = [None] * len(self.cars)
        self.remaining_cars = len(self.cars)

        # ---------------------------------------------------------------
        # GPU setup
        # ---------------------------------------------------------------
        # Pass prebuilt_nets so GPUBatchInference doesn't call
        # FeedForwardNetwork.create() a second time (~170ms saved).
        if _GPU_INFERENCE:
            self._gpu_inference = GPUBatchInference(
                genomes, config, prebuilt_nets=self.nets
            )
            print(f"[GPU] Neural network inference enabled (device: {DEVICE})")
        else:
            self._gpu_inference = None
            print("[GPU] Neural network inference DISABLED – using CPU neat-python fallback")

        if _GPU_RAYCAST:
            self._gpu_raycast = GPURaycast(track.width, track.height)
            print(f"[GPU] Raycasting enabled (CuPy CUDA kernel)")
        else:
            self._gpu_raycast = None
            print("[GPU] Raycasting DISABLED – using CPU fallback (may still be vectorised NumPy)")

        # Pre-allocate a reusable inputs buffer for all cars
        n_inputs = len(config.genome_config.input_keys)
        self._inputs_buf = np.zeros((len(self.cars), n_inputs), dtype=np.float32)
        
        self.vector_physics = VectorizedPhysics(self.cars, track.width, track.height, self.checkpoints)

    def _get_nn(self, i: int) -> NN:
        """Return the NN visualisation for car i, creating it lazily on first access."""
        if self.nns[i] is None:
            _, genome = self.genomes[i]
            # Place the NN visualization in the Stats Panel area (top right)
            self.nns[i] = NN(self.config, genome, (1550, 270))
        return self.nns[i]

    # ------------------------------------------------------------------
    # Track pixel array cache (avoid calling pygame.surfarray every frame)
    # ------------------------------------------------------------------
    _cached_track_surface = None
    _cached_track_pixels = None

    @staticmethod
    def _get_track_pixels(track_surface: pygame.Surface) -> np.ndarray:
        """Return a C-contiguous (H, W) uint8 array where 255 = wall, 0 = drivable.

        Uses the surface's own colour-shift values for portability, and forces
        C-contiguous layout so the CUDA kernel can index with [iy * track_w + ix].
        Cached per-surface so pygame.surfarray is only called when the track changes.
        """
        if CarAI._cached_track_surface is not track_surface:
            r_shift = track_surface.get_shifts()[0]
            packed = pygame.surfarray.array2d(track_surface).T   # (W,H) → (H,W) F-order
            r_channel = (packed >> r_shift) & 0xFF
            wall = (r_channel > 200).astype(np.uint8) * 255
            # CRITICAL: force C-contiguous (row-major) so CUDA kernel indexing is correct
            CarAI._cached_track_pixels  = np.ascontiguousarray(wall)
            CarAI._cached_track_surface = track_surface
        return CarAI._cached_track_pixels

    # ------------------------------------------------------------------
    # Main compute loop
    # ------------------------------------------------------------------

    def compute(self, track: pygame.Surface) -> None:
        """Compute the next move for every alive car.

        Steps:
        1. GPU raycast – batch all alive cars' sensors in one CUDA call.
        2. Inject sensor results into each Car object.
        3. GPU NN inference – run all cars' networks in one batched matmul.
        4. Apply outputs (steer / accelerate) per car.
        5. Update physics (position, collision) per car.
        6. Update fitness.
        """

        # ---- VECTORIZED COMPUTE ------------------------------------
        alive_mask = self.vector_physics.alive
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            self.remaining_cars = 0
            return

        track_pixels = self._get_track_pixels(track)

        # 1. GPU Raycast (using vector_physics centers directly!)
        if self._gpu_raycast is not None:
            cx = self.vector_physics.cx[alive_mask]
            cy = self.vector_physics.cy[alive_mask]
            angles = self.vector_physics.angle[alive_mask]
            
            distances, hit_x, hit_y = self._gpu_raycast.compute(track_pixels, cx, cy, angles)
            
            # FAST INJECT: directly write distances to inputs_buf
            self._inputs_buf[alive_indices] = distances
            
            # Sync hit points for drawing (only for active cars)
            # This is slightly slow but necessary if Pygame rendering is on and you want laser lines
            for li, idx in enumerate(alive_indices):
                self.cars[idx].sensors = [[(int(hx), int(hy)), int(d)] for hx, hy, d in zip(hit_x[li], hit_y[li], distances[li])]
        else:
            # Fallback
            for i in alive_indices:
                self.cars[i].update_center()
                self.cars[i].sensors.clear()
                for sensor_angle in [-90, -45, -20, 0, 20, 45, 90]:
                    self.cars[i].check_sensor(sensor_angle, track)
                self._inputs_buf[i, :] = self.cars[i].get_data()

        # 2. Batched GPU Inference
        if self._gpu_inference is not None:
            all_outputs_full = self._gpu_inference.activate_all(self._inputs_buf)
            all_outputs = all_outputs_full[alive_indices]
        else:
            all_outputs = np.zeros((len(alive_indices), 4), dtype=np.float32)
            for li, i in enumerate(alive_indices):
                all_outputs[li] = self.nets[i].activate(self._inputs_buf[i].tolist())

        choices = np.argmax(all_outputs, axis=1)

        # 3. Vectorized Physics Step
        full_choices = np.zeros(len(self.cars), dtype=np.int32)
        full_choices[alive_indices] = choices
        
        self.vector_physics.step(full_choices, track_pixels)
        self.vector_physics.update_fitness()
        
        # 4. Sync State back to Python objects (for Pygame rendering and NEAT genomes)
        self.vector_physics.sync_to_cars(alive_indices)
        
        # 5. Extract Fitness
        self.remaining_cars = 0
        for li, i in enumerate(alive_indices):
            if self.vector_physics.alive[i]:
                self.remaining_cars += 1
                fit = float(self.vector_physics.max_fitness[i])
                self.genomes[i][1].fitness = fit
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_nn = self._get_nn(i)
                    if self.best_nn is not None:
                        # Update visualizer
                        for node in self.best_nn.nodes:
                            node.inputs = self._inputs_buf[i]
                            node.output = choices[li]
