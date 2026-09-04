# ------------------ IMPORTS ------------------


import neat
import pygame
import numpy as np
from src.render.car import Car, Action
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

    def __init__(self, genomes: neat.DefaultGenome, config: neat.Config, start_position: list, track: Track):
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
        if os.path.exists('data/checkpoints.json'):
            with open('data/checkpoints.json', 'r') as f:
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

        alive_indices = [i for i, car in enumerate(self.cars) if car.alive]
        if not alive_indices:
            return

        # ---- Step 1 & 2: GPU batch raycast --------------------------------
        if self._gpu_raycast is not None:
            track_pixels = self._get_track_pixels(track)
            alive_cars = [self.cars[i] for i in alive_indices]
            cx     = np.array([c.center[0] for c in alive_cars], dtype=np.float32)
            cy     = np.array([c.center[1] for c in alive_cars], dtype=np.float32)
            angles = np.array([c.angle     for c in alive_cars], dtype=np.float32)

            distances, hit_x, hit_y = self._gpu_raycast.compute(track_pixels, cx, cy, angles)

            for li, i in enumerate(alive_indices):
                self.cars[i].inject_sensors(
                    distances[li].tolist(),
                    list(zip(hit_x[li].tolist(), hit_y[li].tolist()))
                )
        # If no GPU raycast, sensors will be computed inside update_sprite()

        # ---- Step 3 & 4: Inputs & Batched GPU Inference -------------------
        for i in alive_indices:
            self._inputs_buf[i, :] = self.cars[i].get_data()

        if self._gpu_inference is not None:
            all_outputs_full = self._gpu_inference.activate_all(self._inputs_buf)
            all_outputs = all_outputs_full[alive_indices]
        else:
            all_outputs = np.zeros((len(alive_indices), 4), dtype=np.float32)
            for li, i in enumerate(alive_indices):
                out = self.nets[i].activate(self._inputs_buf[i].tolist())
                all_outputs[li] = out

        # ---- Step 5: Apply outputs ----------------------------------------
        choices = np.argmax(all_outputs, axis=1)  # (N_alive,)

        for li, i in enumerate(alive_indices):
            car    = self.cars[i]
            choice = int(choices[li])

            # Update NN visualisation node data ONLY for the current best car
            # (avoids iterating nodes for all 2000 cars every frame)
            if self.best_nn is not None and self.nns[i] is self.best_nn:
                for node in self.best_nn.nodes:
                    node.inputs = self.cars[i].get_data()
                    node.output = choice

            if choice == Action.TURN_LEFT:
                car.turn_left()
            elif choice == Action.TURN_RIGHT:
                car.turn_right()
            elif choice == Action.ACCELERATE:
                car.accelerate()
            elif choice == Action.BRAKE:
                car.brake()

        # ---- Step 6: Physics + fitness ------------------------------------
        self.remaining_cars = 0
        for i, car in enumerate(self.cars):
            if not car.alive:
                continue

            if self._gpu_raycast is not None:
                car.update_physics(track)
            else:
                car.update_sprite(track)

            if car.alive:
                self.remaining_cars += 1
                self.genomes[i][1].fitness = car.get_reward()
                if self.genomes[i][1].fitness > self.best_fitness:
                    self.best_fitness = self.genomes[i][1].fitness
                    self.best_nn = self._get_nn(i)   # lazy creation
