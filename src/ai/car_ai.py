"""
One generation of cars: builds the population's cars/networks and steps them.

Per frame (:meth:`CarAI.compute`):

1. Raycast the 7 sensors of every alive car (CUDA kernel, or NumPy fallback).
2. Run every alive car's network in one batched call (PyTorch, or neat-python fallback).
3. ``argmax`` of the 4 outputs picks the action (left / right / accelerate / brake).
4. Advance all cars with :class:`VectorizedPhysics` and update fitness.
5. Sync the arrays back onto the ``Car`` objects (for rendering) and write the
   fitness of every surviving car onto its genome.
"""

import json
import os

import neat
import numpy as np
import pygame

from src.ai.vector_physics import VectorizedPhysics
from src.gpu.gpu_raycast import GPURaycast
from src.render.car import Car
from src.render.neural_network.nn import NN
from src.render.track import Track

# GPU inference needs PyTorch; degrade gracefully to neat-python if it's missing.
try:
    from src.gpu.gpu_inference import GPUBatchInference, DEVICE
    _GPU_INFERENCE = True
except ImportError:
    _GPU_INFERENCE = False


class CarAI:

    TOTAL_GENERATIONS = 0
    MAX_FRAMES = 1200  # default episode length (overridden per track by track_config.json)

    # Where the best car's network is drawn (top of the stats sidebar). The Engine
    # overrides this to match the actual window size.
    NN_POSITION = (1550, 270)

    _status_printed = False

    def __init__(self, genomes, config: neat.Config, start_position: list, track: Track,
                 checkpoints_path: str = 'data/checkpoints.json'):
        CarAI.TOTAL_GENERATIONS += 1

        self.genomes = genomes
        self.config = config

        self.best_fitness = -float('inf')
        self.best_nn = None
        self.best_index = -1

        self.checkpoints = []
        if checkpoints_path and os.path.exists(checkpoints_path):
            with open(checkpoints_path, 'r') as f:
                self.checkpoints = json.load(f)

        self.cars = []
        for _, genome in genomes:
            genome.fitness = 0
            self.cars.append(Car(start_position, track, checkpoints=self.checkpoints))

        # NN visualisation objects are expensive (~115ms for 2000); build lazily, only for
        # the car that is actually shown.
        self.nns = [None] * len(self.cars)
        self.remaining_cars = len(self.cars)

        # ---- Inference backend --------------------------------------------------
        if _GPU_INFERENCE:
            self._gpu_inference = GPUBatchInference(genomes, config)
            self.nets = None   # not needed: GPU tensors are built straight from genomes
        else:
            self._gpu_inference = None
            self.nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]

        # ---- Raycast backend (CUDA kernel, or vectorised NumPy fallback) --------
        self._gpu_raycast = GPURaycast(track.width, track.height)

        if not CarAI._status_printed:
            CarAI._status_printed = True
            print(f"[GPU] NN inference: {f'PyTorch on {DEVICE}' if _GPU_INFERENCE else 'neat-python (CPU)'}")
            print(f"[GPU] Raycasting:   {'CuPy CUDA kernel' if self._gpu_raycast.use_gpu else 'NumPy (CPU)'}")

        n_cars = len(self.cars)
        n_inputs = len(config.genome_config.input_keys)
        n_rays = len(Car.SENSOR_ANGLES)
        self._inputs_buf = np.zeros((n_cars, n_inputs), dtype=np.float32)
        self._hit_x = np.zeros((n_cars, n_rays), dtype=np.float32)
        self._hit_y = np.zeros((n_cars, n_rays), dtype=np.float32)

        self.vector_physics = VectorizedPhysics(self.cars, track.width, track.height, self.checkpoints)

    def _get_nn(self, i: int) -> NN:
        """Return the NN visualisation for car i, creating it lazily on first access."""
        if self.nns[i] is None:
            _, genome = self.genomes[i]
            self.nns[i] = NN(self.config, genome, CarAI.NN_POSITION)
        return self.nns[i]

    def get_sensors(self, i: int) -> list:
        """Sensor rays of car i from the latest frame: ``[[(hit_x, hit_y), distance], ...]``."""
        return [[(int(hx), int(hy)), int(d)]
                for hx, hy, d in zip(self._hit_x[i], self._hit_y[i], self._inputs_buf[i])]

    # ------------------------------------------------------------------
    # Track pixel array cache (avoid calling pygame.surfarray every frame)
    # ------------------------------------------------------------------
    _cached_track_surface = None
    _cached_track_pixels = None

    @staticmethod
    def _get_track_pixels(track_surface: pygame.Surface) -> np.ndarray:
        """Return a C-contiguous (H, W) uint8 array where 255 = wall, 0 = drivable.

        A pixel is a wall if its RED channel is > 200 (so white and red are walls).
        Cached per surface object so pygame.surfarray is only called when the track changes.
        """
        if CarAI._cached_track_surface is not track_surface:
            r_shift = track_surface.get_shifts()[0]
            packed = pygame.surfarray.array2d(track_surface).T   # (W,H) -> (H,W), F-order
            r_channel = (packed >> r_shift) & 0xFF
            wall = (r_channel > 200).astype(np.uint8) * 255
            # CUDA kernel indexes [iy * track_w + ix], so force row-major layout.
            CarAI._cached_track_pixels = np.ascontiguousarray(wall)
            CarAI._cached_track_surface = track_surface
        return CarAI._cached_track_pixels

    # ------------------------------------------------------------------
    # Main compute loop
    # ------------------------------------------------------------------

    def compute(self, track: pygame.Surface) -> None:
        """Advance every alive car by one frame (see module docstring for the pipeline)."""
        vp = self.vector_physics
        alive_indices = np.flatnonzero(vp.alive)
        if len(alive_indices) == 0:
            self.remaining_cars = 0
            return

        track_pixels = self._get_track_pixels(track)

        # 1. Raycast sensors for alive cars, written straight into the input buffer.
        distances, hit_x, hit_y = self._gpu_raycast.compute(
            track_pixels, vp.cx[alive_indices], vp.cy[alive_indices], vp.angle[alive_indices])
        self._inputs_buf[alive_indices] = distances
        self._hit_x[alive_indices] = hit_x
        self._hit_y[alive_indices] = hit_y
        if Car.DRAW_SENSORS:
            for i in alive_indices:
                self.cars[i].sensors = self.get_sensors(i)

        # 2. Batched inference for alive cars only.
        if self._gpu_inference is not None:
            all_outputs = self._gpu_inference.activate_all(self._inputs_buf, rows=alive_indices)
        else:
            all_outputs = np.array([self.nets[i].activate(self._inputs_buf[i].tolist())
                                    for i in alive_indices], dtype=np.float32)

        choices = np.argmax(all_outputs, axis=1)

        # 3. Physics step.
        full_choices = np.zeros(len(self.cars), dtype=np.int32)
        full_choices[alive_indices] = choices
        vp.step(full_choices, track_pixels)
        vp.update_fitness()

        # 4. Sync state back to Car objects (rendering / stats).
        vp.sync_to_cars(alive_indices)

        # 5. Fitness of cars that survived this frame -> genomes.
        survived = vp.alive[alive_indices]
        surv_idx = alive_indices[survived]
        self.remaining_cars = len(surv_idx)
        if self.remaining_cars == 0:
            return

        surv_fit = vp.max_fitness[surv_idx]
        for i, fit in zip(surv_idx.tolist(), surv_fit.tolist()):
            self.genomes[i][1].fitness = fit

        # Track the best car so far (first index wins ties, as a running max would).
        k = int(np.argmax(surv_fit))
        top = float(surv_fit[k])
        if top > self.best_fitness:
            i = int(surv_idx[k])
            self.best_fitness = top
            self.best_index = i
            self.best_nn = self._get_nn(i)
            choice = choices[np.flatnonzero(survived)[k]]
            for node in self.best_nn.nodes:
                node.inputs = self._inputs_buf[i]
                node.output = choice
