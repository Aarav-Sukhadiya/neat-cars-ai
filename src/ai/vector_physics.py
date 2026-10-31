"""
Vectorised (NumPy) physics for the whole population.

Every per-car quantity (position, heading, speed, checkpoint progress, ...)
lives in a flat array indexed by car. One call to :meth:`VectorizedPhysics.step`
advances *all* cars by one frame, so the cost per frame is a handful of
NumPy operations regardless of population size.

Coordinate conventions (shared with :class:`src.render.car.Car`):

* ``x, y``   – top-left corner of the car's bounding box, in track pixels.
* ``cx, cy`` – centre of the car (``x + CAR_SIZE_X / 2`` ...).
* ``angle``  – heading in degrees, counter-clockwise, 0 = facing +x (right).
  Screen y grows downward, so motion uses ``radians(360 - angle)``.
* ``track_pixels`` – ``(H, W)`` uint8 array where ``255`` = wall.
"""

import numpy as np
from src.render.car import Car

# Fitness constants (kept identical to Car.update_physics / Car.get_reward).
CHECKPOINT_REWARD = 1000.0       # base reward for crossing a checkpoint
CHECKPOINT_TIME_BONUS = 500.0    # bonus, minus frames taken since the previous checkpoint
STARVATION_FRAMES = 300          # die if no checkpoint is crossed for this many frames


class VectorizedPhysics:
    def __init__(self, cars, track_width, track_height, checkpoints):
        self.num_cars = len(cars)
        self.cars = cars
        self.track_width = track_width
        self.track_height = track_height

        self.alive = np.ones(self.num_cars, dtype=bool)
        self.x = np.array([c.position[0] for c in cars], dtype=np.float32)
        self.y = np.array([c.position[1] for c in cars], dtype=np.float32)
        self.angle = np.array([c.angle for c in cars], dtype=np.float32)
        self.speed = np.array([c.speed for c in cars], dtype=np.float32)

        self.cx = self.x + Car.CAR_SIZE_X / 2
        self.cy = self.y + Car.CAR_SIZE_Y / 2

        self.driven_distance = np.zeros(self.num_cars, dtype=np.float32)
        self.speed_penalty = np.zeros(self.num_cars, dtype=np.float32)
        self.frames_alive = np.zeros(self.num_cars, dtype=np.int32)
        self.max_fitness = np.zeros(self.num_cars, dtype=np.float32)

        # Collision box corners relative to the car centre (un-rotated).
        half_x, half_y = Car.CAR_SIZE_X / 2, Car.CAR_SIZE_Y / 2
        self._corners_dx = np.array([-half_x, half_x, half_x, -half_x], dtype=np.float32)
        self._corners_dy = np.array([-half_y, -half_y, half_y, half_y], dtype=np.float32)

        # Checkpoint tracking
        self.has_checkpoints = len(checkpoints) > 0
        if self.has_checkpoints:
            self.checkpoints = np.array(checkpoints, dtype=np.float32)  # (N_cp, 4): x1, y1, x2, y2
            self.cp_index = np.zeros(self.num_cars, dtype=np.int32)
            self.cp_fitness = np.zeros(self.num_cars, dtype=np.float32)
            self.frames_since_cp = np.zeros(self.num_cars, dtype=np.int32)
        else:
            # Constants for the no-checkpoint fallback reward (mirrors Car.get_reward).
            diagonal = float(np.hypot(track_width, track_height))
            self._distance_normalizer = diagonal / 2
            self._max_expected_speed = diagonal / 100
            self._penalty_factor = diagonal / 1000
            self._progress_distance = diagonal * 0.75

    @staticmethod
    def ccw(A_x, A_y, B_x, B_y, C_x, C_y):
        """Vectorised counter-clockwise orientation test for points A, B, C."""
        return (C_y - A_y) * (B_x - A_x) > (B_y - A_y) * (C_x - A_x)

    def _active_checkpoints(self):
        return self.checkpoints[self.cp_index % len(self.checkpoints)]  # (N, 4)

    def step(self, choices, track_pixels):
        """Advance every alive car by one frame.

        Args:
            choices: (N,) int array of actions (0=Left, 1=Right, 2=Accelerate, 3=Brake).
                     Entries for dead cars are ignored.
            track_pixels: (H, W) uint8 array, 255 = wall.
        """
        alive_mask = self.alive
        if not np.any(alive_mask):
            return

        self.frames_alive[alive_mask] += 1

        left = (choices == 0) & alive_mask
        right = (choices == 1) & alive_mask
        accel = (choices == 2) & alive_mask
        brake = (choices == 3) & alive_mask

        self.angle[left] += Car.ANGLE_INCREMENT
        self.angle[right] -= Car.ANGLE_INCREMENT

        self.speed[accel] += Car.SPEED_INCREMENT
        self.speed[brake] = np.maximum(Car.MINIMUM_SPEED, self.speed[brake] - Car.SPEED_INCREMENT)

        # Brake penalty (only used by the no-checkpoint fallback reward)
        self.speed_penalty[brake & (self.speed <= Car.MINIMUM_SPEED)] += 1

        # Kinematics
        radians = np.radians(360 - self.angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)

        old_cx = self.cx
        old_cy = self.cy

        self.x[alive_mask] += cos_a[alive_mask] * self.speed[alive_mask]
        self.y[alive_mask] += sin_a[alive_mask] * self.speed[alive_mask]
        self.driven_distance[alive_mask] += self.speed[alive_mask]

        self.cx = self.x + Car.CAR_SIZE_X / 2
        self.cy = self.y + Car.CAR_SIZE_Y / 2

        # Collision: test the 4 rotated corners of the car's box against the wall map.
        # Leaving the track image counts as a crash (as in Car.check_collision).
        hit_wall = np.zeros(self.num_cars, dtype=bool)
        for dx, dy in zip(self._corners_dx, self._corners_dy):
            corner_x = self.cx + dx * cos_a - dy * sin_a
            corner_y = self.cy + dx * sin_a + dy * cos_a

            off_map = ((corner_x < 0) | (corner_x >= self.track_width) |
                       (corner_y < 0) | (corner_y >= self.track_height))
            cx_int = np.clip(corner_x.astype(np.int32), 0, self.track_width - 1)
            cy_int = np.clip(corner_y.astype(np.int32), 0, self.track_height - 1)

            hit_wall |= off_map | (track_pixels[cy_int, cx_int] == 255)

        self.alive[alive_mask & hit_wall] = False

        # Checkpoints: did the centre's path this frame cross the car's next checkpoint line?
        if self.has_checkpoints:
            active_cps = self._active_checkpoints()
            p3_x, p3_y = active_cps[:, 0], active_cps[:, 1]
            p4_x, p4_y = active_cps[:, 2], active_cps[:, 3]

            ccw1 = self.ccw(old_cx, old_cy, self.cx, self.cy, p3_x, p3_y)
            ccw2 = self.ccw(old_cx, old_cy, self.cx, self.cy, p4_x, p4_y)
            ccw3 = self.ccw(old_cx, old_cy, p3_x, p3_y, p4_x, p4_y)
            ccw4 = self.ccw(self.cx, self.cy, p3_x, p3_y, p4_x, p4_y)

            intersect = (ccw1 != ccw2) & (ccw3 != ccw4) & alive_mask

            self.cp_index[intersect] += 1
            self.cp_fitness[intersect] += CHECKPOINT_REWARD + np.maximum(
                0.0, CHECKPOINT_TIME_BONUS - self.frames_since_cp[intersect])
            self.frames_since_cp[intersect] = 0

            self.frames_since_cp[alive_mask] += 1
            starved = self.frames_since_cp > STARVATION_FRAMES
            self.alive[alive_mask & starved] = False

    def update_fitness(self):
        """Update ``max_fitness`` (the NEAT fitness) for every car still alive.

        With checkpoints: ``checkpoint score - distance to the next checkpoint line``,
        keeping the best value ever reached (so fitness never decreases).
        Without checkpoints: a vectorised port of :meth:`Car.get_reward`'s fallback.
        """
        alive_mask = self.alive
        if not np.any(alive_mask):
            return

        if not self.has_checkpoints:
            current_fit = self._fallback_reward()
        else:
            active_cps = self._active_checkpoints()

            px, py = self.cx, self.cy
            sx, sy = active_cps[:, 0], active_cps[:, 1]
            ex, ey = active_cps[:, 2], active_cps[:, 3]

            # Distance from the car centre to the checkpoint segment.
            line_mag_sq = (ex - sx) ** 2 + (ey - sy) ** 2
            u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / np.maximum(1e-5, line_mag_sq)
            u = np.clip(u, 0, 1)
            ix = sx + u * (ex - sx)
            iy = sy + u * (ey - sy)

            dist_to_next = np.hypot(px - ix, py - iy)
            current_fit = self.cp_fitness - dist_to_next

        improved = (current_fit > self.max_fitness) & alive_mask
        self.max_fitness[improved] = current_fit[improved]

    def _fallback_reward(self):
        """Distance/speed based reward used when a track has no checkpoints."""
        distance_reward = self.driven_distance / self._distance_normalizer
        speed_reward = np.sqrt(np.maximum(self.speed, 0) / self._max_expected_speed)
        malus = self.speed_penalty / self._penalty_factor
        progress_factor = np.minimum(1.0, self.driven_distance / self._progress_distance)
        return ((distance_reward + speed_reward - malus) * (1 + progress_factor)).astype(np.float32)

    def sync_to_cars(self, indices):
        """Copy array state back onto the given ``Car`` objects (used for rendering/stats)."""
        for i in indices:
            car = self.cars[i]
            car.position[0] = float(self.x[i])
            car.position[1] = float(self.y[i])
            car.angle = float(self.angle[i])
            car.speed = float(self.speed[i])
            car.alive = bool(self.alive[i])
            car.center = [float(self.cx[i]), float(self.cy[i])]
            car.max_fitness_achieved = float(self.max_fitness[i])
            car.driven_distance = float(self.driven_distance[i])
            car.frames_alive = int(self.frames_alive[i])
            car.refresh_corners_positions()
