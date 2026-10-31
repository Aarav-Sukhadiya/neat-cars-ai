"""
The ``Car`` object: per-car state, drawing, and a scalar (pure-Python) physics model.

During training the authoritative physics state lives in
:class:`src.ai.vector_physics.VectorizedPhysics`; ``Car`` objects are only kept
in sync for rendering. The scalar methods here (``update_physics``,
``check_sensor``, ``get_reward`` ...) are used by manual driving mode
(``play.py``), the CPU fallback path, and the unit tests.
"""

import math
import pygame
from src.render.colors import Color
from src.render.track import Track


class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3


class Car:

    CAR_SIZE_X = 30
    CAR_SIZE_Y = 30

    MINIMUM_SPEED = 10
    ANGLE_INCREMENT = 10
    SPEED_INCREMENT = 1

    DEFAULT_SPEED = 10
    DEFAULT_ANGLE = 0

    COLLISION_SURFACE_COLOR = Color.WHITE

    # Sensor ray directions relative to the car heading (degrees). These are the
    # network's 7 inputs, in order. Shared with the GPU raycaster.
    SENSOR_ANGLES = (-90, -45, -20, 0, 20, 45, 90)

    DRAW_SENSORS = False
    SENSORS_DRAW_DISTANCE = 1920   # max ray length in pixels

    STARVATION_FRAMES = 300        # die if no checkpoint is crossed for this many frames

    def __init__(self, start_position: list, track: Track, checkpoints: list = None):
        self.checkpoints = checkpoints or []
        self.current_checkpoint_index = 0
        self.frames_since_last_checkpoint = 0
        self.checkpoint_fitness = 0.0
        self.max_fitness_achieved = -float('inf')
        self.position = list(start_position)

        self.angle = Car.DEFAULT_ANGLE
        self.speed = Car.DEFAULT_SPEED

        self.center = [
            self.position[0] + Car.CAR_SIZE_X / 2,
            self.position[1] + Car.CAR_SIZE_Y / 2
        ]
        self.corners = []

        self.sensors = []
        self.alive = True

        self.driven_distance = 0
        self.frames_alive = 0
        self.speed_penalty = 0

        # Normalisers for the no-checkpoint fallback reward.
        self.track_diagonal = math.hypot(track.width, track.height)
        self.DISTANCE_NORMALIZER = self.track_diagonal / 2
        self.MAX_EXPECTED_SPEED = self.track_diagonal / 100
        self.penalty_factor = self.track_diagonal / 1000

    # pragma: no cover
    def draw(self, screen: pygame.Surface, camera=None) -> None:
        """Draw the car polygon on the screen, applying camera offset if provided."""
        if not self.corners:
            return

        color = Color.BLUE if self.alive else Color.RED

        if camera:
            shifted_corners = [camera.apply(c) for c in self.corners]
        else:
            shifted_corners = self.corners

        pygame.draw.polygon(screen, color, shifted_corners)

        front_left = shifted_corners[3]
        front_right = shifted_corners[0]
        pygame.draw.line(screen, Color.YELLOW, front_left, front_right, 4)

        if Car.DRAW_SENSORS and self.alive and self.sensors:
            shifted_center = camera.apply(self.center) if camera else self.center
            for sensor in self.sensors:
                hit = camera.apply(sensor[0]) if camera else sensor[0]
                pygame.draw.line(screen, Color.GREEN, shifted_center, hit, 2)
                pygame.draw.circle(screen, Color.RED, hit, 4)

    def check_collision(self, track: pygame.Surface) -> bool:
        """Kill the car if any corner is off the track surface or on a wall pixel."""
        track_x = track.get_width()
        track_y = track.get_height()
        for point in self.corners:
            if point[0] < 0 or point[0] >= track_x or point[1] < 0 or point[1] >= track_y:
                self.alive = False
                return True

            elif track.get_at((int(point[0]), int(point[1]))) == Car.COLLISION_SURFACE_COLOR:
                self.alive = False
                return True

        return False

    def refresh_corners_positions(self) -> None:
        """Recompute the 4 drawn corner positions from ``center`` and ``angle``."""
        length_x = 0.5 * Car.CAR_SIZE_X
        length_y = 0.5 * Car.CAR_SIZE_Y

        corners = []
        for offset in (30, 150, 210, 330):
            rad = math.radians(360 - (self.angle + offset))
            corners.append([
                self.center[0] + math.cos(rad) * length_x,
                self.center[1] + math.sin(rad) * length_y
            ])
        self.corners = corners

    def check_sensor(self, degree: int, track: pygame.Surface) -> None:
        """March a ray from the car centre until it hits a wall and record it in ``sensors``.

        Args:
            degree: Sensor angle relative to the car's heading.
            track: Surface whose ``COLLISION_SURFACE_COLOR`` pixels are walls.
        """
        radians = math.radians(360 - (self.angle + degree))
        cos = math.cos(radians)
        sin = math.sin(radians)
        length = 1

        x, y = int(self.center[0]), int(self.center[1])
        track_x = track.get_width()
        track_y = track.get_height()

        while 0 < x < track_x and 0 < y < track_y and track.get_at((x, y)) != Car.COLLISION_SURFACE_COLOR:
            x = int(self.center[0] + cos * length)
            y = int(self.center[1] + sin * length)

            if length > Car.SENSORS_DRAW_DISTANCE:
                break

            length += 1

        distance = int(math.hypot(x - self.center[0], y - self.center[1]))
        self.sensors.append([(x, y), distance])

    def refresh_sensors(self, track: pygame.Surface) -> None:
        """Recompute all sensors on the CPU (slow; used by manual mode and fallbacks)."""
        self.sensors.clear()
        for sensor_angle in Car.SENSOR_ANGLES:
            self.check_sensor(sensor_angle, track)

    def update_center(self) -> None:
        """Update the center of the car using its top-left position."""
        self.center = [
            int(self.position[0]) + Car.CAR_SIZE_X / 2,
            int(self.position[1]) + Car.CAR_SIZE_Y / 2
        ]

    @staticmethod
    def distance_to_line(p, line_start, line_end):
        """Shortest distance from point ``p`` to the segment ``line_start``-``line_end``."""
        px, py = p
        sx, sy = line_start
        ex, ey = line_end

        line_mag = math.hypot(ex - sx, ey - sy)
        if line_mag == 0:
            return math.hypot(px - sx, py - sy)

        u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / (line_mag ** 2)
        u = max(0, min(1, u))
        ix = sx + u * (ex - sx)
        iy = sy + u * (ey - sy)
        return math.hypot(px - ix, py - iy)

    def check_checkpoint(self, old_center, new_center) -> bool:
        """Advance to the next checkpoint if the segment old->new crossed the current one.

        Returns True if a checkpoint was crossed.
        """
        if not self.checkpoints:
            return False
        target_cp = self.checkpoints[self.current_checkpoint_index % len(self.checkpoints)]
        p3, p4 = (target_cp[0], target_cp[1]), (target_cp[2], target_cp[3])

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        intersect = ccw(old_center, new_center, p3) != ccw(old_center, new_center, p4) and \
            ccw(old_center, p3, p4) != ccw(new_center, p3, p4)

        if intersect:
            self.current_checkpoint_index += 1
            self.checkpoint_fitness += 1000.0 + max(0.0, 500.0 - self.frames_since_last_checkpoint)
            self.frames_since_last_checkpoint = 0
        return intersect

    def update_physics(self, track: pygame.Surface) -> None:
        """Update car physics (position, checkpoints, collision) WITHOUT computing sensors."""
        if self.alive:
            self.frames_alive += 1
        old_center = list(self.center)
        self.update_center()

        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed

        if self.alive:
            self.driven_distance += self.speed

        self.update_center()

        if self.alive and self.checkpoints:
            self.check_checkpoint(old_center, self.center)
            self.frames_since_last_checkpoint += 1
            if self.frames_since_last_checkpoint > Car.STARVATION_FRAMES:
                self.alive = False

        self.refresh_corners_positions()
        self.check_collision(track)

    def get_data(self) -> list[int]:
        """Sensor distances as the network's input vector (padded to 7 entries)."""
        distances = [int(sensor[1]) for sensor in self.sensors]
        distances += [0] * (len(Car.SENSOR_ANGLES) - len(distances))
        return distances

    def get_reward(self) -> float:
        """Scalar fitness (the vectorised equivalent lives in VectorizedPhysics.update_fitness)."""
        if not self.checkpoints:
            # Fallback if the track has no checkpoints
            distance_reward = self.driven_distance / self.DISTANCE_NORMALIZER
            speed_reward = (max(self.speed, 0) / self.MAX_EXPECTED_SPEED) ** 0.5
            malus = self.speed_penalty / self.penalty_factor
            progress_factor = min(1.0, self.driven_distance / (self.track_diagonal * 0.75))
            return (distance_reward + speed_reward - malus) * (1 + progress_factor)

        # Checkpoint progression: base checkpoint score minus distance to the NEXT checkpoint,
        # so moving closer increases fitness. The peak value is kept.
        target_cp = self.checkpoints[self.current_checkpoint_index % len(self.checkpoints)]
        dist_to_next = self.distance_to_line(self.center, (target_cp[0], target_cp[1]), (target_cp[2], target_cp[3]))

        current_fit = self.checkpoint_fitness - dist_to_next
        if current_fit > self.max_fitness_achieved:
            self.max_fitness_achieved = current_fit

        return self.max_fitness_achieved

    def accelerate(self) -> None:
        self.speed += Car.SPEED_INCREMENT

    def brake(self) -> None:
        """Slow down, never below MINIMUM_SPEED (braking at the floor is penalised)."""
        if self.speed > Car.MINIMUM_SPEED:
            self.speed -= Car.SPEED_INCREMENT
        else:
            self.speed = Car.MINIMUM_SPEED
            self.speed_penalty += 1

    def turn_left(self) -> None:
        self.angle += Car.ANGLE_INCREMENT

    def turn_right(self) -> None:
        self.angle -= Car.ANGLE_INCREMENT
