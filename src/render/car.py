# ------------------ IMPORTS ------------------


import pygame
import math
from src.render.colors import Color
from src.render.track import Track


# ------------------ GLOBAL VARIABLES  ------------------





# ------------------ CLASSES ------------------


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

    DRAW_SENSORS = False
    SENSORS_DRAW_DISTANCE = 1920

    # Class-level sprite cache: loaded once, shared across all cars every generation.
    # This eliminates ~490ms of pygame.image.load + convert_alpha per generation start.
    def __init__(self, start_position: list, track: Track, checkpoints: list = None):
        self.checkpoints = checkpoints or []
        self.current_checkpoint_index = 0
        self.frames_since_last_checkpoint = 0
        self.checkpoint_fitness = 0.0
        self.max_fitness_achieved = -float('inf')
        self.old_position = start_position.copy()
        self.position = start_position.copy()

        self.angle = Car.DEFAULT_ANGLE
        self.speed = Car.DEFAULT_SPEED

        self.center = [
            self.position[0] + Car.CAR_SIZE_X / 2,
            self.position[1] + Car.CAR_SIZE_Y / 2
        ]  # Calculate Center

        self.sensors = []
        self.alive = True
        self.has_been_rendered_as_dead = False
        
        self.driven_distance = 0
        self.speed_penalty = 0
        track_width = track.width
        track_height = track.height
        self.track_diagonal = math.sqrt(track_width**2 + track_height**2)
        
        self.DISTANCE_NORMALIZER = self.track_diagonal / 2
        self.MAX_EXPECTED_SPEED = self.track_diagonal / 100
        self.minimum_speed = self.CAR_SIZE_X / 6
        self.angle_increment = math.degrees(math.atan2(self.CAR_SIZE_Y, self.speed * 10))
        self.penalty_factor = self.track_diagonal / 1000

    def draw(self, track: pygame.Surface) -> None:
        """Draw the car polygon on the track (and its sensors if enabled)"""
        
        if not hasattr(self, 'corners') or not self.corners:
            return
            
        color = Color.BLUE if self.alive else Color.RED
        
        # self.corners is ordered: front-right, back-right, back-left, front-left
        pygame.draw.polygon(track, color, self.corners)
        
        # Draw the front "windshield" / nose indicator (between front-left and front-right)
        front_left = self.corners[3]
        front_right = self.corners[0]
        pygame.draw.line(track, Color.YELLOW, front_left, front_right, 4)

        # Draw the car's sensors
        if Car.DRAW_SENSORS and self.alive:
            for sensor in self.sensors:
                position = sensor[0]
                pygame.draw.line(track, Color.GREEN,
                                 self.center, position, 2)
                pygame.draw.circle(track, Color.RED, position, 4)

    def check_collision(self, track: pygame.Surface) -> bool:
        """Check if the car is colliding with the track (by using a color system)

        Args:
            track (pygame.Surface): The track on which the car is being drawn
        """
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
        """Refresh the corners' current positions of the car (used for collision detection)"""
        length_x = 0.5 * Car.CAR_SIZE_X
        length_y = 0.5 * Car.CAR_SIZE_Y

        corner1 = math.radians(360 - (self.angle + 30))
        corner2 = math.radians(360 - (self.angle + 150))
        corner3 = math.radians(360 - (self.angle + 210))
        corner4 = math.radians(360 - (self.angle + 330))

        left_top = [
            self.center[0] + math.cos(corner1) * length_x,
            self.center[1] + math.sin(corner1) * length_y
        ]
        right_top = [
            self.center[0] + math.cos(corner2) * length_x,
            self.center[1] + math.sin(corner2) * length_y
        ]
        left_bottom = [
            self.center[0] + math.cos(corner3) * length_x,
            self.center[1] + math.sin(corner3) * length_y
        ]
        right_bottom = [
            self.center[0] + math.cos(corner4) * length_x,
            self.center[1] + math.sin(corner4) * length_y
        ]

        self.corners = [left_top, right_top, left_bottom, right_bottom]

    def check_sensor(self, degree: int, track: pygame.Surface) -> None:
        """Check the distance between the center of the car and the collision surface to create the sensors

        Args:
            degree (int): The degree (angle) of the sensor from the car's center
            track (pygame.Surface): The track on which the car is being drawn
        """

        # Convert degree to radians because math.cos and math.sin use radians
        radians = math.radians(360 - (self.angle + degree))
        cos = math.cos(radians)
        sin = math.sin(radians)
        length = 1

        x, y = int(self.center[0]), int(self.center[1])
        track_x = track.get_width()
        track_y = track.get_height()

        # While the collision surface is not reached, increment the length of the sensor
        while x < track_x and y < track_y and x > 0 and y > 0 and track.get_at((x, y)) != Car.COLLISION_SURFACE_COLOR:
            x = int(self.center[0] + cos * length)
            y = int(self.center[1] + sin * length)

            # If the max length of a sensor is reached, break the loop
            if length > Car.SENSORS_DRAW_DISTANCE:
                break

            length += 1

        # Distance calculation between the center of the car and the collision surface
        distance = int(math.hypot(x - self.center[0], y - self.center[1]))
        self.sensors.append([(x, y), distance])

    def update_adaptive_parameters(self) -> None:
        """Update the adaptive parameters of the car such as the angle increment, the minimum speed, etc."""
        self.angle_increment = math.degrees(math.atan2(self.CAR_SIZE_Y, self.speed * 10))
        
        min_sensor_distance = min(sensor[1] for sensor in self.sensors) if self.sensors else self.CAR_SIZE_X
        self.minimum_speed = max(self.CAR_SIZE_X / 6, min_sensor_distance / 20)

    def update_center(self) -> None:
        """Update the center of the car using its top-left position"""
        self.center = [
            int(self.position[0]) + Car.CAR_SIZE_X / 2,
            int(self.position[1]) + Car.CAR_SIZE_Y / 2
        ]

    def inject_sensors(self, distances: list, hit_points: list) -> None:
        """Inject pre-computed (GPU) sensor data instead of running the per-car raycast loop.

        Args:
            distances (list[float]): List of 5 sensor distances.
            hit_points (list[tuple]): List of 5 (x, y) hit-point tuples.
        """
        self.sensors = [[(int(hx), int(hy)), int(d)] for (hx, hy), d in zip(hit_points, distances)]

    def distance_to_line(self, p, line_start, line_end):
        """Calculate the shortest distance from point p to a line segment."""
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

    def update_physics(self, track: pygame.Surface) -> None:
        """Update car physics (position, collision) WITHOUT computing sensors."""
        old_center = self.center.copy() if hasattr(self, 'center') and self.center else [self.position[0], self.position[1]]
        self.update_center()

        radians = math.radians(360 - self.angle)
        cos = math.cos(radians)
        sin = math.sin(radians)

        self.position[0] += cos * self.speed
        self.position[1] += sin * self.speed
        
        self.update_center()
        new_center = self.center

        if self.alive and self.checkpoints:
            target_cp = self.checkpoints[self.current_checkpoint_index % len(self.checkpoints)]
            p3, p4 = (target_cp[0], target_cp[1]), (target_cp[2], target_cp[3])
            
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
            intersect = ccw(old_center, new_center, p3) != ccw(old_center, new_center, p4) and \
                        ccw(old_center, p3, p4) != ccw(new_center, p3, p4)
                        
            if intersect:
                self.current_checkpoint_index += 1
                self.checkpoint_fitness += 1000.0 + max(0.0, 500.0 - self.frames_since_last_checkpoint)
                self.frames_since_last_checkpoint = 0
                
            self.frames_since_last_checkpoint += 1
            if self.frames_since_last_checkpoint > 300: # Dead if stuck for 5 seconds
                self.alive = False

        self.update_adaptive_parameters()
        self.refresh_corners_positions()
        self.check_collision(track)

    def update_sprite(self, track: pygame.Surface) -> None:
        """Update the sprite of the car and its new informations (position, center, sensors, etc.)"""

        # Update the sprite
        self.update_center()

        # Radians, cos, sin
        radians = math.radians(360 - self.angle)
        cos = math.cos(radians)
        sin = math.sin(radians)

        # Move car to new position
        self.position[0] += cos * self.speed
        self.position[1] += sin * self.speed

        # Update the driven distance with the speed
        self.update_adaptive_parameters()

        # Calculate Corners
        self.refresh_corners_positions()

        # Check collisions
        self.check_collision(track)

        # Clear radars and rewrite them (-90, -45, 0, 45, 90)
        self.sensors.clear()
        for sensor_angle in range(-90, 90 + 1, 45):
            self.check_sensor(sensor_angle, track)

    def get_data(self) -> list[int]:
        """Get the data of the car's sensors

        Returns:
            list[int]: The list of the sensors' distances
        """
        # Get distances to border
        distances = [int(sensor[1]) for sensor in self.sensors]

        # Ensure list has five elements (to correspond to)
        distances += [0] * (5 - len(distances))

        return distances

    def get_reward(self) -> float:
        if not self.checkpoints:
            # Fallback if the user hasn't drawn any checkpoints yet
            distance_reward = self.driven_distance / self.DISTANCE_NORMALIZER
            speed_reward = (self.speed / self.MAX_EXPECTED_SPEED) ** 0.5
            malus = self.speed_penalty / self.penalty_factor
            progress_factor = min(1.0, self.driven_distance / (self.track_diagonal * 0.75))
            return (distance_reward + speed_reward - malus) * (1 + progress_factor)

        # NEW METRIC: Pure Checkpoint Progression
        target_cp = self.checkpoints[self.current_checkpoint_index % len(self.checkpoints)]
        dist_to_next = self.distance_to_line(self.center, (target_cp[0], target_cp[1]), (target_cp[2], target_cp[3]))
        
        # Smooth interpolation: base checkpoint score + how close it is to the NEXT checkpoint
        # We subtract the distance so that moving closer to the checkpoint INCREASES fitness.
        current_fit = self.checkpoint_fitness - dist_to_next
        if current_fit > self.max_fitness_achieved:
            self.max_fitness_achieved = current_fit
            
        return self.max_fitness_achieved

    
    def accelerate(self) -> None:
        """Accelerate the car"""
        self.speed += Car.SPEED_INCREMENT

    def brake(self) -> None:
        """Brake the car"""
        if self.speed > Car.MINIMUM_SPEED:  # We don't want to go backwards nor going too slow
            self.speed -= Car.SPEED_INCREMENT
        else:  
            self.speed = Car.MINIMUM_SPEED
            self.speed_penalty += 1
        
    def turn_left(self) -> None:
        """Turn the car to the left"""
        self.angle += Car.ANGLE_INCREMENT
        
    def turn_right(self) -> None:
        """Turn the car to the right"""
        self.angle -= Car.ANGLE_INCREMENT