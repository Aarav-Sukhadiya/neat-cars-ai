import numpy as np
import math
from src.render.car import Car

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
        
        # Checkpoint tracking
        self.has_checkpoints = len(checkpoints) > 0
        self.max_fitness = np.zeros(self.num_cars, dtype=np.float32)
        if self.has_checkpoints:
            self.checkpoints = np.array(checkpoints, dtype=np.float32) # (N, 4)
            self.cp_index = np.zeros(self.num_cars, dtype=np.int32)
            self.cp_fitness = np.zeros(self.num_cars, dtype=np.float32)
            self.frames_since_cp = np.zeros(self.num_cars, dtype=np.int32)
            self.max_fitness = np.zeros(self.num_cars, dtype=np.float32)
            
    def ccw(self, A_x, A_y, B_x, B_y, C_x, C_y):
        return (C_y - A_y) * (B_x - A_x) > (B_y - A_y) * (C_x - A_x)
        
    def step(self, choices, track_pixels):
        alive_mask = self.alive
        if not np.any(alive_mask):
            return
            
        self.frames_alive[alive_mask] += 1
        
        # Apply Actions (choices: 0=Left, 1=Right, 2=Accel, 3=Brake)
        left = (choices == 0) & alive_mask
        right = (choices == 1) & alive_mask
        accel = (choices == 2) & alive_mask
        brake = (choices == 3) & alive_mask
        
        self.angle[left] += Car.ANGLE_INCREMENT
        self.angle[right] -= Car.ANGLE_INCREMENT
        
        self.speed[accel] += Car.SPEED_INCREMENT
        self.speed[brake] = np.maximum(Car.MINIMUM_SPEED, self.speed[brake] - Car.SPEED_INCREMENT)
        
        # Brake penalty
        self.speed_penalty[(choices == 3) & alive_mask & (self.speed <= Car.MINIMUM_SPEED)] += 1
        
        # Physics update
        radians = np.radians(360 - self.angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        
        old_cx = self.cx.copy()
        old_cy = self.cy.copy()
        
        self.x[alive_mask] += cos_a[alive_mask] * self.speed[alive_mask]
        self.y[alive_mask] += sin_a[alive_mask] * self.speed[alive_mask]
        self.driven_distance[alive_mask] += self.speed[alive_mask]
        
        self.cx = self.x + Car.CAR_SIZE_X / 2
        self.cy = self.y + Car.CAR_SIZE_Y / 2
        
        # Collisions (4 corners)
        # Car corners relative to center
        # length = hypot(20/2, 15/2) = 12.5
        # angles = atan2(+/-7.5, +/-10)
        # To rotate: cx + dx*cos - dy*sin
        L2, W2 = Car.CAR_SIZE_X / 2, Car.CAR_SIZE_Y / 2
        corners_dx = np.array([-L2, L2, L2, -L2], dtype=np.float32)
        corners_dy = np.array([-W2, -W2, W2, W2], dtype=np.float32)
        
        c_cos = np.cos(radians)
        c_sin = np.sin(radians)
        
        hit_wall = np.zeros(self.num_cars, dtype=bool)
        for i in range(4):
            corner_x = self.cx + corners_dx[i] * c_cos - corners_dy[i] * c_sin
            corner_y = self.cy + corners_dx[i] * c_sin + corners_dy[i] * c_cos
            
            cx_int = np.clip(corner_x.astype(np.int32), 0, self.track_width - 1)
            cy_int = np.clip(corner_y.astype(np.int32), 0, self.track_height - 1)
            
            # track_pixels is (H, W) array where 255 is wall
            hits = track_pixels[cy_int, cx_int] == 255
            hit_wall |= hits
            
        self.alive[alive_mask & hit_wall] = False
        
        # Checkpoints
        if self.has_checkpoints:
            active_cps = self.checkpoints[self.cp_index % len(self.checkpoints)] # (N, 4)
            p3_x, p3_y = active_cps[:, 0], active_cps[:, 1]
            p4_x, p4_y = active_cps[:, 2], active_cps[:, 3]
            
            ccw1 = self.ccw(old_cx, old_cy, self.cx, self.cy, p3_x, p3_y)
            ccw2 = self.ccw(old_cx, old_cy, self.cx, self.cy, p4_x, p4_y)
            ccw3 = self.ccw(old_cx, old_cy, p3_x, p3_y, p4_x, p4_y)
            ccw4 = self.ccw(self.cx, self.cy, p3_x, p3_y, p4_x, p4_y)
            
            intersect = (ccw1 != ccw2) & (ccw3 != ccw4) & alive_mask
            
            self.cp_index[intersect] += 1
            self.cp_fitness[intersect] += 1000.0 + np.maximum(0.0, 500.0 - self.frames_since_cp[intersect])
            self.frames_since_cp[intersect] = 0
            
            self.frames_since_cp[alive_mask] += 1
            starved = self.frames_since_cp > 300
            self.alive[alive_mask & starved] = False
            
    def update_fitness(self):
        if not self.has_checkpoints:
            for i, c in enumerate(self.cars):
                if not self.alive[i]: continue
                # Unused fallback in vectorized for now
                pass
            return
            
        alive_mask = self.alive
        if not np.any(alive_mask):
            return
            
        active_cps = self.checkpoints[self.cp_index % len(self.checkpoints)]
        
        px, py = self.cx, self.cy
        sx, sy = active_cps[:, 0], active_cps[:, 1]
        ex, ey = active_cps[:, 2], active_cps[:, 3]
        
        line_mag_sq = (ex - sx)**2 + (ey - sy)**2
        u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / np.maximum(1e-5, line_mag_sq)
        u = np.clip(u, 0, 1)
        ix = sx + u * (ex - sx)
        iy = sy + u * (ey - sy)
        
        dist_to_next = np.hypot(px - ix, py - iy)
        current_fit = self.cp_fitness - dist_to_next
        
        improved = (current_fit > self.max_fitness) & alive_mask
        self.max_fitness[improved] = current_fit[improved]
        
    def sync_to_cars(self, alive_indices):
        """Only sync the cars that need to be drawn or accessed"""
        for i in alive_indices:
            car = self.cars[i]
            car.position[0] = float(self.x[i])
            car.position[1] = float(self.y[i])
            car.angle = float(self.angle[i])
            car.speed = float(self.speed[i])
            car.alive = self.alive[i]
            car.center = [float(self.cx[i]), float(self.cy[i])]
            car.max_fitness_achieved = float(self.max_fitness[i])
            car.refresh_corners_positions()
