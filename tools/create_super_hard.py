import pygame
import math
import numpy as np
import json
import os

pygame.init()
WIDTH = 4000
HEIGHT = 4000
surf = pygame.Surface((WIDTH, HEIGHT))
surf.fill((255, 255, 255))

points = [
    (500, 3500), (3500, 3500), (3500, 3000), (1000, 3000), (1000, 2500),
    (3000, 2500), (3000, 2000), (2000, 2000), (1500, 1500), (1000, 1500),
    (1000, 1000), (2000, 1000), (2000, 2000), (3000, 1500), (3500, 1500),
    (3500, 500), (1500, 500), (1500, 1000), (500, 1500), (500, 2500), (500, 3500)
]

def catmull_rom_spline(P0, P1, P2, P3, num_points=20):
    t = np.linspace(0, 1, num_points)
    t2 = t * t; t3 = t2 * t
    q1 = -t3 + 2.0*t2 - t; q2 = 3.0*t3 - 5.0*t2 + 2.0
    q3 = -3.0*t3 + 4.0*t2 + t; q4 = t3 - t2
    pts = []
    for i in range(num_points):
        x = 0.5 * (P0[0]*q1[i] + P1[0]*q2[i] + P2[0]*q3[i] + P3[0]*q4[i])
        y = 0.5 * (P0[1]*q1[i] + P1[1]*q2[i] + P2[1]*q3[i] + P3[1]*q4[i])
        pts.append((x, y))
    return pts

def generate_spline(points, num_points=20):
    padded = [points[-2]] + points + [points[1], points[2]]
    spline_pts = []
    for i in range(1, len(padded)-3):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], num_points))
    return spline_pts

smooth_points = generate_spline(points, 40)

def draw_thick_line(surface, p1, p2, thickness=80):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    steps = max(int(dist), 1)
    for i in range(steps):
        x = p1[0] + dx * (i / steps)
        y = p1[1] + dy * (i / steps)
        pygame.draw.circle(surface, (0, 0, 0), (int(x), int(y)), thickness)

for i in range(len(smooth_points)-1):
    draw_thick_line(surf, smooth_points[i], smooth_points[i+1])

obstacles = [
    (1500, 3500, 30), (2500, 3500, 30), (2000, 3000, 25), 
    (1500, 2500, 35), (3500, 1000, 40), (1000, 1250, 25),
]
for obs_x, obs_y, obs_r in obstacles:
    pygame.draw.circle(surf, (255, 255, 255), (obs_x, obs_y), obs_r)

os.makedirs("assets/tracks", exist_ok=True)
pygame.image.save(surf, "assets/tracks/super_hard.png")

checkpoints = []
accumulated_dist = 0.0
checkpoint_spacing = 150.0  

INTERSECTIONS = [(2000, 2000), (1500, 1000)]
DANGER_ZONE = 250

for i in range(1, len(smooth_points) - 1):
    p0 = smooth_points[i-1]
    p1 = smooth_points[i]
    p2 = smooth_points[i+1]
    
    segment_dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    accumulated_dist += segment_dist
    
    if accumulated_dist >= checkpoint_spacing:
        is_intersection = False
        for inter in INTERSECTIONS:
            if math.hypot(p1[0] - inter[0], p1[1] - inter[1]) < DANGER_ZONE:
                is_intersection = True
                break
                
        if is_intersection:
            continue
            
        accumulated_dist = 0.0
        
        dx = p2[0] - p0[0]; dy = p2[1] - p0[1]
        length = math.hypot(dx, dy)
        if length == 0: continue
        dx /= length; dy /= length
        
        nx = -dy; ny = dx
        width = 120
        cx1 = p1[0] + nx * width; cy1 = p1[1] + ny * width
        cx2 = p1[0] - nx * width; cy2 = p1[1] - ny * width
        checkpoints.append([int(cx1), int(cy1), int(cx2), int(cy2)])

os.makedirs("data/tracks", exist_ok=True)
with open("data/tracks/super_hard.json", "w") as f:
    json.dump(checkpoints, f, indent=4)
