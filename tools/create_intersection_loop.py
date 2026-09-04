import pygame
import math
import numpy as np
import json
import os

pygame.init()
WIDTH = 4000
HEIGHT = 4000
surf = pygame.Surface((WIDTH, HEIGHT))
surf.fill((255, 255, 255)) # White is wall

# Figure-8 with a perfect 90-degree intersection at (2000, 2000)
points = [
    (2000, 3000),   # Start (facing UP)
    (2000, 2500),
    (2000, 2000),   # CROSS UP
    (2000, 1500),
    (2000, 1000),
    (2500, 500),
    (3000, 1000),
    (3000, 2000),
    (2500, 2000),
    (2000, 2000),   # CROSS LEFT
    (1500, 2000),
    (1000, 2000),
    (1000, 3000),
    (1500, 3500),
    (2000, 3000)    # End
]

def catmull_rom_spline(P0, P1, P2, P3, num_points=20):
    t = np.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t
    
    q1 = -t3 + 2.0*t2 - t
    q2 = 3.0*t3 - 5.0*t2 + 2.0
    q3 = -3.0*t3 + 4.0*t2 + t
    q4 = t3 - t2

    pts = []
    for i in range(num_points):
        x = 0.5 * (P0[0]*q1[i] + P1[0]*q2[i] + P2[0]*q3[i] + P3[0]*q4[i])
        y = 0.5 * (P0[1]*q1[i] + P1[1]*q2[i] + P2[1]*q3[i] + P3[1]*q4[i])
        pts.append((x, y))
    return pts

def generate_spline(points, num_points=20):
    # Pad to make it loop smoothly
    padded = [points[-2]] + points + [points[1], points[2]]
    spline_pts = []
    for i in range(1, len(padded)-2):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], num_points))
    return spline_pts

smooth_points = generate_spline(points, 40)

def draw_thick_line(surface, p1, p2, thickness=85):
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

os.makedirs("assets/tracks", exist_ok=True)
pygame.image.save(surf, "assets/tracks/intersection_loop.png")
print("Intersection Loop track generated successfully!")

# --- GENERATE CHECKPOINTS ---
checkpoints = []
accumulated_dist = 0.0
checkpoint_spacing = 150.0  
INTERSECTION = (2000, 2000)
DANGER_ZONE = 300  # Do not place any checkpoints within 300 pixels of the intersection

for i in range(1, len(smooth_points) - 1):
    p0 = smooth_points[i-1]
    p1 = smooth_points[i]
    p2 = smooth_points[i+1]
    
    segment_dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    accumulated_dist += segment_dist
    
    if accumulated_dist >= checkpoint_spacing:
        dist_to_intersection = math.hypot(p1[0] - INTERSECTION[0], p1[1] - INTERSECTION[1])
        
        if dist_to_intersection < DANGER_ZONE:
            # Skip this checkpoint. Do NOT reset accumulated_dist.
            # This ensures a checkpoint is dropped immediately as soon as we leave the danger zone!
            # (which places it at the starting curve of the loop, exactly as requested).
            continue
            
        accumulated_dist = 0.0
        
        dx = p2[0] - p0[0]
        dy = p2[1] - p0[1]
        
        length = math.hypot(dx, dy)
        if length == 0: continue
        dx /= length
        dy /= length
        
        nx = -dy
        ny = dx
        
        width = 120
        cx1 = p1[0] + nx * width
        cy1 = p1[1] + ny * width
        cx2 = p1[0] - nx * width
        cy2 = p1[1] - ny * width
        
        checkpoints.append([int(cx1), int(cy1), int(cx2), int(cy2)])

os.makedirs("data/tracks", exist_ok=True)
with open("data/tracks/intersection_loop.json", "w") as f:
    json.dump(checkpoints, f, indent=4)
print(f"Generated {len(checkpoints)} checkpoints safely outside the intersection!")
