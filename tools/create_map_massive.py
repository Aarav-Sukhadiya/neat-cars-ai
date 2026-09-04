import pygame
import math
import numpy as np

pygame.init()
WIDTH = 4000
HEIGHT = 4000
surf = pygame.Surface((WIDTH, HEIGHT))
surf.fill((255, 255, 255)) # White is wall

# Let's build a huge track! Start at 200, 950
points = [
    (200, 950),     # START (faces right)
    (600, 950),     # Short straight
    (1000, 800),    # Curve up-right
    (1000, 400),    # Up
    (1400, 200),    # Curve right
    (2200, 200),    # Top straight
    (2600, 600),    # Curve down-right
    (2800, 1200),   # Down
    (2400, 1600),   # S-curve left
    (2600, 2000),   # S-curve right
    (2400, 2400),   # S-curve left
    (2800, 2800),   # Curve right-down
    (3200, 3200),   # Bottom right corner
    (2000, 3400),   # Bottom straight
    (1000, 3400),   # Bottom straight
    (400, 3000),    # Curve up-left
    (200, 2000),    # Left straight up
    (400, 1400),    # Chicane right
    (200, 1200),    # Chicane left
    (200, 950)      # Back to start
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
    padded = [(50, 950)] + points + [points[1]]
    spline_pts = []
    for i in range(1, len(padded)-2):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], num_points))
    return spline_pts

smooth_points = generate_spline(points, 40)

def draw_thick_line(surface, p1, p2, thickness=85): # Kept same road width
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

# ADD OBSTACLES (White circles / polygons on the black track)
obstacles = [
    (1800, 200, 25),   # Top straight block
    (2500, 1800, 30),  # Middle of S-curves
    (2000, 3400, 40),  # Bottom straight block
    (300, 1700, 20),   # Left straight block
]
for obs_x, obs_y, obs_r in obstacles:
    pygame.draw.circle(surf, (255, 255, 255), (obs_x, obs_y), obs_r)

pygame.image.save(surf, "assets/track.png")
print("Massive complex track generated successfully!")

import json
# Generate Checkpoints automatically along the spine
checkpoints = []
# Calculate evenly spaced checkpoints based on actual distance
accumulated_dist = 0.0
checkpoint_spacing = 150.0  # Place a checkpoint exactly every 150 pixels

for i in range(1, len(smooth_points) - 1):
    p0 = smooth_points[i-1]
    p1 = smooth_points[i]
    p2 = smooth_points[i+1]
    
    segment_dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    accumulated_dist += segment_dist
    
    if accumulated_dist >= checkpoint_spacing:
        accumulated_dist = 0.0
        
        # Calculate direction vector using p0 and p2 for a smoother tangent
        dx = p2[0] - p0[0]
        dy = p2[1] - p0[1]
        
        length = math.hypot(dx, dy)
        if length == 0: continue
        dx /= length
        dy /= length
        
        # Perpendicular vector
        nx = -dy
        ny = dx
        
        width = 120
        cx1 = p1[0] + nx * width
        cy1 = p1[1] + ny * width
        cx2 = p1[0] - nx * width
        cy2 = p1[1] - ny * width
        
        checkpoints.append([int(cx1), int(cy1), int(cx2), int(cy2)])

with open("data/checkpoints.json", "w") as f:
    json.dump(checkpoints, f, indent=4)
print("Checkpoints generated and saved successfully!")
