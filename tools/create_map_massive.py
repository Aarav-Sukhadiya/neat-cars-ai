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
    (200, 950),     # START
    (400, 950),     # Long straight acceleration
    (1800, 950),    # Super long straight
    (2200, 850),    # Smooth right turn
    (2200, 400),    # Upwards straight
    (1800, 200),    # Left hairpin entry
    (1600, 400),    # Left hairpin exit
    (1600, 1000),   # Downwards straight
    (1400, 1300),   # Chicane right
    (1200, 1200),   # Chicane left
    (1000, 1500),   # Drop down
    (1200, 2000),   # Deep south loop
    (2000, 2400),   # Wide sweeper
    (3000, 2400),   # Long bottom straight
    (3500, 2200),   # Hard up turn
    (3600, 1000),   # Massive upward sprint (high speed!)
    (3400, 500),    # Hook left
    (2800, 300),    # Hook left again
    (2400, 500),    # Snaking S curve start
    (2600, 800),    
    (2400, 1100),
    (2600, 1400),
    (2400, 1700),   # S curve end
    (1800, 1700),   # Cut across
    (1000, 2200),
    (500, 2500),    # Far left hook
    (200, 2000),    # Go up
    (300, 1500),
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
    (1000, 950, 20),   # First straight minor bump
    (1600, 950, 30),   # Another block
    (1400, 1250, 25),  # In the chicane
    (3200, 2400, 40),  # Big block on bottom straight
    (3600, 1600, 30),  # Right side sprint block
    (2500, 1100, 25),  # Middle of S-curve
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
