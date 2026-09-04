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

# Super difficult track geometry
points = [
    (500, 3500),    # Start
    (3500, 3500),   # Long straight Right
    (3500, 3000),   # Hairpin turn Up
    (1000, 3000),   # Straight Left
    (1000, 2500),   # Hairpin turn Up
    (3000, 2500),   # Straight Right
    (3000, 2000),   # Curve Up
    (2000, 2000),   # Sharp Left
    (1500, 1500),   # Diagonal Up-Left (Entering Loop 1)
    (1000, 1500),   # Loop Left
    (1000, 1000),   # Loop Up
    (2000, 1000),   # Loop Right
    (2000, 2000),   # INTERSECTION 1 (Crossing previous point 7)
    (3000, 1500),   # Diagonal Down-Right
    (3500, 1500),   # Curve Right
    (3500, 500),    # Long Straight Up
    (1500, 500),    # Straight Left
    (1500, 1000),   # INTERSECTION 2 (Crossing Loop 1 at point 11)
    (500, 1500),    # Diagonal Down-Left
    (500, 2500),    # Straight Down
    (500, 3500)     # Back to Start
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
    padded = [points[-2]] + points + [points[1], points[2]]
    spline_pts = []
    for i in range(1, len(padded)-2):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], num_points))
    return spline_pts

smooth_points = generate_spline(points, 40)

def draw_thick_line(surface, p1, p2, thickness=80): # Narrower track = harder!
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

# ADD OBSTACLES (White circles that block parts of the road)
obstacles = [
    (1500, 3500, 30), # On the start straight
    (2500, 3500, 30), # On the start straight
    (2000, 3000, 25), # Hairpin straight 1
    (1500, 2500, 35), # Hairpin straight 2
    (3500, 1000, 40), # Blocking the far right vertical straight
    (1000, 1250, 25), # Inside the loop
]
for obs_x, obs_y, obs_r in obstacles:
    pygame.draw.circle(surf, (255, 255, 255), (obs_x, obs_y), obs_r)

os.makedirs("assets/tracks", exist_ok=True)
pygame.image.save(surf, "assets/tracks/super_hard.png")
print("Super Hard track generated successfully!")

# --- SMART CHECKPOINT GENERATOR ---
checkpoints = []
accumulated_dist = 0.0
checkpoint_spacing = 150.0  

# Automatically detect intersections by comparing spline points against each other
# A point is near an intersection if it is close to another point that is far away in spline index
DANGER_ZONE = 250

for i in range(1, len(smooth_points) - 1):
    p0 = smooth_points[i-1]
    p1 = smooth_points[i]
    p2 = smooth_points[i+1]
    
    segment_dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    accumulated_dist += segment_dist
    
    if accumulated_dist >= checkpoint_spacing:
        # Check if we are inside a Danger Zone (Intersection)
        is_intersection = False
        for j in range(len(smooth_points)):
            # If the index is far away (not just the adjacent segment)
            if abs(i - j) > 80:
                dist_to_other = math.hypot(p1[0] - smooth_points[j][0], p1[1] - smooth_points[j][1])
                if dist_to_other < DANGER_ZONE:
                    is_intersection = True
                    break
                    
        if is_intersection:
            # Skip checkpoint, wait until we exit the intersection zone
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
with open("data/tracks/super_hard.json", "w") as f:
    json.dump(checkpoints, f, indent=4)
print(f"Generated {len(checkpoints)} checkpoints safely outside intersections!")
