import pygame
import math
import numpy as np

pygame.init()
surf = pygame.Surface((1500, 1080))
surf.fill((255, 255, 255)) # White is wall

# Complex Track Waypoints
points = [
    (200, 950),   # Start (bottom left straight)
    (1300, 950),  # End of bottom straight
    (1400, 850),  # Right Sweeper Entry
    (1300, 750),  # Right Sweeper Exit
    (800, 750),   # Back straight mid
    (600, 700),   # Chicane entry
    (500, 600),   # Chicane mid
    (600, 500),   # Chicane exit
    (1100, 500),  # Top-ish right
    (1350, 350),  # Hairpin entry
    (1100, 150),  # Hairpin exit
    (500, 150),   # Long top straight
    (200, 300),   # Left sweeping curve
    (150, 500),   # Left sweeping curve 2
    (300, 650),   # S-curve entry
    (150, 800),   # S-curve exit
    (200, 950)    # Back to start
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
    # Pad points to wrap around for closed loop
    padded = [points[-2]] + points + [points[1]]
    spline_pts = []
    for i in range(1, len(padded)-2):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], num_points))
    return spline_pts

smooth_points = generate_spline(points, 30)

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

pygame.image.save(surf, "assets/track.png")
print("Complex track generated successfully!")
