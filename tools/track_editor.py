import pygame
import math
import numpy as np
import json
import sys
import json
import os

pygame.init()
SCREEN_W, SCREEN_H = 1280, 720
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Track Editor - Draw Super Large Maps")
font = pygame.font.SysFont("monospace", 16)

# Map Settings
MAP_W, MAP_H = 8000, 8000
nodes = []
road_width = 85
camera_x, camera_y = MAP_W // 2 - SCREEN_W // 2, MAP_H // 2 - SCREEN_H // 2
zoom = 1.0

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

def get_spline_points(closed=False):
    if len(nodes) < 2: return []
    pts = list(nodes)
    if closed:
        padded = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        padded = [pts[0]] + pts + [pts[-1], pts[-1]]
        
    spline_pts = []
    end_idx = len(padded)-3 if closed else len(padded)-2
    for i in range(1, end_idx):
        spline_pts.extend(catmull_rom_spline(padded[i-1], padded[i], padded[i+1], padded[i+2], 20))
    return spline_pts

def draw_thick_line(surface, p1, p2, thickness, color=(0,0,0)):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    steps = max(int(dist), 1)
    for i in range(steps):
        x = p1[0] + dx * (i / steps)
        y = p1[1] + dy * (i / steps)
        pygame.draw.circle(surface, color, (int(x), int(y)), thickness)

def save_track(closed):
    print("Saving track...")
    surf = pygame.Surface((MAP_W, MAP_H))
    surf.fill((255, 255, 255))
    
    spline = get_spline_points(closed)
    for i in range(len(spline)-1):
        draw_thick_line(surf, spline[i], spline[i+1], road_width)
        
    pygame.image.save(surf, "assets/tracks/custom.png")
    
    checkpoints = []
    accumulated = 0.0
    for i in range(1, len(spline) - 1):
        p0, p1, p2 = spline[i-1], spline[i], spline[i+1]
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        accumulated += dist
        if accumulated >= 150.0:
            accumulated = 0.0
            dx, dy = p2[0] - p0[0], p2[1] - p0[1]
            L = math.hypot(dx, dy)
            if L == 0: continue
            nx, ny = -dy/L, dx/L
            w = road_width + 30
            checkpoints.append([
                int(p1[0] + nx * w), int(p1[1] + ny * w),
                int(p1[0] - nx * w), int(p1[1] - ny * w)
            ])
            
    with open("data/tracks/custom.json", "w") as f:
        json.dump(checkpoints, f, indent=4)
        
    # Update config automatically
    config_path = "config/track_config.json"
    if os.path.exists(config_path) and nodes:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if "custom" in cfg["tracks"]:
            cfg["tracks"]["custom"]["start_pos"] = [int(nodes[0][0]), int(nodes[0][1])]
            # Guess starting angle based on first 2 nodes
            if len(nodes) > 1:
                dx = nodes[1][0] - nodes[0][0]
                dy = nodes[1][1] - nodes[0][1]
                angle = math.degrees(math.atan2(-dy, dx))
                cfg["tracks"]["custom"]["start_angle"] = int(angle)
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=4)
                
    print("Saved custom.png and custom.json! You can now run: python main.py --track 3")

running = True
closed = False
panning = False
last_mouse = (0, 0)
clock = pygame.time.Clock()

while running:
    mx, my = pygame.mouse.get_pos()
    world_x = (mx / zoom) + camera_x
    world_y = (my / zoom) + camera_y
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                nodes.append((world_x, world_y))
            elif event.button == 3:
                panning = True
                last_mouse = (mx, my)
            elif event.button == 4:
                zoom = min(2.0, zoom + 0.1)
            elif event.button == 5:
                zoom = max(0.1, zoom - 0.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                panning = False
        elif event.type == pygame.MOUSEMOTION:
            if panning:
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                camera_x -= dx / zoom
                camera_y -= dy / zoom
                last_mouse = (mx, my)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                if nodes: nodes.pop()
            elif event.key == pygame.K_c:
                closed = not closed
            elif event.key == pygame.K_s:
                save_track(closed)
            elif event.key == pygame.K_EQUALS:
                road_width += 5
            elif event.key == pygame.K_MINUS:
                road_width = max(10, road_width - 5)

    screen.fill((40, 40, 40))
    
    def w2s(wx, wy):
        return int((wx - camera_x) * zoom), int((wy - camera_y) * zoom)
    
    # Draw Grid
    grid_size = int(1000 * zoom)
    if grid_size > 10:
        for x in range(0, int(MAP_W * zoom), grid_size):
            pygame.draw.line(screen, (60, 60, 60), (x - camera_x*zoom, 0), (x - camera_x*zoom, SCREEN_H))
        for y in range(0, int(MAP_H * zoom), grid_size):
            pygame.draw.line(screen, (60, 60, 60), (0, y - camera_y*zoom), (SCREEN_W, y - camera_y*zoom))
    
    # Draw Spline
    spline = get_spline_points(closed)
    if len(spline) > 1:
        for i in range(len(spline)-1):
            p1 = w2s(*spline[i])
            p2 = w2s(*spline[i+1])
            pygame.draw.line(screen, (100, 100, 100), p1, p2, max(1, int(road_width * 2 * zoom)))
            pygame.draw.line(screen, (255, 255, 0), p1, p2, max(1, int(2 * zoom)))
            
    # Draw Nodes
    for i, node in enumerate(nodes):
        sx, sy = w2s(*node)
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        pygame.draw.circle(screen, color, (sx, sy), max(4, int(10 * zoom)))
        if i > 0:
            psx, psy = w2s(*nodes[i-1])
            pygame.draw.line(screen, (255, 100, 100), (psx, psy), (sx, sy), 1)
            
    if closed and len(nodes) > 1:
        psx, psy = w2s(*nodes[-1])
        sx, sy = w2s(*nodes[0])
        pygame.draw.line(screen, (255, 100, 100), (psx, psy), (sx, sy), 1)

    # UI
    info = [
        "TRACK EDITOR (8000x8000)",
        "Left Click: Place Node",
        "Right Click + Drag: Pan Camera",
        "Scroll Wheel: Zoom in/out",
        f"Zoom: {zoom:.1f}x",
        "Ctrl+Z: Undo Node",
        f"C: Toggle Closed Loop ({'ON' if closed else 'OFF'})",
        f"+/-: Road Width ({road_width})",
        "S: Save Track to ID 99"
    ]
    for i, text in enumerate(info):
        s = font.render(text, True, (255, 255, 255))
        screen.blit(s, (10, 10 + i * 25))

    pygame.display.flip()
    clock.tick(60)
