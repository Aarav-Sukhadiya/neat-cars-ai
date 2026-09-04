import pygame
import json
import math
import sys
import numpy as np
import os

if len(sys.argv) < 2:
    print("Usage: python auto_import.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
pygame.init()
try:
    base_img = pygame.image.load(img_path).convert_alpha()
except Exception as e:
    print(f"Failed to load image: {e}")
    sys.exit(1)

W, H = base_img.get_size()
screen = pygame.display.set_mode((min(1280, W), min(720, H)))
pygame.display.set_caption("Auto Checkpoint Generator")
font = pygame.font.SysFont("monospace", 14)

# Binarize image (Black/Dark = track, Light = wall)
# We assume track pixels have RGB average < 128
track_mask = np.zeros((W, H), dtype=bool)
for y in range(H):
    for x in range(W):
        r, g, b, a = base_img.get_at((x, y))
        if (r + g + b) / 3 < 128 and a > 0:
            track_mask[x, y] = True

def snap_to_center(x, y, max_radius=300):
    """Casts rays to find the geometric center of the track at the given point."""
    if not track_mask[int(x), int(y)]:
        return x, y
    
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    distances = []
    
    for angle in angles:
        dx, dy = math.cos(angle), math.sin(angle)
        dist = 0
        while dist < max_radius:
            nx, ny = int(x + dx * dist), int(y + dy * dist)
            if nx < 0 or nx >= W or ny < 0 or ny >= H or not track_mask[nx, ny]:
                break
            dist += 1
        distances.append((angle, dist))
        
    # Find opposite pairs and move point to the midpoint of the longest span
    best_span = 0
    best_center = (x, y)
    for i in range(8):
        a1, d1 = distances[i]
        a2, d2 = distances[i+8]
        span = d1 + d2
        if span > best_span:
            best_span = span
            # Midpoint along this axis
            move = (d1 - d2) / 2
            best_center = (x + math.cos(a1) * move, y + math.sin(a1) * move)
            
    return best_center

nodes = []
camera_x, camera_y = 0, 0
zoom = 1.0

def generate_checkpoints(nodes):
    if len(nodes) < 2: return []
    
    # 1. Spline interpolation for smoothness
    def catmull_rom(P0, P1, P2, P3, steps=10):
        pts = []
        for t in np.linspace(0, 1, steps):
            t2, t3 = t*t, t*t*t
            q1 = -t3 + 2.0*t2 - t
            q2 = 3.0*t3 - 5.0*t2 + 2.0
            q3 = -3.0*t3 + 4.0*t2 + t
            q4 = t3 - t2
            pts.append((0.5 * (P0[0]*q1 + P1[0]*q2 + P2[0]*q3 + P3[0]*q4),
                        0.5 * (P0[1]*q1 + P1[1]*q2 + P2[1]*q3 + P3[1]*q4)))
        return pts

    spline = []
    padded = [nodes[0]] + nodes + [nodes[-1], nodes[-1]]
    for i in range(1, len(padded)-2):
        spline.extend(catmull_rom(padded[i-1], padded[i], padded[i+1], padded[i+2], 10))

    # 2. Place checkpoints by walking the spline
    checkpoints = []
    acc = 0.0
    spacing = 100.0
    
    # Track width varies, so we cast rays to find edges for the checkpoint
    for i in range(1, len(spline)-1):
        p0, p1, p2 = spline[i-1], spline[i], spline[i+1]
        dist = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        acc += dist
        if acc >= spacing:
            acc = 0.0
            dx, dy = p2[0]-p0[0], p2[1]-p0[1]
            L = math.hypot(dx, dy)
            if L == 0: continue
            nx, ny = -dy/L, dx/L
            
            # Raycast left and right to find walls
            w_left = 0
            while w_left < 800:
                cx, cy = int(p1[0] + nx * w_left), int(p1[1] + ny * w_left)
                if cx < 0 or cx >= W or cy < 0 or cy >= H or not track_mask[cx, cy]: break
                w_left += 1
                
            w_right = 0
            while w_right < 800:
                cx, cy = int(p1[0] - nx * w_right), int(p1[1] - ny * w_right)
                if cx < 0 or cx >= W or cy < 0 or cy >= H or not track_mask[cx, cy]: break
                w_right += 1
                
            left_pt = (p1[0] + nx * w_left, p1[1] + ny * w_left)
            right_pt = (p1[0] - nx * w_right, p1[1] - ny * w_right)
            checkpoints.append([int(left_pt[0]), int(left_pt[1]), int(right_pt[0]), int(right_pt[1])])
            
    return checkpoints

running = True
panning = False
last_mouse = (0,0)
checkpoints = []

while running:
    mx, my = pygame.mouse.get_pos()
    wx = (mx / zoom) + camera_x
    wy = (my / zoom) + camera_y

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Snap click to geometric center of track
                nx, ny = snap_to_center(wx, wy)
                nodes.append((nx, ny))
                checkpoints = generate_checkpoints(nodes)
            elif event.button == 3:
                panning = True
                last_mouse = (mx, my)
            elif event.button == 4:
                zoom = min(3.0, zoom + 0.1)
            elif event.button == 5:
                zoom = max(0.1, zoom - 0.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                panning = False
        elif event.type == pygame.MOUSEMOTION:
            if panning:
                camera_x -= (mx - last_mouse[0]) / zoom
                camera_y -= (my - last_mouse[1]) / zoom
                last_mouse = (mx, my)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and nodes:
                nodes.pop()
                checkpoints = generate_checkpoints(nodes)
            elif event.key == pygame.K_s and len(nodes) >= 2:
                # Save checkpoints
                with open("data/tracks/imported.json", "w") as f:
                    json.dump(checkpoints, f, indent=4)
                
                # Copy image
                pygame.image.save(base_img, "assets/tracks/imported.png")
                
                # Update config
                cfg_path = "config/track_config.json"
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r") as f: cfg = json.load(f)
                    cfg["tracks"]["imported"] = {
                        "name": "Imported Track",
                        "image": "assets/tracks/imported.png",
                        "checkpoints": "data/tracks/imported.json",
                        "max_frames": 7200,
                        "start_pos": [int(nodes[0][0]), int(nodes[0][1])],
                        "start_angle": int(math.degrees(math.atan2(-(nodes[1][1]-nodes[0][1]), nodes[1][0]-nodes[0][0]))) if len(nodes)>1 else 0
                    }
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=4)
                print("Track imported successfully! Run with: python main.py --track 4")
                sys.exit(0)

    # Render
    screen.fill((20, 20, 20))
    
    # Blit background image
    scaled_w, scaled_h = int(W * zoom), int(H * zoom)
    sub_rect = pygame.Rect(max(0, int(camera_x)), max(0, int(camera_y)), 
                           min(W - max(0, int(camera_x)), int(screen.get_width() / zoom)), 
                           min(H - max(0, int(camera_y)), int(screen.get_height() / zoom)))
    
    if sub_rect.width > 0 and sub_rect.height > 0:
        sub_img = base_img.subsurface(sub_rect)
        scaled_sub = pygame.transform.scale(sub_img, (int(sub_rect.width * zoom), int(sub_rect.height * zoom)))
        screen.blit(scaled_sub, (max(0, -camera_x * zoom), max(0, -camera_y * zoom)))

    def w2s(x, y): return int((x - camera_x) * zoom), int((y - camera_y) * zoom)
    
    # Draw Nodes
    for i, node in enumerate(nodes):
        sx, sy = w2s(*node)
        pygame.draw.circle(screen, (0, 0, 255), (sx, sy), max(2, int(8*zoom)))
        if i > 0:
            psx, psy = w2s(*nodes[i-1])
            pygame.draw.line(screen, (200, 200, 255), (psx, psy), (sx, sy), 2)
            
    # Draw Generated Checkpoints (Numbered!)
    for i, cp in enumerate(checkpoints):
        p1, p2 = w2s(cp[0], cp[1]), w2s(cp[2], cp[3])
        pygame.draw.line(screen, (0, 255, 0), p1, p2, 2)
        # Draw number in center
        cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 12)
        txt = font.render(str(i+1), True, (255, 255, 255))
        screen.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2))

    # UI Instructions
    instructions = [
        "AUTO CHECKPOINT GENERATOR",
        "1. Click roughly along the track (it auto-snaps to exact center)",
        "2. The checkpoints are auto-generated with numbers",
        "3. Press 'S' to save and import",
        "Scroll: Zoom, Right Click: Pan, Ctrl+Z: Undo"
    ]
    for i, text in enumerate(instructions):
        txt = font.render(text, True, (0, 0, 0))
        screen.blit(txt, (12, 12 + i * 20))
        txt = font.render(text, True, (255, 255, 255))
        screen.blit(txt, (10, 10 + i * 20))

    pygame.display.flip()
