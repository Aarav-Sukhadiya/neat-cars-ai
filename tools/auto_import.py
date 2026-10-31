"""Semi-automatic checkpoint generator for an existing track image.

    python tools/auto_import.py assets/tracks/<name>.png

Click roughly along the road; each click snaps to the road's centre. A Catmull-Rom
spline through the clicks is walked and a checkpoint is placed every SPACING px,
spanning wall-to-wall (found by raycasting).

Controls: left click = add node, Z = undo, scroll = zoom, right-drag = pan, S = save.
Saving writes data/tracks/<name>.json (and copies the image to assets/tracks/ if it
lives elsewhere) and registers <name> in config/track_config.json.
"""

import json
import math
import os
import shutil
import sys

import numpy as np
import pygame

SPACING = 100.0          # px between generated checkpoints
MAX_HALF_WIDTH = 800     # max raycast distance to each wall
CONFIG_PATH = "config/track_config.json"


def build_track_mask(surface: pygame.Surface) -> np.ndarray:
    """(W, H) bool array: True where the pixel is road (dark and not transparent)."""
    rgb = pygame.surfarray.array3d(surface).astype(np.uint16)       # (W, H, 3)
    dark = rgb.sum(axis=2) < 3 * 128                                # mean < 128 (same as (r+g+b)/3 < 128)
    if surface.get_flags() & pygame.SRCALPHA:
        dark &= pygame.surfarray.array_alpha(surface) > 0
    return dark


def snap_to_center(track_mask, x, y, max_radius=300):
    """Move (x, y) to the midpoint of the widest road span through it (16 rays)."""
    W, H = track_mask.shape
    if not (0 <= int(x) < W and 0 <= int(y) < H) or not track_mask[int(x), int(y)]:
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

    best_span = 0
    best_center = (x, y)
    for i in range(8):   # opposite ray pairs
        a1, d1 = distances[i]
        _, d2 = distances[i + 8]
        if d1 + d2 > best_span:
            best_span = d1 + d2
            move = (d1 - d2) / 2
            best_center = (x + math.cos(a1) * move, y + math.sin(a1) * move)
    return best_center


def catmull_rom(P0, P1, P2, P3, steps=10):
    pts = []
    for t in np.linspace(0, 1, steps):
        t2, t3 = t * t, t * t * t
        q1 = -t3 + 2.0 * t2 - t
        q2 = 3.0 * t3 - 5.0 * t2 + 2.0
        q3 = -3.0 * t3 + 4.0 * t2 + t
        q4 = t3 - t2
        pts.append((0.5 * (P0[0] * q1 + P1[0] * q2 + P2[0] * q3 + P3[0] * q4),
                    0.5 * (P0[1] * q1 + P1[1] * q2 + P2[1] * q3 + P3[1] * q4)))
    return pts


def _ray_to_wall(track_mask, px, py, nx, ny):
    W, H = track_mask.shape
    d = 0
    while d < MAX_HALF_WIDTH:
        cx, cy = int(px + nx * d), int(py + ny * d)
        if cx < 0 or cx >= W or cy < 0 or cy >= H or not track_mask[cx, cy]:
            break
        d += 1
    return d


def generate_checkpoints(track_mask, nodes):
    """Walk a spline through ``nodes`` and drop a wall-to-wall checkpoint every SPACING px."""
    if len(nodes) < 2:
        return []

    padded = [nodes[0]] + nodes + [nodes[-1], nodes[-1]]
    spline = []
    for i in range(1, len(padded) - 2):
        spline.extend(catmull_rom(padded[i - 1], padded[i], padded[i + 1], padded[i + 2], 10))

    checkpoints = []
    acc = 0.0
    for i in range(1, len(spline) - 1):
        p0, p1, p2 = spline[i - 1], spline[i], spline[i + 1]
        acc += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if acc < SPACING:
            continue
        acc = 0.0
        dx, dy = p2[0] - p0[0], p2[1] - p0[1]
        L = math.hypot(dx, dy)
        if L == 0:
            continue
        nx, ny = -dy / L, dx / L
        w_left = _ray_to_wall(track_mask, p1[0], p1[1], nx, ny)
        w_right = _ray_to_wall(track_mask, p1[0], p1[1], -nx, -ny)
        checkpoints.append([int(p1[0] + nx * w_left), int(p1[1] + ny * w_left),
                            int(p1[0] - nx * w_right), int(p1[1] - ny * w_right)])
    return checkpoints


def save_track(img_path, nodes, checkpoints):
    """Write checkpoints + register the track. Returns the new track's index."""
    name = os.path.splitext(os.path.basename(img_path))[0]
    image_dest = f"assets/tracks/{name}.png"
    if os.path.abspath(img_path) != os.path.abspath(image_dest):
        os.makedirs(os.path.dirname(image_dest), exist_ok=True)
        shutil.copyfile(img_path, image_dest)

    cp_path = f"data/tracks/{name}.json"
    os.makedirs(os.path.dirname(cp_path), exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump(checkpoints, f, indent=4)

    cfg = {"tracks": {}}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    cfg.setdefault("tracks", {})[name] = {
        "name": name.replace("_", " ").title(),
        "image": image_dest,
        "checkpoints": cp_path,
        "max_frames": 7200,
        "start_pos": [int(nodes[0][0]), int(nodes[0][1])],
        "start_angle": int(math.degrees(math.atan2(-(nodes[1][1] - nodes[0][1]), nodes[1][0] - nodes[0][0]))),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
    return list(cfg["tracks"]).index(name)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/auto_import.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    pygame.init()
    try:
        base_img = pygame.image.load(img_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    W, H = base_img.get_size()
    screen = pygame.display.set_mode((min(1280, W), min(720, H)))
    base_img = base_img.convert_alpha()
    pygame.display.set_caption("Auto Checkpoint Generator")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    track_mask = build_track_mask(base_img)

    nodes = []
    checkpoints = []
    camera_x, camera_y = 0.0, 0.0
    zoom = 1.0
    panning = False
    last_mouse = (0, 0)

    instructions = [
        "AUTO CHECKPOINT GENERATOR",
        "1. Click roughly along the track (it auto-snaps to exact center)",
        "2. The checkpoints are auto-generated with numbers",
        "3. Press 'S' to save and import",
        "Scroll: Zoom, Right Click: Pan, Z: Undo",
    ]

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        wx = (mx / zoom) + camera_x
        wy = (my / zoom) + camera_y

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    nodes.append(snap_to_center(track_mask, wx, wy))
                    checkpoints = generate_checkpoints(track_mask, nodes)
                elif event.button == 3:
                    panning = True
                    last_mouse = (mx, my)
                elif event.button == 4:
                    zoom = min(3.0, zoom + 0.1)
                elif event.button == 5:
                    zoom = max(0.1, zoom - 0.1)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                panning = False
            elif event.type == pygame.MOUSEMOTION and panning:
                camera_x -= (mx - last_mouse[0]) / zoom
                camera_y -= (my - last_mouse[1]) / zoom
                last_mouse = (mx, my)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and nodes:
                    nodes.pop()
                    checkpoints = generate_checkpoints(track_mask, nodes)
                elif event.key == pygame.K_s and len(nodes) >= 2:
                    idx = save_track(img_path, nodes, checkpoints)
                    print(f"Track imported successfully! Run with: python main.py --track {idx}")
                    pygame.quit()
                    sys.exit(0)

        screen.fill((20, 20, 20))

        sub_rect = pygame.Rect(max(0, int(camera_x)), max(0, int(camera_y)),
                               min(W - max(0, int(camera_x)), int(screen.get_width() / zoom)),
                               min(H - max(0, int(camera_y)), int(screen.get_height() / zoom)))
        if sub_rect.width > 0 and sub_rect.height > 0:
            scaled_sub = pygame.transform.scale(base_img.subsurface(sub_rect),
                                                (int(sub_rect.width * zoom), int(sub_rect.height * zoom)))
            screen.blit(scaled_sub, (max(0, -camera_x * zoom), max(0, -camera_y * zoom)))

        def w2s(x, y):
            return int((x - camera_x) * zoom), int((y - camera_y) * zoom)

        for i, node in enumerate(nodes):
            sx, sy = w2s(*node)
            pygame.draw.circle(screen, (0, 0, 255), (sx, sy), max(2, int(8 * zoom)))
            if i > 0:
                pygame.draw.line(screen, (200, 200, 255), w2s(*nodes[i - 1]), (sx, sy), 2)

        for i, cp in enumerate(checkpoints):
            p1, p2 = w2s(cp[0], cp[1]), w2s(cp[2], cp[3])
            pygame.draw.line(screen, (0, 255, 0), p1, p2, 2)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 12)
            txt = font.render(str(i), True, (255, 255, 255))
            screen.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))

        for i, text in enumerate(instructions):
            screen.blit(font.render(text, True, (0, 0, 0)), (12, 12 + i * 20))
            screen.blit(font.render(text, True, (255, 255, 255)), (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
