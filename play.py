"""Drive a track yourself with WASD.

    python play.py [--track ID] [--crash]

Scroll to zoom. By default collisions are off ("god mode"); pass --crash to
enable them, then press R to respawn after crashing.
"""

import argparse
import json
import math
import os
import sys

import pygame

from src.render.car import Car
from src.render.colors import Color
from src.render.minimap import Minimap
from src.render.track import Track

SIDEBAR_WIDTH = 300
TRACK_CONFIG_PATH = "config/track_config.json"


class HumanCar(Car):
    ACCEL = 0.15
    BRAKE = 0.6
    FRICTION = 0.15
    MAX_SPEED = 18.0
    MAX_REVERSE = -8.0
    TURN_SPEED = 1.75

    def __init__(self, position, track, checkpoints, collisions=False):
        super().__init__(position, track, checkpoints=checkpoints)
        self.speed = 0.0
        self.collisions = collisions

    def handle_input(self, keys):
        if keys[pygame.K_w]:
            self.speed += self.ACCEL
        elif keys[pygame.K_s]:
            self.speed -= self.BRAKE
        else:
            if self.speed > 0:
                self.speed -= self.FRICTION
            elif self.speed < 0:
                self.speed += self.FRICTION
            if abs(self.speed) < 0.3:
                self.speed = 0

        self.speed = max(self.MAX_REVERSE, min(self.speed, self.MAX_SPEED))

        # Steering flips in reverse, like a real car.
        if abs(self.speed) > 0.5:
            steer_dir = 1 if self.speed > 0 else -1
            if keys[pygame.K_a]:
                self.angle += self.TURN_SPEED * steer_dir
            if keys[pygame.K_d]:
                self.angle -= self.TURN_SPEED * steer_dir

    def update(self, track: pygame.Surface) -> None:
        old_center = list(self.center)
        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed
        self.update_center()
        self.check_checkpoint(old_center, self.center)

        self.refresh_corners_positions()
        if self.collisions:
            self.check_collision(track)
        self.refresh_sensors(track)


def load_track_entry(track_id: int):
    with open(TRACK_CONFIG_PATH, "r") as f:
        tracks = json.load(f).get("tracks", {})
    keys = list(tracks.keys())
    if not 0 <= track_id < len(keys):
        sys.exit(f"Invalid track ID. Choose between 0 and {len(keys) - 1}.")
    return tracks[keys[track_id]]


def track_help() -> str:
    try:
        with open(TRACK_CONFIG_PATH, "r") as f:
            tracks = json.load(f).get("tracks", {})
        return "Track ID: " + ", ".join(f"{i}={t.get('name', k)}" for i, (k, t) in enumerate(tracks.items()))
    except (OSError, ValueError):
        return "Track ID to load (0 = default)"


def main():
    parser = argparse.ArgumentParser(description="Play Neat Cars manually using WASD!")
    parser.add_argument('--track', type=int, default=0, help=track_help())
    parser.add_argument('--crash', action='store_true', help="Enable wall collisions (default: god mode).")
    args = parser.parse_args()

    track_data = load_track_entry(args.track)
    track_img_path = track_data["image"]
    start_pos = track_data.get("start_pos", [200, 200])
    start_angle = track_data.get("start_angle", 0)
    checkpoints_path = track_data.get("checkpoints")

    pygame.init()
    info = pygame.display.Info()
    WIDTH = min(1920, int(info.current_w * 0.95))
    HEIGHT = min(1080, int(info.current_h * 0.95))
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neat Cars - Manual Driving Mode")
    clock = pygame.time.Clock()

    try:
        track_img = pygame.image.load(track_img_path).convert()
    except Exception as e:
        sys.exit(f"Error loading track image: {e}")
    track_w, track_h = track_img.get_width(), track_img.get_height()

    track = Track(track_w, track_h)
    track.surface.blit(track_img, (0, 0))

    minimap = Minimap(SIDEBAR_WIDTH - 40, (SIDEBAR_WIDTH - 40) * (track_h / track_w), track_w, track_h)
    minimap.set_track_surface(track.surface)

    checkpoints = []
    if checkpoints_path and os.path.exists(checkpoints_path):
        with open(checkpoints_path, 'r') as f:
            checkpoints = json.load(f)

    # start_pos is the car's centre (same convention as the training engine).
    spawn = [start_pos[0] - Car.CAR_SIZE_X / 2, start_pos[1] - Car.CAR_SIZE_Y / 2]

    def new_car():
        c = HumanCar(list(spawn), track, checkpoints, collisions=args.crash)
        c.angle = start_angle
        return c

    car = new_car()

    hud_font = pygame.font.SysFont(None, 36)
    label_fonts = {}   # checkpoint label font per pixel size (SysFont is slow to create)

    def label_font(size):
        if size not in label_fonts:
            label_fonts[size] = pygame.font.SysFont("monospace", size)
        return label_fonts[size]

    game_w = WIDTH - SIDEBAR_WIDTH
    game_h = HEIGHT
    min_zoom = max(0.05, game_w / track_w, game_h / track_h)   # never show beyond the track
    zoom = max(1.0, min_zoom)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom = min(4.0, zoom + 0.1)
                elif event.button == 5:
                    zoom = max(min_zoom, zoom - 0.1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and not car.alive:
                car = new_car()

        keys = pygame.key.get_pressed()
        if car.alive:
            car.handle_input(keys)
            car.update(track.surface)

        screen.fill((15, 15, 20))

        view_w = game_w / zoom
        view_h = game_h / zoom

        # Camera follows the car, clamped so we never read outside the track surface.
        cam_x = max(0, min(car.center[0] - view_w / 2, track_w - view_w))
        cam_y = max(0, min(car.center[1] - view_h / 2, track_h - view_h))

        # 1. Track: scale only the visible sub-rectangle.
        sx, sy = int(cam_x), int(cam_y)
        sub_rect = pygame.Rect(sx, sy, min(int(view_w), track_w - sx), min(int(view_h), track_h - sy))
        screen.blit(pygame.transform.scale(track.surface.subsurface(sub_rect), (game_w, game_h)), (0, 0))

        def w2s(x, y):
            return int((x - cam_x) * zoom), int((y - cam_y) * zoom)

        # 2. Checkpoints (next one highlighted)
        if checkpoints:
            font = label_font(max(10, int(14 * zoom)))
            next_cp = car.current_checkpoint_index % len(checkpoints)
            for i, cp in enumerate(checkpoints):
                p1, p2 = w2s(cp[0], cp[1]), w2s(cp[2], cp[3])
                color = (255, 255, 0) if i == next_cp else (0, 255, 0)
                pygame.draw.line(screen, color, p1, p2, max(1, int(2 * zoom)))
                cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                txt = font.render(str(i), True, (255, 255, 255))
                screen.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))

        # 3. Car & sensors
        if car.alive:
            pygame.draw.polygon(screen, Color.BLUE, [w2s(c[0], c[1]) for c in car.corners])
            shifted_center = w2s(car.center[0], car.center[1])
            for sensor in car.sensors:
                shifted_sensor = w2s(sensor[0][0], sensor[0][1])
                pygame.draw.line(screen, Color.GREEN, shifted_center, shifted_sensor, max(1, int(2 * zoom)))
                pygame.draw.circle(screen, Color.RED, shifted_sensor, max(2, int(4 * zoom)))

        # 4. Sidebar
        pygame.draw.rect(screen, (30, 30, 35), pygame.Rect(game_w, 0, SIDEBAR_WIDTH, HEIGHT))
        pygame.draw.line(screen, (100, 100, 100), (game_w, 0), (game_w, HEIGHT), 2)

        texts = [
            "MANUAL DRIVING MODE",
            f"Track: {track_data.get('name', args.track)}",
            "",
            f"Speed: {car.speed:.1f} px/f",
            f"Fitness: {car.get_reward():.1f}",
            f"Checkpoints: {car.current_checkpoint_index}",
            "",
            f"STATUS: {'ALIVE' if car.alive else 'CRASHED'}",
            f"Collisions: {'ON' if args.crash else 'OFF (god mode)'}",
            "",
            "CONTROLS:",
            "W: Accelerate",
            "S: Brake",
            "A: Turn Left",
            "D: Turn Right",
            "R: Respawn (If Dead)",
        ]

        y_offset = 20
        for text in texts:
            color = Color.GREEN if "ALIVE" in text else (Color.RED if "CRASHED" in text else Color.WHITE)
            screen.blit(hud_font.render(text, True, color), (game_w + 20, y_offset))
            y_offset += 40

        minimap_pos = (game_w + 20, HEIGHT - minimap.height - 20)
        minimap.draw(screen, [car], pygame.Rect(cam_x, cam_y, view_w, view_h), minimap_pos)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
