import pygame
import argparse
import sys
import os

from src.render.track import Track
from src.render.car import Car
from src.render.camera import Camera
from src.render.minimap import Minimap
from src.render.colors import Color

WIDTH = 1920
HEIGHT = 1080
SIDEBAR_WIDTH = 300

class HumanCar(Car):
    def __init__(self, position, track, checkpoints):
        super().__init__(position, track, checkpoints=checkpoints)
        self.speed = 0.0

    def handle_input(self, keys):
        if keys[pygame.K_w]:
            self.speed += 0.15
        elif keys[pygame.K_s]:
            self.speed -= 0.6  # Stronger brakes
        else:
            # Friction
            if self.speed > 0:
                self.speed -= 0.15
            elif self.speed < 0:
                self.speed += 0.15
            if abs(self.speed) < 0.3:
                self.speed = 0

        # Cap speed
        self.speed = max(-8.0, min(self.speed, 18.0))

        # Steering feels better when it's crisp, and flips in reverse
        if abs(self.speed) > 0.5:
            turn_speed = 1.75
            steer_dir = 1 if self.speed > 0 else -1
            if keys[pygame.K_a]:
                self.angle += turn_speed * steer_dir
            if keys[pygame.K_d]:
                self.angle -= turn_speed * steer_dir

    def update_sprite(self, track: pygame.Surface) -> None:
        import math
        self.update_center()
        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed
        
        self.refresh_corners_positions()
        # self.check_collision(track)  # God mode: No crashing!
        
        self.sensors.clear()
        for sensor_angle in [-90, -45, -20, 0, 20, 45, 90]:
            self.check_sensor(sensor_angle, track)

def main():
    import json
    import os
    track_help = "Track ID to load (0 = default)"
    try:
        with open("config/track_config.json", "r") as f:
            tconfig = json.load(f)
        tracks_dict = tconfig.get("tracks", {})
        track_keys = list(tracks_dict.keys())
        help_parts = []
        for i, key in enumerate(track_keys):
            name = tracks_dict[key].get("name", key)
            help_parts.append(f"{i}={name}")
        track_help = "Track ID: " + ", ".join(help_parts)
    except:
        pass

    parser = argparse.ArgumentParser(description="Play Neat Cars manually using WASD!")
    parser.add_argument('--track', type=int, default=0, help=track_help)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neat Cars - Manual Driving Mode")
    clock = pygame.time.Clock()

    # Track Configuration

    import json
    
    # Load track config dynamically
    with open("config/track_config.json", "r") as f:
        tconfig = json.load(f)
        
    tracks_dict = tconfig.get("tracks", {})
    track_keys = list(tracks_dict.keys())
    
    if 0 <= args.track < len(track_keys):
        track_key = track_keys[args.track]
        track_data = tracks_dict[track_key]
        
        track_img_path = track_data["image"]
        start_pos = track_data.get("start_pos", [200, 200])
        start_angle = track_data.get("start_angle", 0)
        checkpoints_path = track_data.get("checkpoints", None)
        
        # Width/height will be read directly from the image file below
        
    else:
        print(f"Invalid track ID. Choose between 0 and {len(track_keys) - 1}.")
        sys.exit(1)
    # Initialize components
    try:
        track_img = pygame.image.load(track_img_path).convert()
        track_w, track_h = track_img.get_width(), track_img.get_height()
    except Exception as e:
        print(f"Error loading track image: {e}")
        sys.exit(1)
        
    track = Track(track_w, track_h)
    track.surface.blit(track_img, (0, 0))

    camera = Camera(WIDTH - SIDEBAR_WIDTH, HEIGHT, track_w, track_h)
    minimap = Minimap(SIDEBAR_WIDTH - 40, (SIDEBAR_WIDTH - 40) * (track_h / track_w), track_w, track_h)
    minimap.set_track_surface(track.surface)

    # Load checkpoints so you can see your score
    import json
    checkpoints = []
    if os.path.exists(checkpoints_path):
        with open(checkpoints_path, 'r') as f:
            checkpoints = json.load(f)

    car = HumanCar(list(start_pos), track, checkpoints)
    car.angle = start_angle

    font = pygame.font.SysFont(None, 36)

    zoom = 1.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom = min(4.0, zoom + 0.1)
                elif event.button == 5:
                    zoom -= 0.1 # Clamped later
                    
            # Reset car if dead
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not car.alive:
                    car = HumanCar(list(start_pos), track, checkpoints)
                    car.angle = start_angle

        # Handle Keyboard Input
        keys = pygame.key.get_pressed()
        if car.alive:
            car.handle_input(keys)
            car.update_sprite(track.surface)

        # Render
        screen.fill((15, 15, 20))
        
        game_w = WIDTH - SIDEBAR_WIDTH
        game_h = HEIGHT
        
        # Clamp zoom to prevent crashing subsurface
        min_zoom = max(game_w / track_w, game_h / track_h, 0.1)
        zoom = max(min_zoom, zoom)
        
        view_w = game_w / zoom
        view_h = game_h / zoom
        
        cam_x = max(0, min(car.center[0] - view_w / 2, track_w - view_w))
        cam_y = max(0, min(car.center[1] - view_h / 2, track_h - view_h))
        
        # 1. Track Subsurface
        sub_rect = pygame.Rect(int(cam_x), int(cam_y), int(view_w), int(view_h))
        visible_track = track.surface.subsurface(sub_rect)
        scaled_track = pygame.transform.scale(visible_track, (game_w, game_h))
        screen.blit(scaled_track, (0, 0))
        
        def w2s(x, y):
            return int((x - cam_x) * zoom), int((y - cam_y) * zoom)

        # 2. Checkpoints
        if car.checkpoints:
            font = pygame.font.SysFont("monospace", max(10, int(14*zoom)))
            for i, cp in enumerate(car.checkpoints):
                p1, p2 = w2s(cp[0], cp[1]), w2s(cp[2], cp[3])
                pygame.draw.line(screen, (0, 255, 0), p1, p2, max(1, int(2*zoom)))
                # Number them
                cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                txt = font.render(str(i+1), True, (255, 255, 255))
                screen.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2))

        # 3. Car & Sensors
        if car.alive:
            shifted_corners = [w2s(c[0], c[1]) for c in car.corners]
            pygame.draw.polygon(screen, Color.BLUE, shifted_corners)
            
            shifted_center = w2s(car.center[0], car.center[1])
            for sensor in car.sensors:
                shifted_sensor = w2s(sensor[0][0], sensor[0][1])
                pygame.draw.line(screen, Color.GREEN, shifted_center, shifted_sensor, max(1, int(2*zoom)))
                pygame.draw.circle(screen, Color.RED, shifted_sensor, max(2, int(4*zoom)))

        # 4. Sidebar UI
        sidebar_rect = pygame.Rect(WIDTH - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
        pygame.draw.rect(screen, (30, 30, 35), sidebar_rect)
        pygame.draw.line(screen, (100, 100, 100), (WIDTH - SIDEBAR_WIDTH, 0), (WIDTH - SIDEBAR_WIDTH, HEIGHT), 2)

        # HUD Text
        texts = [
            f"MANUAL DRIVING MODE",
            f"Track: {track_data.get('name', args.track)}",
            "",
            f"Speed: {car.speed:.1f} px/f",
            f"Fitness: {car.get_reward():.1f}",
            "",
            f"STATUS: {'ALIVE' if car.alive else 'CRASHED / STARVED'}",
            "",
            "CONTROLS:",
            "W: Accelerate",
            "S: Brake",
            "A: Turn Left",
            "D: Turn Right",
            "R: Respawn (If Dead)"
        ]

        y_offset = 20
        for text in texts:
            color = Color.GREEN if "ALIVE" in text else (Color.RED if "CRASHED" in text else Color.WHITE)
            rendered_text = font.render(text, True, color)
            screen.blit(rendered_text, (WIDTH - SIDEBAR_WIDTH + 20, y_offset))
            y_offset += 40

        # Minimap
        minimap_x = WIDTH - SIDEBAR_WIDTH + 20
        minimap_y = HEIGHT - minimap.height - 20
        cam_rect = pygame.Rect(cam_x, cam_y, view_w, view_h)
        minimap.draw(screen, [car], cam_rect, (minimap_x, minimap_y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
