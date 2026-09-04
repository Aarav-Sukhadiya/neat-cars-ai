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
            self.speed += 0.5
        elif keys[pygame.K_s]:
            self.speed -= 1.0  # Stronger brakes
        else:
            # Friction
            if self.speed > 0:
                self.speed -= 0.25
            elif self.speed < 0:
                self.speed += 0.25
            if abs(self.speed) < 0.3:
                self.speed = 0

        # Cap speed
        self.speed = max(-10.0, min(self.speed, 25.0))

        # Steering feels better when it's crisp, and flips in reverse
        if abs(self.speed) > 0.5:
            turn_speed = 6.0
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
        self.check_collision(track)
        
        self.sensors.clear()
        for sensor_angle in [-90, -45, -20, 0, 20, 45, 90]:
            self.check_sensor(sensor_angle, track)

def main():
    parser = argparse.ArgumentParser(description="Play Neat Cars manually using WASD!")
    parser.add_argument('--track', type=int, default=0, help="Track ID to play (0=Massive, 1=Intersection, 2=Super Hard)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neat Cars - Manual Driving Mode")
    clock = pygame.time.Clock()

    # Track Configuration
    if args.track == 0:
        track_img_path = "assets/tracks/massive_loop.png"
        track_w, track_h = 4000, 3000
        start_pos = [200, 950]
        checkpoints_path = "data/tracks/massive_loop.json"
    elif args.track == 1:
        track_img_path = "assets/tracks/intersection_loop.png"
        track_w, track_h = 4000, 4000
        start_pos = [2000, 2900]
        checkpoints_path = "data/tracks/intersection_loop.json"
    elif args.track == 2:
        track_img_path = "assets/tracks/super_hard.png"
        track_w, track_h = 4000, 4000
        start_pos = [500, 3500]
        checkpoints_path = "data/tracks/super_hard.json"
    else:
        print("Invalid track ID. Choose 0, 1, or 2.")
        sys.exit(1)

    # Initialize components
    track = Track(track_w, track_h)
    try:
        track_img = pygame.image.load(track_img_path).convert()
        track.surface.blit(track_img, (0, 0))
    except Exception as e:
        print(f"Error loading track image: {e}")
        sys.exit(1)

    camera = Camera(WIDTH - SIDEBAR_WIDTH, HEIGHT, track_w, track_h)
    minimap = Minimap(SIDEBAR_WIDTH - 40, (SIDEBAR_WIDTH - 40) * (track_h / track_w), track_w, track_h)
    minimap.set_track_surface(track.surface)

    # Load checkpoints so you can see your score
    import json
    checkpoints = []
    if os.path.exists(checkpoints_path):
        with open(checkpoints_path, 'r') as f:
            checkpoints = json.load(f)

    car = HumanCar(start_pos, track, checkpoints)

    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Reset car if dead
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not car.alive:
                    car = HumanCar(start_pos, track, checkpoints)

        # Handle Keyboard Input
        keys = pygame.key.get_pressed()
        if car.alive:
            car.handle_input(keys)
            car.update_sprite(track.surface)

        camera.update(car.center)

        # Render
        screen.fill((15, 15, 20))
        
        # 1. Track
        screen.blit(track.surface, (-camera.offset_x, -camera.offset_y))

        # 2. Checkpoints
        if car.checkpoints:
            for cp in car.checkpoints:
                pygame.draw.line(screen, (0, 255, 0), camera.apply((cp[0], cp[1])), camera.apply((cp[2], cp[3])), 2)

        # 3. Car & Sensors
        if car.alive:
            shifted_corners = [camera.apply(c) for c in car.corners]
            pygame.draw.polygon(screen, Color.BLUE, shifted_corners)
            
            shifted_center = camera.apply(car.center)
            for sensor in car.sensors:
                shifted_sensor = camera.apply(sensor[0])
                pygame.draw.line(screen, Color.GREEN, shifted_center, shifted_sensor, 2)
                pygame.draw.circle(screen, Color.RED, shifted_sensor, 4)

        # 4. Sidebar UI
        sidebar_rect = pygame.Rect(WIDTH - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
        pygame.draw.rect(screen, (30, 30, 35), sidebar_rect)
        pygame.draw.line(screen, (100, 100, 100), (WIDTH - SIDEBAR_WIDTH, 0), (WIDTH - SIDEBAR_WIDTH, HEIGHT), 2)

        # HUD Text
        texts = [
            f"MANUAL DRIVING MODE",
            f"Track: {args.track}",
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
        cam_rect = pygame.Rect(camera.offset_x, camera.offset_y, camera.width, camera.height)
        minimap.draw(screen, [car], cam_rect, (minimap_x, minimap_y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
