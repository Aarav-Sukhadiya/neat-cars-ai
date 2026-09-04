import pygame
from src.render.colors import Color

class Minimap:
    def __init__(self, width: int, height: int, track_width: int, track_height: int):
        self.width = width
        self.height = height
        self.track_width = track_width
        self.track_height = track_height
        
        self.scale_x = width / float(track_width)
        self.scale_y = height / float(track_height)
        
        self.surface = pygame.Surface((width, height))
        self.surface.set_alpha(200) # Slightly transparent
        
        self.track_preview = None
        
    def set_track_surface(self, track_surface: pygame.Surface) -> None:
        """Cache a scaled down version of the track."""
        self.track_preview = pygame.transform.smoothscale(track_surface, (self.width, self.height))
        
    def world_to_minimap(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Convert a world position to a minimap position."""
        return (int(pos[0] * self.scale_x), int(pos[1] * self.scale_y))
        
    def draw(self, screen: pygame.Surface, cars: list, camera_rect: pygame.Rect, dest_pos: tuple[int, int]) -> None:
        """Draw the minimap and blit it to the screen."""
        # pragma: no cover
        if self.track_preview:
            self.surface.blit(self.track_preview, (0, 0))
        else:
            self.surface.fill((50, 50, 50))
            
        # Draw camera view rect on minimap
        cam_x, cam_y = self.world_to_minimap((camera_rect.x, camera_rect.y))
        cam_w = int(camera_rect.width * self.scale_x)
        cam_h = int(camera_rect.height * self.scale_y)
        pygame.draw.rect(self.surface, Color.WHITE, (cam_x, cam_y, cam_w, cam_h), 1)
            
        # Draw cars on minimap
        for car in cars:
            if not getattr(car, 'has_been_rendered_as_dead', False) or getattr(car, 'alive', False):
                mx, my = self.world_to_minimap(car.center)
                color = Color.GREEN if car.alive else Color.RED
                pygame.draw.circle(self.surface, color, (mx, my), 2)
                
        # Draw a border around the minimap
        pygame.draw.rect(self.surface, Color.WHITE, (0, 0, self.width, self.height), 2)
                
        screen.blit(self.surface, dest_pos)
