import re

with open('src/render/car.py', 'r') as f:
    content = f.read()

# Remove sprite paths
content = re.sub(r'CAR_SPRITE_PATH = "assets/car.png"\nDEAD_CAR_SPRITE_PATH = "assets/dead_car.png"', '', content)

# Remove shared sprite methods and vars
content = re.sub(r'    _shared_sprite: pygame\.Surface = None\n    _shared_dead_sprite: pygame\.Surface = None\n\n    @classmethod.*?return cls\._shared_dead_sprite\n\n', '', content, flags=re.DOTALL)

# Update __init__
init_code_old = """        self.old_position = start_position.copy()
        # Use shared class-level sprite (loaded once, never per-car per-generation)
        self._sprite = self._get_shared_sprite()

        # Assigning the current sprite to this variable sprite (the one which will be rotated)
        # we need this to avoid out of memory errors from pygame
        self.sprite = self._sprite

        self.position = start_position.copy()"""
init_code_new = """        self.old_position = start_position.copy()
        self.position = start_position.copy()"""
content = content.replace(init_code_old, init_code_new)

# Update draw
draw_code_old = """    def draw(self, track: pygame.Surface) -> None:
        \"\"\"Draw the car on the track (and its sensors if enabled)

        Args:
            track (pygame.Surface): The track on which the car will be drawn
        \"\"\"
        
        if self.alive:
            track.blit(self.sprite, self.position)
        else:
            # Switch to the dead sprite (loaded once via class-level cache)
            dead_sprite = Car._get_shared_dead_sprite()
            sprite_as_rect = dead_sprite.get_rect()
            rotated_dead = pygame.transform.rotate(dead_sprite, self.angle)
            sprite_as_rect.center = rotated_dead.get_rect().center
            track.blit(rotated_dead.subsurface(sprite_as_rect), self.position)

        if Car.DRAW_SENSORS:
            for sensor in self.sensors:
                pygame.draw.line(track, Color.GREEN, self.center, sensor[0], 1)
                pygame.draw.circle(track, Color.RED, sensor[0], 3)"""
draw_code_new = """    def draw(self, track: pygame.Surface) -> None:
        \"\"\"Draw the car polygon on the track\"\"\"
        if not self.corners:
            return
            
        color = Color.BLUE if self.alive else Color.RED
        pygame.draw.polygon(track, color, self.corners)

        if Car.DRAW_SENSORS:
            for sensor in self.sensors:
                pygame.draw.line(track, Color.GREEN, self.center, sensor[0], 1)
                pygame.draw.circle(track, Color.RED, sensor[0], 3)"""
content = content.replace(draw_code_old, draw_code_new)

# Update update_center and remove cache
center_code_old = """    _ROTATED_SPRITES_CACHE = {}

    def update_center(self) -> None:
        \"\"\"Update the center of the car after a rotation (when it turns left or right)\"\"\"
        
        # Pull from cache instead of rotating dynamically every frame
        if self.angle not in Car._ROTATED_SPRITES_CACHE:
            sprite_as_rect = self._sprite.get_rect()
            rotated_sprite = pygame.transform.rotate(self._sprite, self.angle)
            sprite_as_rect.center = rotated_sprite.get_rect().center
            Car._ROTATED_SPRITES_CACHE[self.angle] = rotated_sprite.subsurface(sprite_as_rect)
            
        self.sprite = Car._ROTATED_SPRITES_CACHE[self.angle]
        # Calculate New Center
        self.center = [
            int(self.position[0]) + Car.CAR_SIZE_X / 2,
            int(self.position[1]) + Car.CAR_SIZE_Y / 2
        ]"""
center_code_new = """    def update_center(self) -> None:
        \"\"\"Update the center of the car using its top-left position\"\"\"
        self.center = [
            int(self.position[0]) + Car.CAR_SIZE_X / 2,
            int(self.position[1]) + Car.CAR_SIZE_Y / 2
        ]"""
content = content.replace(center_code_old, center_code_new)

with open('src/render/car.py', 'w') as f:
    f.write(content)

