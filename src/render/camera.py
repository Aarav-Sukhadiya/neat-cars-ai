class Camera:
    def __init__(self, width: int, height: int, track_width: int, track_height: int):
        self.width = width
        self.height = height
        self.track_width = track_width
        self.track_height = track_height
        
        self.offset_x = 0
        self.offset_y = 0
        
    def update(self, target_pos: tuple[float, float]) -> None:
        """Update the camera offset to center on the target position, clamped to track bounds."""
        x, y = target_pos
        
        # Calculate ideal offset to center the target
        ideal_x = int(x - self.width / 2)
        ideal_y = int(y - self.height / 2)
        
        # Clamp to track boundaries
        max_offset_x = max(0, self.track_width - self.width)
        max_offset_y = max(0, self.track_height - self.height)
        
        self.offset_x = max(0, min(ideal_x, max_offset_x))
        self.offset_y = max(0, min(ideal_y, max_offset_y))
        
    def apply(self, pos: tuple[float, float]) -> tuple[float, float]:
        """Convert a world position to a screen position."""
        return (pos[0] - self.offset_x, pos[1] - self.offset_y)
