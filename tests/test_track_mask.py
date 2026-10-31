import json
import os
import pygame
from unittest.mock import patch
from src.render.engine import Engine

os.environ["SDL_VIDEODRIVER"] = "dummy"

def test_engine_loads_mask_separately(tmp_path):
    visual_path = tmp_path / "visual.bmp"
    mask_path = tmp_path / "mask.bmp"
    
    pygame.init()
    pygame.display.set_mode((1, 1))
    
    vis_surf = pygame.Surface((100, 100))
    vis_surf.fill((255, 0, 0)) # Red visual
    pygame.image.save(vis_surf, str(visual_path))
    
    mask_surf = pygame.Surface((100, 100))
    mask_surf.fill((0, 0, 0)) # Black mask
    pygame.image.save(mask_surf, str(mask_path))
    
    mock_config = {
        "current_track": "test_track",
        "tracks": {
            "test_track": {
                "image": str(visual_path),
                "mask": str(mask_path),
                "checkpoints": "data/checkpoints.json"
            }
        }
    }
    
    config_path = tmp_path / "track_config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)
        
    original_exists = os.path.exists
    def mock_exists(path):
        if path == "config/track_config.json":
            return True
        if "hall_of_fame" in path:
            return False
        return original_exists(path)

    original_open = open
    def smart_open(path, *args, **kwargs):
        if path == "config/track_config.json":
            return original_open(config_path, *args, **kwargs)
        if "hall_of_fame" in str(path):
            raise FileNotFoundError("Mocked file not found")
        return original_open(path, *args, **kwargs)

    with patch('src.render.engine.os.path.exists', side_effect=mock_exists), \
         patch('builtins.open', side_effect=smart_open):
        engine = Engine("config/neat_config.ini", False, 10, headless=True, track_id=0)
        
    assert hasattr(engine.track, 'mask_surface'), "Track should have a mask_surface property"
    assert engine.track.mask_surface != engine.track.surface, "Mask surface should be a separate object"
    
    assert engine.track.surface.get_at((50, 50)) == (255, 0, 0, 255)
    assert engine.track.mask_surface.get_at((50, 50)) == (0, 0, 0, 255)
