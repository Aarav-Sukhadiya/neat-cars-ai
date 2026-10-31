import sys
import json
import os
from unittest.mock import patch

import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"

import importlib.util
spec = importlib.util.spec_from_file_location("manual_checkpoint_tool", "tools/manual_checkpoint_tool.py")
manual_checkpoint_tool = importlib.util.module_from_spec(spec)
sys.modules["manual_checkpoint_tool"] = manual_checkpoint_tool
spec.loader.exec_module(manual_checkpoint_tool)

@pytest.fixture(autouse=True)
def setup_dummy_image(tmp_path):
    img_path = tmp_path / "dummy_path.bmp"
    surf = pygame.Surface((800, 600))
    pygame.image.save(surf, str(img_path))
    yield str(img_path)

def test_missing_argument(capsys):
    with patch.object(sys, 'argv', ['manual_checkpoint_tool.py']):
        with pytest.raises(SystemExit) as excinfo:
            manual_checkpoint_tool.main()
        assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.out

def test_camera_wasd_movement(setup_dummy_image):
    import collections
    keys = collections.defaultdict(bool)
    keys[pygame.K_w] = True
    keys[pygame.K_d] = True
    
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.event.get', return_value=[pygame.event.Event(pygame.QUIT)]), \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        manual_checkpoint_tool.main()

def test_checkpoint_placement_and_undo(setup_dummy_image):
    import collections
    keys = collections.defaultdict(bool)
    
    events = [
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (100, 100)})],
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (200, 200)})],
        [pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_u})],
        [pygame.event.Event(pygame.QUIT)]
    ]
    
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.event.get', side_effect=events), \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        manual_checkpoint_tool.main()

@patch('manual_checkpoint_tool.json.dump')
@patch('manual_checkpoint_tool.open')
@patch('manual_checkpoint_tool.os.makedirs')
@patch('manual_checkpoint_tool.os.path.exists', return_value=True)
def test_save_checkpoints(mock_exists, mock_makedirs, mock_open, mock_json_dump, setup_dummy_image):
    import collections
    keys = collections.defaultdict(bool)
    
    events = [
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (100, 100)})],
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (150, 150)})],
        [pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN})]
    ]
    
    def mock_json_load(f):
        return {"tracks": {}}
        
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.event.get', side_effect=events), \
         patch('manual_checkpoint_tool.json.load', side_effect=mock_json_load), \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        with pytest.raises(SystemExit) as excinfo:
            manual_checkpoint_tool.main()
        assert excinfo.value.code == 0
            
    assert mock_json_dump.call_count == 2

def test_scale_and_rotate(setup_dummy_image):
    import collections
    keys = collections.defaultdict(bool)
    keys[pygame.K_PLUS] = True
    
    events = [
        [],
        [pygame.event.Event(pygame.QUIT)]
    ]
    
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.mouse.get_pressed', return_value=(False, False, True)), \
         patch('pygame.event.get', side_effect=events), \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        manual_checkpoint_tool.main()


def test_s_key_pans_but_does_not_save(setup_dummy_image):
    """Regression: KEYDOWN S (also 'pan down') used to save the track and exit."""
    import collections
    keys = collections.defaultdict(bool)
    keys[pygame.K_s] = True
    events = [
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': (100, 100)})],
        [pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_s})],
        [pygame.event.Event(pygame.QUIT)],
    ]
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.event.get', side_effect=events), \
         patch('manual_checkpoint_tool.json.dump') as dump, \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        manual_checkpoint_tool.main()   # returns normally (no SystemExit)
    dump.assert_not_called()


def test_saved_start_is_behind_first_checkpoint(setup_dummy_image, tmp_path, monkeypatch):
    """Regression: start_pos was the midpoint of checkpoint 0, so it was never crossed."""
    import collections
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "track_config.json").write_text('{"tracks": {}}')
    keys = collections.defaultdict(bool)
    # Two vertical checkpoints (angle 90 => line along +y), second one 200 px to the right.
    events = [
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1})],
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1})],
        [pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN})],
    ]
    mouse = iter([(300, 300)] * 2 + [(500, 300)] * 100)
    with patch('pygame.key.get_pressed', return_value=keys), \
         patch('pygame.mouse.get_pos', side_effect=lambda: next(mouse)), \
         patch('pygame.event.get', side_effect=events), \
         patch.object(sys, 'argv', ['manual_checkpoint_tool.py', setup_dummy_image]):
        with pytest.raises(SystemExit):
            manual_checkpoint_tool.main()
    cfg = json.loads((tmp_path / "config" / "track_config.json").read_text())
    entry = next(iter(cfg["tracks"].values()))
    with open(entry["checkpoints"]) as f:
        cps = json.load(f)
    mid_x = (cps[0][0] + cps[0][2]) / 2
    assert entry["start_angle"] == 0                       # heading right, towards checkpoint 1
    assert entry["start_pos"][0] == round(mid_x) - manual_checkpoint_tool.SPAWN_BEHIND_CP0
