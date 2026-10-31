import json

import numpy as np
import pygame
import pytest

from src.render.car import Car
from src.render.track import Track


def _load_tool(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auto_import_mask_matches_per_pixel_rule():
    auto_import = _load_tool("auto_import")
    pygame.init()
    surf = pygame.Surface((40, 30), pygame.SRCALPHA)
    rng = np.random.default_rng(0)
    for x in range(40):
        for y in range(30):
            surf.set_at((x, y), tuple(int(v) for v in rng.integers(0, 256, 4)))
    mask = auto_import.build_track_mask(surf)
    for x in range(40):
        for y in range(30):
            r, g, b, a = surf.get_at((x, y))
            assert mask[x, y] == ((r + g + b) / 3 < 128 and a > 0)


def test_auto_import_generates_checkpoints_across_road():
    auto_import = _load_tool("auto_import")
    mask = np.zeros((600, 200), dtype=bool)
    mask[:, 60:140] = True                       # horizontal road, y in [60, 140)
    cps = auto_import.generate_checkpoints(mask, [(50, 100), (300, 100), (550, 100)])
    assert len(cps) >= 3
    for x1, y1, x2, y2 in cps:
        assert abs(x1 - x2) <= 1                  # perpendicular to the road
        assert {min(y1, y2), max(y1, y2)} == {59, 140}   # first wall pixel on each side


def test_validate_track_flags_bad_start(tmp_path):
    validate_track = _load_tool("validate_track")
    pygame.init()
    surf = pygame.Surface((400, 400))
    surf.fill((255, 255, 255))
    pygame.draw.rect(surf, (0, 0, 0), (100, 100, 200, 200))
    img = tmp_path / "t.bmp"
    pygame.image.save(surf, str(img))
    cps = tmp_path / "c.json"
    cps.write_text(json.dumps([[250, 100, 250, 300], [150, 100, 150, 300]]))
    good = {"image": str(img), "checkpoints": str(cps), "start_pos": [150, 200], "start_angle": 0}
    errors, _ = validate_track.validate("t", good)
    assert errors == []
    bad = dict(good, start_pos=[105, 200])       # overlaps the left wall
    errors, _ = validate_track.validate("t", bad)
    assert any("overlaps a wall" in e for e in errors)


@pytest.mark.skipif(not pygame.image.get_extended(), reason="pygame built without PNG support")
def test_all_configured_tracks_have_valid_start():
    validate_track = _load_tool("validate_track")
    pygame.init()
    with open("config/track_config.json") as f:
        tracks = json.load(f)["tracks"]
    for key, entry in tracks.items():
        errors, _ = validate_track.validate(key, entry)
        assert not errors, (key, errors)


def test_human_car_scores_checkpoints():
    """Regression: manual mode never advanced checkpoints, so its fitness never grew."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("play", "play.py")
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    pygame.init()
    surf = pygame.Surface((1000, 1000))
    surf.fill((0, 0, 0))
    track = Track(1000, 1000)
    car = play.HumanCar([100, 100], track, [[200, 0, 200, 300]])
    car.angle = 0
    car.speed = 10
    for _ in range(12):
        car.update(surf)
    assert car.current_checkpoint_index == 1
    assert car.get_reward() > 0
    assert len(car.sensors) == len(Car.SENSOR_ANGLES)
