"""Engine tests on a synthetic track inside a temp directory.

The Engine uses relative paths (config/, data/), so each test chdirs into a temp
dir: the real data/hall_of_fame.pkl and config files are never touched.
"""

import json
import os
import pickle

import neat
import pygame
import pytest

from src.ai.car_ai import CarAI
from src.render.car import Car
from src.render import engine as engine_mod
from src.render.engine import Engine, StagnationTermination, SuccessTermination

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEAT_CONFIG = os.path.join(ROOT, "config", "neat_config.ini")


@pytest.fixture(autouse=True)
def restore_class_state():
    saved = (Car.DEFAULT_ANGLE, CarAI.MAX_FRAMES, CarAI.NN_POSITION)
    yield
    Car.DEFAULT_ANGLE, CarAI.MAX_FRAMES, CarAI.NN_POSITION = saved


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A 2000x1000 track: white border wall, black open interior, one far checkpoint."""
    pygame.init()
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    surf = pygame.Surface((2000, 1000))
    surf.fill((255, 255, 255))
    pygame.draw.rect(surf, (0, 0, 0), (50, 50, 1900, 900))
    pygame.image.save(surf, str(tmp_path / "track.bmp"))
    with open(tmp_path / "cps.json", "w") as f:
        json.dump([[800, 400, 800, 600]], f)
    cfg = {
        "current_track": "open",
        "tracks": {
            "other": {"image": "missing.bmp", "checkpoints": "cps.json"},
            "open": {"name": "Open", "image": "track.bmp", "checkpoints": "cps.json",
                     "max_frames": 30, "start_pos": [200, 500], "start_angle": 0},
        },
    }
    with open(tmp_path / "config" / "track_config.json", "w") as f:
        json.dump(cfg, f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _population(n=12):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, NEAT_CONFIG)
    config.pop_size = n
    pop = neat.Population(config)
    return pop, config


def _run_generation(engine, pop, config):
    engine.population = pop
    engine.run_simulation(list(pop.population.items()), config)


def test_track_id_none_uses_current_track(workdir):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True)
    assert engine.track_image_path == "track.bmp"
    assert CarAI.MAX_FRAMES == 30
    assert engine.decided_car_pos == [200 - Car.CAR_SIZE_X / 2, 500 - Car.CAR_SIZE_Y / 2]


def test_invalid_track_id_raises(workdir):
    with pytest.raises(ValueError):
        Engine(NEAT_CONFIG, False, 1, headless=True, track_id=7)


def test_missing_track_image_raises(workdir):
    # Previously this silently trained on a blank all-wall track.
    with pytest.raises(RuntimeError, match="missing.bmp"):
        Engine(NEAT_CONFIG, False, 1, headless=True, track_id=0)


def test_generation_runs_exactly_max_frames(workdir, monkeypatch, capsys):
    """Regression: frame_count used to be incremented twice per frame (half-length episodes)."""
    engine = Engine(NEAT_CONFIG, False, 1, headless=True)
    engine.hall_of_fame = []
    calls = {"n": 0}
    original = CarAI.compute

    def counting(self, track):
        calls["n"] += 1
        return original(self, track)

    monkeypatch.setattr(CarAI, "compute", counting)
    pop, config = _population()
    _run_generation(engine, pop, config)
    capsys.readouterr()
    assert calls["n"] == CarAI.MAX_FRAMES


def test_end_of_generation_stats_and_hall_of_fame(workdir, capsys):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True)
    engine.hall_of_fame = []
    pop, config = _population()
    _run_generation(engine, pop, config)
    capsys.readouterr()

    stats = engine.prev_gen_stats
    assert set(stats) == {"gen", "best_fitness", "median_fitness", "survival_rate", "avg_speed"}
    # Regression: avg_speed was always 0 (driven distance was never synced from the physics arrays).
    assert stats["avg_speed"] > 0

    with open("data/hall_of_fame.pkl", "rb") as f:
        hof = pickle.load(f)
    assert 0 < len(hof) <= engine_mod.HALL_OF_FAME_SIZE
    fits = [fit for fit, _ in hof]
    assert fits == sorted(fits, reverse=True)
    # Stored genomes are copies, not the live population objects.
    live = {id(g) for g in pop.population.values()}
    assert all(id(g) not in live for _, g in engine.hall_of_fame)


def test_hall_of_fame_keeps_older_entry_on_tie(workdir):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True)
    pop, _ = _population(2)
    g_old, g_new = list(pop.population.values())
    g_new.connections = g_old.connections   # identical brain signature
    engine.hall_of_fame = [(5.0, g_old)]
    g_new.fitness = 5.0
    engine._update_hall_of_fame([(1, g_new)])
    assert len(engine.hall_of_fame) == 1 and engine.hall_of_fame[0][1] is g_old


def test_target_survival_raises_success(workdir, capsys):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True, target_survival=0.0001)
    engine.hall_of_fame = []
    pop, config = _population()
    with pytest.raises(SuccessTermination):
        _run_generation(engine, pop, config)
    capsys.readouterr()


def test_stagnation_raises(workdir, capsys):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True)
    engine.hall_of_fame = []
    big = (1e12, -100)    # unbeatable records set long ago
    engine.all_time_best_fitness = engine.all_time_best_median = engine.all_time_survival_rate = big
    pop, config = _population()
    with pytest.raises(StagnationTermination):
        _run_generation(engine, pop, config)
    capsys.readouterr()


def test_infinite_never_stagnates(workdir, capsys):
    engine = Engine(NEAT_CONFIG, False, 1, headless=True, infinite=True)
    engine.hall_of_fame = []
    big = (1e12, -100)
    engine.all_time_best_fitness = engine.all_time_best_median = engine.all_time_survival_rate = big
    pop, config = _population()
    _run_generation(engine, pop, config)
    capsys.readouterr()


def test_visual_generation_renders(workdir, capsys):
    engine = Engine(NEAT_CONFIG, False, 1, headless=False)
    engine.hall_of_fame = []
    pop, config = _population()
    _run_generation(engine, pop, config)
    capsys.readouterr()
    assert engine.screen is not None
    assert CarAI.NN_POSITION[0] == engine.WIDTH - engine.SIDEBAR_WIDTH + 30
