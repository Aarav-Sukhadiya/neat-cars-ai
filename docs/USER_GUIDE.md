# NEAT Cars — User Guide

Last updated: 2026-11-01

## What it is

NEAT Cars evolves neural networks that learn to drive race tracks. Each generation, 1024 cars each see the road through 7 distance sensors and pick one of 4 actions per frame: turn left, turn right, accelerate or brake. Cars that get further through the track's checkpoints score higher, and the best brains breed the next generation.

Quick start, from the project folder:

```bash
pip install -r requirements.txt
python main.py              # watch it learn on the default track
python main.py --headless   # train much faster, stats in the terminal
```

## Installation

You need Python 3.11 or newer. Install into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

| Package | Required? | What it does |
| --- | --- | --- |
| pygame | Yes | Window, drawing, loading track images |
| neat-python | Yes | The evolution algorithm |
| numpy | Yes | Vectorised car physics |
| torch | Optional | Runs all networks at once on the GPU; without it, neat-python runs them on the CPU |
| cupy-cuda12x | Optional | GPU sensor raycasting; without it, NumPy does it on the CPU. Pick the wheel matching your CUDA version |

Two checks before your first run:

- **PNG support:** `python -c "import pygame; print(pygame.image.get_extended())"` must print `True`. If it prints `False`, run `pip install pygame-ce` instead of pygame.
- **Broken venv:** if `python` inside the venv can't find installed packages (e.g. after your Python version manager moved to a newer version), delete `venv/` and recreate it with the commands above.

## Training the AI

Run `python main.py` with any of these flags. `python main.py --help` lists the tracks on your machine.

| Flag | What it does |
| --- | --- |
| `--track <id>` | Pick a track by its number (default: `current_track` in `config/track_config.json`) |
| `--headless` | No window; stats print to the terminal and training runs about 2–10x faster |
| `--visual` | Show the window (the default) |
| `--infinite` | Never stop: no generation limit and no stagnation stop |
| `--target-survival <pct>` | Stop once this % of a generation survives the timer, and save the best brain |

| ID | Track | Episode length | Notes |
| --- | --- | --- | --- |
| 0 | Massive Loop | 60 s | Long straights, smooth curves; good first track |
| 1 | Intersection Loop | 60 s | A four-way crossroad |
| 2 | Super Hard | 120 s | Narrow roads, hairpins, two intersections, obstacles |
| 3 | NoLoopNoObstacle | 120 s | Imported circuit; its checkpoints need re-placing before it trains well |

Common runs:

```bash
python main.py --track 1                                  # watch it learn the crossroad
python main.py --track 2 --headless --target-survival 40  # train fast until 40% survive
python main.py --track 2 --headless --infinite            # leave it overnight
```

**In the window** the camera follows the current leader. You see numbered checkpoints (the start line is green), the leader's sensor rays, a minimap at the top-left, and a sidebar with the leader's network plus live stats: generation, cars alive, time left, best and median fitness, top speed and brain size.

**In headless mode** the terminal shows the same live stats, the previous generation's results, and all-time records with the generation each was set in.

Fitness is checkpoint progress: about +1000 per checkpoint crossed in order, a bonus for crossing quickly, minus the distance to the next one. A car dies when it touches a wall, leaves the map, or goes 5 seconds without a new checkpoint.

## What gets saved, and when training stops

| File | When it's written | Contents |
| --- | --- | --- |
| `data/hall_of_fame.pkl` | After every generation | The 16 best distinct brains ever seen. Loaded at start-up and seeded into the first generation, so training resumes from your best cars |
| `data/best_car_brain.pkl` | When training ends | The single best genome |
| `data/neat_checkpoints/` | Every 5 generations, window mode only | neat-python population checkpoints |

Training stops when any of these happens (in each case the best brain is saved):

1. 1000 generations have run (unless `--infinite`).
2. The `--target-survival` percentage is reached.
3. Best fitness, median fitness and survival rate have all failed to set a new record for 50 generations (unless `--infinite`).

To start completely fresh, delete `data/hall_of_fame.pkl`. Do this when switching to a very different track: saved brains keep their scores from the old track.

## Driving a track yourself

```bash
python play.py --track 3           # god mode: walls don't stop you
python play.py --track 2 --crash   # walls count; press R to respawn
```

| Key | Action |
| --- | --- |
| W | Accelerate |
| S | Brake, then reverse |
| A / D | Steer left / right (flips when reversing) |
| Scroll wheel | Zoom in / out |
| R | Respawn after a crash (`--crash` only) |

The sidebar shows your speed, fitness and checkpoints crossed, scored the same way as the AI. The next checkpoint is drawn in yellow. Use this to test a new track before training on it.

## Creating your own tracks

A track is an image plus an ordered list of checkpoint lines, registered in `config/track_config.json`. On the image, any pixel whose red channel is above 200 is a wall, so draw roads in black or dark colours on white.

Pick one of three ways to make one:

| Tool | Start from | How |
| --- | --- | --- |
| `python tools/track_editor.py` | Nothing | Click to place spline nodes on an 8000x8000 canvas. C toggles a closed loop, +/- changes road width, S saves as the `custom` track with checkpoints generated |
| `python tools/manual_checkpoint_tool.py <image>` | Any track image | WASD pans, +/- sets line length, hold right-click to rotate, left-click places a checkpoint, U undoes, ENTER saves and registers the track |
| `python tools/auto_import.py <image>` | Any track image | Click roughly along the road; points snap to the centre and wall-to-wall checkpoints appear. Z undoes, scroll zooms, right-drag pans, S saves |

Then check and test it:

```bash
python tools/validate_track.py           # fix every ERROR, review the warnings
python play.py --track <id> --crash      # drive it yourself
python main.py --track <id>              # train
```

Rules for checkpoints that train well:

1. Place them in driving order, across the road from wall to wall, never along it.
2. Start the car a little **before** checkpoint 0, not on it. The tools do this for you (40 px behind).
3. Keep gaps between checkpoints well under 800 px. Cars die after 5 seconds without a new one.
4. Don't let neighbouring checkpoints cross each other.
5. At intersections, leave a gap around the crossing so cars can't score the wrong branch.

Each track entry in `config/track_config.json` takes these fields:

| Field | Meaning |
| --- | --- |
| `name` | Display name |
| `image` | Track image path |
| `mask` | Optional separate collision image, so the visual image can have decorations |
| `checkpoints` | Path to the checkpoint JSON: a list of `[x1, y1, x2, y2]` |
| `max_frames` | Episode length; 60 frames = 1 second |
| `start_pos` | Where the car's centre spawns, in pixels |
| `start_angle` | Spawn heading in degrees: 0 = right, 90 = up, 180 = left, 270 = down |

Track numbers follow the order of entries in that file.

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite runs without a display and never touches your real `data/` files. The tests in `tests/test_gpu.py` need a CUDA GPU and fail without one; run `python -m pytest -q --ignore=tests/test_gpu.py` on a machine without a GPU.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Could not load track image ... no PNG support` | Your pygame can't read PNGs: `pip uninstall pygame && pip install pygame-ce` |
| `Invalid track id N` | Use a number listed by `python main.py --help` |
| `ModuleNotFoundError` inside the venv | The venv points at a Python that changed; delete `venv/` and recreate it |
| Every car dies in the first second | The start pose is in a wall or facing one. Run `python tools/validate_track.py <id>` and fix `start_pos` / `start_angle` |
| Cars drive but fitness stays near 0 | They aren't crossing checkpoint 0. Make sure the car spawns before it, and run the validator |
| `[GPU] ... (CPU)` at start-up | torch or cupy isn't installed or can't see CUDA. Training still works, just slower |
| Training stops on its own | It hit the 50-generation stagnation stop; add `--infinite` to keep going |
