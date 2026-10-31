# NEAT Cars AI

A GPU-accelerated neuroevolution (NEAT) environment where AI cars learn to drive around custom-built, highly complex race tracks using raycast sensors.

Each car sees the world through 7 distance sensors and picks one of 4 actions (left, right, accelerate, brake) every frame. A population of 1024 networks is evolved with [neat-python](https://github.com/CodeReclaimers/neat-python), and fitness is measured by progress through the track's checkpoints.

📖 **Docs:** [User guide](docs/USER_GUIDE.md) · [Architecture & internals](docs/ARCHITECTURE.md) · [Track authoring guide](docs/TRACKS.md) · [Code audit: bugs fixed & optimisations](docs/AUDIT.md)

## Setup

Python 3.11+ is required. Install the dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for the test suite
```

* **GPU acceleration is optional.** Without PyTorch the networks run through neat-python, and without CuPy the sensors are raycast with NumPy. Install the CuPy wheel that matches your CUDA version (`cupy-cuda12x`, `cupy-cuda11x`, ...).
* **pygame must have PNG support.** Check with `python -c "import pygame; print(pygame.image.get_extended())"`. If it prints `False` (it can happen with source builds on new Python versions), install `pygame-ce` instead.
* If your Python install changes (for example a version manager moves "latest"), recreate the venv: it pins the interpreter it was created with.

## Running the Simulation

```bash
python main.py
```
*Trains with the GUI on the `current_track` from `config/track_config.json`.*

### Command Line Arguments

| Flag | Meaning |
|---|---|
| `--track <id>` | Track index in `config/track_config.json` (see `python main.py --help`). |
| `--headless` | No window: progress is printed to the terminal, and generations run much faster. |
| `--visual` | GUI mode (the default). |
| `--infinite` | No generation limit and no 50-generation stagnation stop. Good for overnight runs. |
| `--target-survival <pct>` | Stop once a generation's survival rate reaches this percentage, and save the best brain. |

Default tracks:

| ID | Name | Notes |
|---|---|---|
| `0` | Massive Loop | Long straights, smooth curves. |
| `1` | Intersection Loop | A 90° four-way crossroad that tests intersection handling. |
| `2` | Super Hard | Narrow roads, two hairpins, two intersections, physical obstacles. |
| `3` | NoLoopNoObstacle | Imported real-world-style circuit. Its checkpoint data needs cleanup (see [AUDIT](docs/AUDIT.md#known-issues)). |

### Examples

```bash
python main.py --track 1                                  # watch the AI learn the intersection map
python main.py --track 2 --headless --target-survival 40  # train fast until 40% survive
python main.py --track 2 --headless --infinite            # leave it running overnight
```

### Outputs

| File | Written |
|---|---|
| `data/hall_of_fame.pkl` | After every generation: the 16 best distinct genomes ever seen. Loaded at start-up and injected into generation 0. |
| `data/best_car_brain.pkl` | When training ends (generation limit, target reached, or stagnation). |
| `data/neat_checkpoints/` | GUI mode only: neat-python checkpoints every 5 generations. |

## Manual Driving Mode

```bash
python play.py --track 3           # god mode (no crashes)
python play.py --track 2 --crash   # with wall collisions; press R to respawn
```
* **W / A / S / D** to drive, and the **scroll wheel** to zoom.
* The sidebar shows your fitness and checkpoint count, scored the same way as the AI. The next checkpoint is highlighted in yellow.

## Track Creation & Tools

See the **[track authoring guide](docs/TRACKS.md)** for the file formats and the full workflow.

| Tool | Purpose |
|---|---|
| `tools/track_editor.py` | Draw an 8000×8000 track from scratch with Catmull-Rom splines; auto-generates checkpoints and registers the `custom` track. |
| `tools/manual_checkpoint_tool.py <image>` | Place checkpoints by hand on any track image (WASD pan, +/- length, hold right-click to rotate, left-click to place, U undo, **ENTER** save). |
| `tools/auto_import.py <image>` | Click along the road; nodes snap to the centre and wall-to-wall checkpoints are generated automatically. |
| `tools/validate_track.py [ids...]` | Checks registered tracks: start pose vs. walls, checkpoint gaps, crossed or oversized checkpoints. **Run it after creating a track.** |
| `tools/create_*.py` | Scripts that generated the built-in tracks. |

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
The suite runs headless (SDL dummy driver). The GPU tests need CUDA and fail without it.

## Technical Architecture & Optimizations

* **Vectorized Physics:** All cars live in flat NumPy arrays (`src/ai/vector_physics.py`), and one frame for the whole population is a handful of array operations.
* **GPU Vision:** A custom CuPy/CUDA kernel (`src/gpu/gpu_raycast.py`) marches all `cars × 7` rays in parallel, with one thread per ray and a single device→host copy per frame.
* **GPU Brains:** All networks of a generation are packed into one dense tensor and evaluated with batched matmuls (`src/gpu/gpu_inference.py`). This is mathematically exact versus neat-python, and runs only as many iterations as the deepest network needs.
* **Subsurface Rendering:** Huge maps are never scaled in full. Only the visible viewport is drawn, and off-screen cars are culled.
* **Data-Driven Configuration:** Every per-track setting lives in `config/track_config.json`.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
