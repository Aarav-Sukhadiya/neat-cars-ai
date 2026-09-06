# NEAT Cars AI

A GPU-accelerated neuroevolution (NEAT) environment where AI cars learn to drive around custom-built, highly complex race tracks using raycast sensors.

## Setup

Ensure you have Python 3.11+ installed, then install the dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Simulation

You can start the simulation by running the `main.py` script. We have built several custom flags to control how the AI trains and which track it runs on.

### Basic Usage
```bash
python main.py
```
*Runs the simulation with the UI enabled on the default track.*

### Command Line Arguments

Tracks are dynamically loaded from `config/track_config.json`. By default, the following tracks are available:
* `--track <id>`: Select the track to train on. 
  * `0` = **Massive Loop**. Long straights, smooth curves.
  * `1` = **Intersection Loop**. Features a 90-degree four-way crossroad to test intersection logic.
  * `2` = **Super Hard**. Extremely narrow roads, two hairpins, two intersections, and physical obstacles.
  * `3` = **NoLoopNoObstacle**. A large imported real-world style circuit.
* `--headless`: Run the simulation without the PyGame GUI. This relies entirely on GPU matrix math and removes rendering bottlenecks, allowing you to train generations in seconds instead of minutes.
* `--infinite`: Bypasses the maximum simulation limit and disables the 50-generation stagnation kill-switch. Use this to let the AI brute-force complex maps overnight.
* `--target-survival <float>`: Sets an early "Win Condition". If the generation's survival rate reaches this percentage (e.g., `40`), the engine will terminate and save the best brains.

### Helpful Examples

**1. Watch the AI learn on the new Intersection map:**
```bash
python main.py --track 1
```

**2. Train the AI as fast as possible on the Super Hard map until 40% survive:**
```bash
python main.py --track 2 --headless --target-survival 40
```

**3. Leave the AI to train infinitely overnight without giving up:**
```bash
python main.py --track 2 --headless --infinite
```

## Manual Driving Mode

Want to test the tracks yourself? Run the manual play script:
```bash
python play.py --track 3
```
* Use **W, A, S, D** to drive the car.
* Use **Scroll Wheel** to zoom in and out of the massive tracks seamlessly.

## Track Creation & Tools

We provide several robust tools in the `tools/` directory to create and import custom maps:

### 1. Manual Checkpoint Editor
Have an image of a race track (e.g., an F1 circuit) that you want the AI to learn? Import it and manually place the checkpoints:
```bash
python tools/manual_checkpoint_tool.py assets/tracks/your_track.png
```
* **WASD** to pan the camera across huge maps.
* **+ / -** to adjust checkpoint length.
* **Hold Right Click** to perfectly rotate the checkpoint line.
* **Left Click** to stamp the checkpoint.
* **U** to undo.
* **ENTER** to save the track. This automatically generates the JSON data and registers your new track into `config/track_config.json`!

### 2. Track Editor
Draw massive 8000x8000 tracks entirely from scratch using smooth Catmull-Rom splines:
```bash
python tools/track_editor.py
```
Click to place nodes, and the tool will render the curved roads and auto-generate geometrically perfect perpendicular checkpoints.

### 3. Auto Importer
Attempts to automatically crawl a track image (BFS algorithm) to find the centerline and generate checkpoints via raycasting.
```bash
python tools/auto_import.py assets/tracks/your_track.png
```

## Technical Architecture & Optimizations

* **Vectorized Physics:** Pygame object loops were replaced with highly parallel NumPy matrix mathematics (`src/ai/vector_physics.py`), evaluating thousands of cars instantly.
* **GPU Vision:** Sensor raycasting uses a custom CuPy/CUDA GPU kernel (`src/gpu/gpu_raycast.py`). It avoids CPU pixel-walking by processing all `Cars x Rays` on the GPU simultaneously.
* **Subsurface Rendering:** Huge 8000x8000 maps are not scaled in full; instead, a Pygame Subsurface viewport follows the car, enabling buttery-smooth 60 FPS panning and zooming. 
* **Data-Driven Configuration:** Track logic and starting angles are never hardcoded. All map settings exist in `config/track_config.json`.
