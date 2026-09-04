# NEAT Cars AI

A GPU-accelerated neuroevolution (NEAT) environment where AI cars learn to drive around custom-built, highly complex race tracks using raycast sensors.

## Setup

Ensure you have Python 3.11 installed, then install the dependencies into a virtual environment:

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

* `--track <id>`: Select the track to train on. 
  * `0` = **Massive Loop** (Default). Long straights, smooth curves.
  * `1` = **Intersection Loop**. Features a 90-degree four-way crossroad to test intersection logic.
  * `2` = **Super Hard**. Extremely narrow roads, two hairpins, two intersections, and physical obstacles.
* `--headless`: Run the simulation without the PyGame GUI. This relies entirely on GPU matrix math and removes rendering bottlenecks, allowing you to train generations in seconds instead of minutes.
* `--infinite`: Bypasses the maximum simulation limit and disables the 50-generation stagnation kill-switch. Use this to let the AI brute-force complex maps overnight.
* `--target-survival <float>`: Sets an early "Win Condition". If the generation's survival rate reaches this percentage (e.g., `40`), the engine will triumphantly terminate and save the best brains.

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

## How it Works

* **Hardware:** Each car is equipped with a 7-sensor radar array `[-90°, -45°, -20°, 0°, 20°, 45°, 90°]` that casts rays to detect walls and obstacles.
* **Brain:** The NEAT algorithm evolves the topology of the neural networks, adding hidden nodes and connections to process sensor data into steering and acceleration commands.
* **GPU Pipeline:** Raycasting and neural network inference are batched and executed on the GPU using `cupy` and `torch`, allowing 256+ cars to evaluate simultaneously in a fraction of a millisecond per frame.
