# Architecture

## Overview

```
main.py ──► Engine (src/render/engine.py)
              │  loads config/track_config.json, track image + collision mask,
              │  Hall of Fame (data/hall_of_fame.pkl)
              ▼
            neat.Population.run(run_simulation, max_generations)
              │
              ▼  once per generation
            run_simulation(genomes)
              ├─ CarAI(genomes)                       src/ai/car_ai.py
              │    ├─ GPUBatchInference (all networks) src/gpu/gpu_inference.py
              │    ├─ GPURaycast (sensors)             src/gpu/gpu_raycast.py
              │    └─ VectorizedPhysics (all cars)     src/ai/vector_physics.py
              │
              ├─ loop until all cars dead or MAX_FRAMES reached:
              │    CarAI.compute()  ── raycast → infer → act → physics → fitness
              │    render frame (GUI) or print stats (headless)
              │
              └─ _end_generation(): records, Hall of Fame, termination checks
```

## Module map

| Module | Responsibility |
|---|---|
| `main.py` | CLI parsing and building the `Engine`. |
| `play.py` | Manual driving mode (`HumanCar` with its own handling model). |
| `src/render/engine.py` | Track loading, the NEAT run loop, per-generation stats, Hall of Fame, termination, GUI/terminal output. |
| `src/ai/car_ai.py` | One generation: builds the cars and inference backend, and runs the per-frame pipeline. |
| `src/ai/vector_physics.py` | Authoritative physics state for all cars (NumPy arrays): kinematics, collisions, checkpoints, fitness. |
| `src/gpu/gpu_raycast.py` | Sensor raycasting: a CUDA kernel via CuPy, or a chunked NumPy fallback. |
| `src/gpu/gpu_inference.py` | Batched, exact evaluation of every genome's feed-forward network with PyTorch. |
| `src/render/car.py` | `Car`: per-car render state, plus a scalar physics/reward model used by manual mode, fallbacks and tests. Defines the shared constants (`SENSOR_ANGLES`, sizes, speeds). |
| `src/render/camera.py`, `minimap.py`, `stats_panel.py`, `neural_network/` | GUI pieces. |
| `src/render/track.py` | Holds the visual `surface` and the collision `mask_surface`. |
| `tools/` | Track authoring and validation (see [TRACKS.md](TRACKS.md)). |

## Per-frame pipeline (`CarAI.compute`)

1. **Raycast.** For each alive car, 7 rays are cast at `Car.SENSOR_ANGLES = (-90, -45, -20, 0, 20, 45, 90)`° relative to its heading. Each ray steps 1px at a time from the car centre and stops at the first wall pixel, at the first step off the map, or after 1920 steps. The distances are written straight into the network input buffer.
2. **Inference.** The networks of the alive cars are evaluated in one batched call.
3. **Action.** `argmax` over the 4 outputs gives `0` = turn left, `1` = turn right, `2` = accelerate, `3` = brake.
4. **Physics** (`VectorizedPhysics.step`):
   * Turning changes heading by ±10° per frame. Accelerating adds +1 px/frame, and braking subtracts 1 but never goes below 10 px/frame (`Car.MINIMUM_SPEED`). Cars therefore always move.
   * Position advances by `speed` along the heading.
   * **Death:** any of the 4 corners of the 30×30 collision box is on a wall pixel or off the map, **or** 300 frames pass without crossing a checkpoint (starvation).
5. **Fitness** (`update_fitness`), see below.
6. **Sync.** Array state is copied onto the `Car` objects of cars alive this frame, for rendering. Each surviving car's fitness is written to its genome, and the best car (first index on ties) gets its network visualiser.

### Conventions

* `angle` is in degrees, counter-clockwise, with **0 = facing +x (right) and 90 = facing up**. Screen y grows downward, so motion uses `radians(360 - angle)`.
* `start_pos` in the track config is the **car centre**. The physics stores the top-left corner (`centre - 15`).
* **Wall rule:** a mask pixel is a wall if its **red channel > 200**. White and pure red are walls; black, grey and blue are road.

## Fitness

With checkpoints (the normal case):

```
crossing checkpoint k:    cp_fitness += 1000 + max(0, 500 - frames_since_previous_checkpoint)
every frame:              current = cp_fitness - distance(car centre, next checkpoint segment)
fitness (genome):         max(current) over the car's life   (never decreases; frozen at death)
```

A car must cross checkpoints **in order**. Only the next checkpoint (`cp_index % N`) is tested, and the index wraps around, so laps keep scoring.

Without checkpoints (a fallback ported from `Car.get_reward`):
`(driven_distance / (diag/2) + sqrt(speed / (diag/100)) - brake_penalty / (diag/1000)) * (1 + min(1, driven_distance / (0.75·diag)))`.

## GPU inference: why it's exact

`GPUBatchInference` packs genome *i* into `W[i, to, from]` with the node layout `[inputs | outputs | hidden]`, and iterates

```
X ← tanh(2.5 · (B + R ⊙ (W X)))       (input columns re-clamped every iteration)
```

This is neat-python's `tanh_activation(bias + response · Σ w·x)`. After *k* iterations, every node whose longest incoming path has ≤ *k* edges holds its exact feed-forward value. The tensor is iterated exactly `max_depth` = the longest path over the whole population, so all outputs are exact. Input-less nodes (neat-python's "bias neurons") settle after one iteration, and dead-end hidden nodes that neat-python prunes cannot reach an output. This assumes every node uses `tanh` / `sum`, as configured in `config/neat_config.ini`. `tests/test_optimisations.py` checks the equivalence on deeply mutated genomes.

## Raycasting

The CUDA kernel uses one thread per (car, ray) and writes `[distances | hit_x | hit_y]` into one buffer, so each frame needs a single device→host copy. The wall map is uploaded once, and again only when a different array is passed (an identity check that holds a reference to the cached array). The NumPy fallback computes the same values, chunked to 1024 rays at a time to bound memory.

## Generation lifecycle (`Engine`)

* **Generation 0 only:** Hall-of-Fame genomes replace the first population members, the population is re-speciated, and the node-id counter is advanced past every id loaded from disk.
* **Episode length:** `max_frames` from the track config, in simulated frames at 60 per virtual second.
* **End of generation** records:
  * *Best fitness*: best genome among **survivors** (a car must survive the timer to set a record).
  * *Peak median*: the highest median fitness of the whole population seen during the generation.
  * *Survival rate*: % of cars alive when the timer ends.
  * *Avg speed*: driven distance / frames alive of the generation's fittest car, in px/s.
* **Hall of Fame:** previous HoF plus all genomes, sorted by fitness (stable, so older entries win ties), de-duplicated by brain signature (enabled connection keys and weights rounded to 2 d.p.), top 16 kept. Only genomes that make the cut are deep-copied. Saved every generation.
* **Termination:**
  * the generation limit (1000, or none with `--infinite`);
  * `--target-survival` reached (`SuccessTermination`);
  * the fitness, median and survival records all older than 50 generations (`StagnationTermination`, disabled by `--infinite`).

  In each case the best genome is saved to `data/best_car_brain.pkl`.

## NEAT configuration highlights (`config/neat_config.ini`)

| Setting | Value | Note |
|---|---|---|
| `pop_size` | 1024 | Cars per generation. |
| `num_inputs` / `num_outputs` | 7 / 4 | Must match `Car.SENSOR_ANGLES` and the 4 actions. |
| `feed_forward`, `initial_connection` | True, full | Required by `GPUBatchInference`. |
| `activation_options`, `aggregation_options` | tanh, sum | Required by `GPUBatchInference`. |
| `elitism`, `species_elitism` | 2, 2 | |
| `max_stagnation` | 20 | Species-level (neat-python), separate from the Engine's 50-generation stop. |
| `fitness_threshold` | 1e8 | Effectively never reached; training ends by the other conditions. |

## GUI

The camera follows the fittest alive car. The main view shows the track, numbered checkpoints (the start line is green), the cars (culled to the viewport), and the leader's sensor rays. A minimap with car dots (green alive, red crashed) sits at the top-left. The right sidebar holds the stats panel and the leader's network: inputs are shaded by sensor distance, and the chosen action is highlighted.
