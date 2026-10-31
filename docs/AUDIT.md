# Code audit (2026-11-01)

A full pass over the project covering bugs, lossless performance work, and cleanup. Every change was verified with the test suite (47 → 72 tests) and a determinism harness, described below.

## How "lossless" was verified

A harness ran a seeded, mutated 1024-genome population for 600 frames on each of the 4 tracks, on both the original and the modified code. It recorded every frame's position, heading, speed, alive mask, fitness and checkpoint index for every car, plus the final genome fitnesses, the best-network choice, and the Hall of Fame.

* **After the optimisation pass, every recorded array was bit-identical** on all 4 tracks (the GPU paths, CUDA on an RTX 5070, were included).
* After the bug-fix pass, the per-frame physics on tracks 0–2 was **still bit-identical**. Only the intended things changed: generation length, the avg-speed stat, and track 3's spawn.

## Bugs fixed

| # | Severity | Where | Bug | Fix |
|---|---|---|---|---|
| 1 | **High** | `engine.py` | `frame_count` was incremented **twice per frame**, so every generation ran only half its `max_frames` (e.g. 30 s instead of 60 s), and the on-screen timer counted down at 2× speed. | One increment per frame. Training episodes are now as long as configured. |
| 2 | **High** | `track_config.json` | Track 3 spawned at `[100, 400]` heading `0°`, overlapping the left wall (11px clearance for a 15px half-width) and facing across the road. **Every car died within 5 frames**, so the track was untrainable. | Spawn at `[107, 476]` heading `70°`: 40px behind checkpoint 0, pointing down the road. |
| 3 | **High** | `manual_checkpoint_tool.py`, track 3 | The spawn was set to checkpoint 0's **midpoint**. A centre that starts on the line never registers crossing it, so cars chase a checkpoint behind them and the whole population starves with fitness 0. | The tool now spawns 40px behind checkpoint 0. Track 3 went from best fitness 0 to ≈4300 in one generation. |
| 4 | **High** | `engine.py` | If the track image failed to load, the engine **silently trained on a blank all-wall 4000×4000 track** (every car dies on frame 1, forever). | Raises a clear error, with a hint when pygame lacks PNG support. |
| 5 | Medium | `vector_physics.py` / `engine.py` | The "Avg Speed" stat and record were **always 0**: `driven_distance` and `frames_alive` were never synced from the physics arrays. | Synced, and the stats read the arrays directly. |
| 6 | Medium | `manual_checkpoint_tool.py` | **S** pans the camera down (held) *and* triggered "save & exit" on key-down, so panning down saved half-finished work and quit. | Save is ENTER only. |
| 7 | Medium | `track_editor.py` | Saving patched `start_pos`/`start_angle` into whichever entry was named `custom`, which was **the NoLoopNoObstacle track**, corrupting its spawn. If no `custom` entry existed, the new track was never registered. | The editor creates or updates its own `custom` entry pointing at `custom.png`/`custom.json`. Track 3's key was renamed to `no_loop`. |
| 8 | Medium | `vector_physics.py` | Off-map corners were clamped onto the edge pixel, so on an image with road touching the border, cars could drive off the map and never die. | Leaving the image counts as a crash (as in `Car.check_collision`). |
| 9 | Medium | `vector_physics.py` | Tracks without checkpoints got **fitness 0 for every genome** (the fallback was a `pass`). | A vectorised port of `Car.get_reward`'s distance/speed fallback. |
| 10 | Medium | `gpu_raycast.py` | The track cache compared `id()` of arrays. A freed array's id can be reused by a new one, which leaves a **stale wall map** on the GPU. | Holds a reference and compares with `is`. |
| 11 | Low | `engine.py` | `pygame.display.Info()` reports the *window* size once a window exists, which could give a negative sidebar width (crash). | Uses the desktop size, with a minimum of 800×600. |
| 12 | Low | `play.py` | A `SysFont` was created **every frame** and overwrote the HUD font (HUD text size changed with zoom). | Fonts are cached. |
| 13 | Low | `play.py` | Crashed when a track has no `checkpoints` entry. | Guarded. |
| 14 | Low | `play.py` | Manual-mode fitness never progressed (checkpoints were never checked). | Checkpoints are scored, and the next one is highlighted. |
| 15 | Low | `play.py` | `start_pos` was used as the top-left corner (training uses the centre). | Same convention as training. |
| 16 | Low | `play.py` | A complex number came out of `sqrt` when reversing (no-checkpoint reward). | Speed is clamped at 0 in the reward. |
| 17 | Low | `node.py` | `Node.inputs` placeholder had 5 entries for 7 sensors (`IndexError` if drawn before the first update). | 7 entries. |
| 18 | Low | `auto_import.py` | Undo was labelled Ctrl+Z but bound to Z, and the success message hard-coded `--track 4`. | Fixed both. |
| 19 | Low | `main.py` | An invalid `--track` silently fell back to the current track, and `--headless --visual` were not exclusive. | Errors out; the flags are mutually exclusive. |
| 20 | Low | `colors.py` | `GREEN` and `BLUE` were defined twice. | De-duplicated; the effective values are unchanged. |
| 21 | Low | tests | `test_engine.py` swallowed every exception (`try/except: pass`) and **overwrote the real `data/hall_of_fame.pkl`**. | Rewritten against a temporary synthetic track, with real assertions. |

Every bug above that is reachable from a unit test has a regression test. Each test was mutation-checked: it fails when the bug is re-introduced.

## Optimisations (lossless)

Measured on an RTX 5070 Laptop GPU with a 1024-car population on tracks 0 and 2:

| | Before | After |
|---|---|---|
| Generation setup (`CarAI.__init__`) | 27.5 ms | **12.6 ms** (2.2×) |
| `CarAI.compute()` per frame | 2.06–2.14 ms | **1.59–1.68 ms** (≈1.3×) |
| Full headless generation (same frame count) | 737–784 ms | **332–352 ms** (≈2.2×) |

What changed:

* **No redundant network builds.** `FeedForwardNetwork.create` ran for every genome even though the GPU path never used the result. It is now built only for the CPU fallback.
* **Exact iteration count.** GPU inference iterated `hidden_nodes + 1` times. It now iterates the true longest path, which is the minimum that gives exact results.
* **Alive-only inference.** Only alive cars are evaluated (bit-identical outputs, verified).
* **One transfer per frame.** The raycaster does one device→host copy instead of three, and caches the sensor offsets.
* **No per-frame Python lists.** Building `[[(x, y), d], ...]` for all 7000+ rays every frame is gone. Hit points stay in arrays, and lists are built only for the car being drawn.
* **Vectorised bookkeeping.** Best-car tracking, metrics and camera targeting are array ops. Previously this was Python loops over all cars per frame, including `Car.get_reward()` for every car just to aim the camera, and building throw-away NN visualisers.
* **Cheaper Hall of Fame.** It used to deep-copy all 1024 genomes every generation and keep 16. Now it deep-copies only the genomes that make the cut.
* **Faster terminal output.** The headless display redraws with an ANSI escape plus one write, instead of spawning `clear` 15 times a second.
* **GUI.** Checkpoint number labels are rendered once and cached (they used to be re-rendered every frame), and off-screen cars are culled.
* **CPU raycast fallback** is chunked: about 300 MB of temporaries per frame became a few MB, with identical output.
* **`auto_import.py`** builds its road mask with one array op instead of a `get_at()` per pixel (seconds to minutes on big maps).

## Cleanup

* Engine split into focused methods. The dead UI-state machine (`draw`, `handle_drawing_track`, `instructions`, brush sizing, an unused `Car`) is gone, and the termination exceptions are module-level classes.
* Sensor angles and ray length are defined once (`Car.SENSOR_ANGLES`, `Car.SENSORS_DRAW_DISTANCE`) and shared with the GPU code. Fitness constants are named.
* Removed dead code: `Car.update_sprite`, `update_adaptive_parameters`, `inject_sensors`, unused attributes, the never-set minimap "rendered as dead" flag, and unused imports.
* Removed `tools/checkpoint_editor.py`: it read `assets/track.png` / `data/checkpoints.json`, which don't exist, and is superseded by `manual_checkpoint_tool.py`.
* Module and function docstrings throughout.
* `.gitignore` now covers `.coverage`, `htmlcov/` and neat checkpoints, which are written to `data/neat_checkpoints/` instead of the repo root. Added `requirements-dev.txt` and `tests/conftest.py` (headless SDL).

## Known issues

These were deliberately **not** changed because they would alter training behaviour or need design decisions:

1. **Track 3's checkpoint data is low quality.** `python tools/validate_track.py 3` reports about 70 warnings: many consecutive checkpoints cross each other, and several (e.g. #2, #5) run *along* the road and are 3× longer than usual. Re-place them with `tools/manual_checkpoint_tool.py assets/tracks/no_loop.png`.
2. **Network inputs are raw pixel distances (0–1920).** With `tanh` and weights around ±1, most first-layer activations saturate. Normalising them (e.g. `distance / 1920`) would likely speed up learning a lot, but it changes behaviour and makes existing Hall-of-Fame brains obsolete.
3. **There is no fitness gradient before the first checkpoint.** `max_fitness` starts at 0 while `cp_fitness - distance` is negative until a checkpoint is crossed, so approaching checkpoint 0 earns nothing. Spawning near checkpoint 0 hides this.
4. **The collision box is 30×30 with corners at ±15px**, but the drawn car uses corners at radius 15 and ±30°. The drawn car is smaller than its hitbox.
5. **Hall-of-Fame fitness values are not track-specific.** Elites trained on one track are ranked by their old scores when you switch tracks.
6. **Your `venv/` is broken.** It points at mise's `python/latest`, which is now 3.15, while the packages are installed for 3.14. Its pygame build also lacks PNG support (`pygame.image.get_extended() == False`). Recreate it (see the README).
