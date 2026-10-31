"""Sanity-check the tracks registered in config/track_config.json.

    python tools/validate_track.py            # all tracks
    python tools/validate_track.py 2 3        # only track ids 2 and 3

Checks, using the same wall rule as training (red channel > 200 on the mask):
  * the image / mask / checkpoints files exist and load
  * the start pose does not overlap a wall, and the road ahead is not blocked
  * gaps between consecutive checkpoint midpoints (large gaps starve cars,
    which die after 300 frames without a checkpoint)
  * checkpoint midpoints on a wall, abnormally long checkpoints, and consecutive
    checkpoints that cross each other (usually checkpoints laid ALONG the road)

Exit code is 1 if any ERROR was found.
"""

import json
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.render.car import Car  # noqa: E402

CONFIG_PATH = "config/track_config.json"
MAX_GAP = 800              # px between consecutive checkpoint midpoints
MIN_CLEAR_AHEAD = 60       # px of road required in front of the spawn point
LONG_FACTOR = 3.0          # checkpoint longer than this x the median is suspicious


def wall_map(path):
    surf = pygame.image.load(path)
    packed = pygame.surfarray.array3d(surf)          # (W, H, 3)
    return (packed[:, :, 0] > 200).T                 # (H, W) True = wall


def _segments_cross(a, b):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    p1, p2, p3, p4 = (a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def _is_wall(wall, x, y):
    h, w = wall.shape
    return not (0 <= x < w and 0 <= y < h) or bool(wall[int(y), int(x)])


def validate(key, entry):
    errors, warnings = [], []
    for field in ("image", "checkpoints"):
        if field not in entry or not os.path.exists(entry[field]):
            errors.append(f"missing {field} file: {entry.get(field)}")
    mask_path = entry.get("mask", entry.get("image"))
    if errors:
        return errors, warnings
    try:
        wall = wall_map(mask_path)
    except pygame.error as e:
        return [f"cannot load mask {mask_path}: {e}"], warnings

    # --- Start pose (start_pos is the car centre; corners as in VectorizedPhysics) ---
    sx, sy = entry.get("start_pos", [200, 950])
    ang = entry.get("start_angle", 0)
    r = math.radians(360 - ang)
    c, s = math.cos(r), math.sin(r)
    hx, hy = Car.CAR_SIZE_X / 2, Car.CAR_SIZE_Y / 2
    if any(_is_wall(wall, sx + dx * c - dy * s, sy + dx * s + dy * c)
           for dx in (-hx, hx) for dy in (-hy, hy)):
        errors.append(f"start pose {[sx, sy]} @ {ang} deg overlaps a wall (every car dies on frame 1)")
    clear = 0
    while clear < 2000 and not _is_wall(wall, sx + c * clear, sy + s * clear):
        clear += 1
    if clear < MIN_CLEAR_AHEAD:
        errors.append(f"only {clear}px of road ahead of the start pose (heading {ang} deg)")

    # --- Checkpoints ---
    with open(entry["checkpoints"]) as f:
        cps = json.load(f)
    if not cps:
        warnings.append("no checkpoints (training falls back to distance-based fitness)")
        return errors, warnings

    mids = [((a[0] + a[2]) / 2, (a[1] + a[3]) / 2) for a in cps]
    lengths = [math.hypot(a[2] - a[0], a[3] - a[1]) for a in cps]
    median_len = float(np.median(lengths))
    n = len(cps)
    for i in range(n):
        j = (i + 1) % n
        gap = math.hypot(mids[j][0] - mids[i][0], mids[j][1] - mids[i][1])
        if gap > MAX_GAP:
            errors.append(f"gap {i}->{j} is {gap:.0f}px (> {MAX_GAP})")
        if _is_wall(wall, *mids[i]):
            warnings.append(f"checkpoint {i} midpoint {tuple(int(v) for v in mids[i])} is on a wall")
        if lengths[i] > LONG_FACTOR * median_len:
            warnings.append(f"checkpoint {i} is {lengths[i]:.0f}px long (median {median_len:.0f}px)")
        if n > 2 and _segments_cross(cps[i], cps[j]):
            warnings.append(f"checkpoints {i} and {j} cross each other")

    d0 = Car.distance_to_line((sx, sy), cps[0][:2], cps[0][2:])
    if d0 > MAX_GAP:
        warnings.append(f"start is {d0:.0f}px from checkpoint 0")
    return errors, warnings


def main(argv):
    pygame.init()
    with open(CONFIG_PATH) as f:
        tracks = json.load(f)["tracks"]
    keys = list(tracks)
    ids = [int(a) for a in argv] if argv else range(len(keys))

    any_error = False
    for i in ids:
        key = keys[i]
        errors, warnings = validate(key, tracks[key])
        status = "FAIL" if errors else ("WARN" if warnings else "OK")
        print(f"[{status}] track {i}: {tracks[key].get('name', key)} ({key})")
        for e in errors:
            print(f"    ERROR: {e}")
        for w in warnings[:15]:
            print(f"    warn:  {w}")
        if len(warnings) > 15:
            print(f"    ... and {len(warnings) - 15} more warnings")
        any_error |= bool(errors)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
