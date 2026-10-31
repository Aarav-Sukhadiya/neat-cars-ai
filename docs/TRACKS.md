# Track authoring guide

## Files that make up a track

| File | Format |
|---|---|
| `assets/tracks/<name>.png` | Track image. **Collision rule:** a pixel is a wall when its red channel is > 200. Use black or dark colours for road and white for walls. |
| *(optional)* a separate mask image | Same size as the image, used only for collisions. Lets the visual image contain red or white decorations that would otherwise count as walls. |
| `data/tracks/<name>.json` | Checkpoints: a JSON list of `[x1, y1, x2, y2]` line segments **in driving order**. Checkpoint 0 is the start line. |
| entry in `config/track_config.json` | See below. |

## `config/track_config.json`

```json
{
  "current_track": "massive_loop",
  "tracks": {
    "massive_loop": {
      "name": "Massive Loop",
      "image": "assets/tracks/massive_loop.png",
      "checkpoints": "data/tracks/massive_loop.json",
      "max_frames": 3600,
      "start_pos": [200, 950],
      "start_angle": 0
    }
  }
}
```

| Key | Meaning |
|---|---|
| `current_track` | Used when `--track` is not given. |
| `tracks` | Order matters: `--track N` selects the N-th entry. |
| `image` | Visual image (required). |
| `mask` | Collision image (optional, defaults to `image`). |
| `checkpoints` | Checkpoint JSON path. |
| `max_frames` | Episode length in frames (60 frames = 1 virtual second). |
| `start_pos` | Spawn point = the **car centre**, in image pixels. |
| `start_angle` | Spawn heading in degrees, counter-clockwise: `0` = right, `90` = up, `180` = left, `270` = down. |

## Rules for good checkpoints

The training is only as good as the checkpoints, and `tools/validate_track.py` checks all of these rules:

1. **Spawn before checkpoint 0, not on it.** A car whose centre starts exactly on a checkpoint line never registers crossing it, and the whole population starves. The tools spawn 40px behind checkpoint 0.
2. **The start pose must be clear.** The 30×30 car box must not touch a wall, and the heading must point down the road. Otherwise every car dies on frame 1.
3. **Lay checkpoints across the road, wall to wall.** A checkpoint that runs *along* the road can be crossed sideways, or never.
4. **Keep the gaps short.** Cars die after 300 frames (5 s) without a new checkpoint. At the minimum speed of 10 px/frame that is 3000px, but keep gaps well under 800px.
5. **Consecutive checkpoints must not cross each other.**
6. **At intersections**, leave a gap around the crossing (see `INTERSECTIONS` in `tools/create_super_hard.py`). Otherwise cars can score the wrong branch.

## Workflow

```bash
# A) Draw a new track from scratch (8000x8000, splines). S saves it as the "custom" track.
python tools/track_editor.py

# B) Use an existing image, placing checkpoints by hand
python tools/manual_checkpoint_tool.py assets/tracks/my_track.png

# B') ...or semi-automatically: click along the road; checkpoints span wall to wall
python tools/auto_import.py assets/tracks/my_track.png

# Then always:
python tools/validate_track.py          # fix every ERROR; review the warnings
python play.py --track <id> --crash     # drive it yourself
python main.py --track <id>
```
