"""Train NEAT cars. See README.md / docs/ for details.

    python main.py [--track ID] [--headless | --visual] [--infinite] [--target-survival PCT]
"""

import argparse
import json
import sys

from src.render.engine import Engine, TRACK_CONFIG_PATH

NEAT_CONFIG_PATH = "config/neat_config.ini"
ENABLE_NEAT_REPORTERS = True   # neat-python stdout stats + checkpoints (GUI mode only)
MAX_SIMULATIONS = 1000         # generation limit (ignored with --infinite)


def _track_help() -> str:
    try:
        with open(TRACK_CONFIG_PATH) as f:
            tracks = json.load(f).get("tracks", {})
        return "Track ID: " + ", ".join(f"{i}={t.get('name', k)}" for i, (k, t) in enumerate(tracks.items()))
    except (OSError, ValueError):
        return "Track ID to load"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NEAT Cars AI Simulation.")
    parser.add_argument('--track', type=int, default=None,
                        help=_track_help() + " (default: current_track in track_config.json)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--headless', action='store_true', help="Run without a GUI (much faster).")
    mode.add_argument('--visual', action='store_false', dest='headless', help="Run with the PyGame GUI (default).")
    parser.add_argument('--infinite', action='store_true',
                        help="Train with no generation limit and no stagnation termination.")
    parser.add_argument('--target-survival', type=float, default=0.0,
                        help="Stop once a generation's survival rate reaches this percentage.")
    args = parser.parse_args()

    try:
        engine = Engine(NEAT_CONFIG_PATH, ENABLE_NEAT_REPORTERS, MAX_SIMULATIONS, headless=args.headless,
                        track_id=args.track, infinite=args.infinite, target_survival=args.target_survival)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
    engine.run()


if __name__ == "__main__":
    main()
