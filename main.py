# ------------------ IMPORTS ------------------

import argparse
from src.render.engine import Engine


# ------------------ GLOBAL VARIABLES ------------------


NEAT_CONFIG_PATH = "config/neat_config.ini"
RAY_CAST = True
MAX_SIMULATIONS = 1000


# ------------------ MAIN FUNCTION ------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NEAT Cars AI Simulation.")
    parser.add_argument('--track', type=int, default=0, help="Track ID to load (0 = default)")
    parser.add_argument('--headless', action='store_true', help="Run the simulation without a GUI (faster).")
    parser.add_argument('--visual', action='store_false', dest='headless', help="Run the simulation with PyGame GUI.")
    # Default to headless if they don't specify, or default to visual.
    # The user asked for "option of headless mode and visual model by --visual or --headless flags"
    # Let's set the default to visual since games usually default to visual.
    parser.set_defaults(headless=False)
    args = parser.parse_args()

    engine = Engine(NEAT_CONFIG_PATH, RAY_CAST, MAX_SIMULATIONS, headless=args.headless, track_id=args.track)
    engine.run()


# ------------------ MAIN CALL ------------------


if __name__ == "__main__":
    main()
