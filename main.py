"""Jeu de Hex à deux joueurs ou contre un ordinateur à règles."""

import argparse
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main():
    parser = argparse.ArgumentParser(description="Hex — jeu de connexion")
    parser.add_argument("--size", type=int, choices=(3, 5, 7, 9, 11), default=9,
                        help="taille du plateau (9 par défaut)")
    parser.add_argument("--ai", action="store_true",
                        help="jouer contre l'ordinateur à règles")
    args = parser.parse_args()
    try:
        from hexgame.app import HexApp
    except ModuleNotFoundError as error:
        if error.name != "pygame":
            raise
        parser.exit(1, "Pygame est absent. Installez-le avec :\n"
                    "  python -m pip install -r requirements.txt\n")
    HexApp(size=args.size, versus_ai=args.ai).run()


if __name__ == "__main__":
    main()
