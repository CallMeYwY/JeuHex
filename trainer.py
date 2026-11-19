"""Centre d'entraînement des réseaux de neurones pour le Hex.

    python3 trainer.py                                  interface graphique
    python3 trainer.py --console --reseau exemple_mlp --taille 7 --adversaire regles
    python3 trainer.py --generer 500 --taille 9         parties pré-jouées par l'ordi à règles
    python3 trainer.py --liste                          réseaux et fichiers de parties
"""

import argparse
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MIN_SIZE, MAX_SIZE = 3, 25


def board_size(text):
    value = int(text)
    if not (MIN_SIZE <= value <= MAX_SIZE) or value % 2 == 0:
        raise argparse.ArgumentTypeError(
            "la taille doit être impaire, entre {} et {}".format(MIN_SIZE, MAX_SIZE))
    return value


def console(args):
    from hextrainer.session import OPPONENTS, Session, TrainConfig

    last = [0.0]

    def emit(kind, data):
        if kind == "hello":
            print("Réseau « {} », plateau {}×{}, adversaire : {}. Poids {}.".format(
                args.reseau, args.taille, args.taille, OPPONENTS[args.adversaire],
                "repris de " + data["dossier"] if data["charge"] else "neufs"))
        elif kind == "stats" and time.monotonic() - last[0] > 2:
            last[0] = time.monotonic()
            metrics = "  ".join("{} {:.3f}".format(k, v) for k, v in data["mesures"].items())
            print("{:>8} parties  {:6.1f}/s  {}".format(data["parties"], data["parties_par_s"],
                                                       metrics))
        elif kind == "eval":
            print("  évaluation : {:.0%} contre l'ordi à règles, {:.0%} contre le hasard".format(
                data.get("vs_regles", 0), data.get("vs_aleatoire", 0)))
        elif kind == "checkpoint":
            print("  sauvegardé dans", data["dossier"])

    config = TrainConfig(network=args.reseau, size=args.taille, opponent=args.adversaire,
                         dataset=args.fichier, swap_rule=not args.sans_echange,
                         max_games=args.parties, max_seconds=args.duree,
                         eval_every=args.eval, save_every=60.0)
    session = Session(config, emit)
    try:
        session.run()
    except KeyboardInterrupt:
        print("\nArrêt demandé, sauvegarde…")
        session.save()


def main():
    parser = argparse.ArgumentParser(description="Hex — centre d'entraînement des réseaux")
    parser.add_argument("--taille", type=board_size, default=7,
                        help="taille impaire du plateau (7 par défaut)")
    parser.add_argument("--console", action="store_true",
                        help="entraîner sans interface, à pleine vitesse")
    parser.add_argument("--reseau", default="exemple_mlp", help="dossier dans reseaux/")
    parser.add_argument("--adversaire", default="autojeu",
                        choices=("autojeu", "regles", "aleatoire", "parties"))
    parser.add_argument("--fichier", help="fichier .jsonl pour l'adversaire « parties »")
    parser.add_argument("--parties", type=int, help="arrêter après ce nombre de parties")
    parser.add_argument("--duree", type=float, help="arrêter après ce nombre de secondes")
    parser.add_argument("--eval", type=int, default=1000,
                        help="parties entre deux évaluations (0 : aucune)")
    parser.add_argument("--sans-echange", action="store_true",
                        help="désactiver la règle d'échange")
    parser.add_argument("--generer", type=int, metavar="N",
                        help="générer N parties ordi à règles contre lui-même")
    parser.add_argument("--liste", action="store_true", help="lister réseaux et parties")
    args = parser.parse_args()

    if args.liste:
        from hextrainer import registry
        from hextrainer.dataset import list_datasets
        print("Réseaux :", ", ".join(registry.discover()) or "aucun")
        for path, count in list_datasets():
            print("Parties : {} ({} parties)".format(path.name, count))
        return
    if args.generer:
        from hextrainer.dataset import generate
        path, written = generate(args.taille, args.generer, progress=lambda d, t: print(
            "\r{} / {}".format(d, t), end="", flush=True))
        print("\n{} parties ajoutées à {}".format(written, path))
        return
    if args.adversaire == "parties" and not args.fichier:
        parser.error("--adversaire parties demande --fichier parties/xxx.jsonl")
    if args.console:
        console(args)
        return
    try:
        from hextrainer.ui import TrainerApp
    except ModuleNotFoundError as error:
        if error.name not in ("pygame", "numpy"):
            raise
        parser.exit(1, "{} est absent. Installez les dépendances avec :\n"
                    "  python3 -m pip install -r requirements.txt\n".format(error.name))
    TrainerApp(size=args.taille, network=args.reseau).run()


if __name__ == "__main__":
    sys.exit(main())
