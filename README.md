# Hex

- **À deux** — deux joueurs se relaient sur le même écran.
- **Ordi à règles** — jouez contre l’ordinateur, qui choisit ses coups à l’aide de règles tactiques et d’une estimation des chemins.

## Lancer le jeu

Depuis le dossier du projet :

```sh
python3 -m pip install -r requirements.txt
python3 main.py
```

Pour démarrer directement contre l’ordinateur sur un plateau 7 × 7 :

```sh
python3 main.py --ai --size 7
```

Dans le jeu, choisissez la taille et l’adversaire pour la prochaine partie, puis cliquez sur **Nouvelle partie**.

Rouge relie le haut au bas, bleu relie la gauche à la droite. Après le premier coup, le second joueur peut échanger les couleurs. La première chaîne continue reliant les deux bords de sa couleur gagne.

Raccourcis clavier : **N** pour une nouvelle partie, **U** pour annuler le dernier tour, **E** pour échanger les couleurs et **H** pour l’aide.

## Centre d’entraînement des réseaux

Une application distincte du jeu, `trainer.py`, entraîne des réseaux de neurones
que vous écrivez vous-mêmes. Elle reprend les règles, l’ordi à règles et le
style graphique du jeu, sans rien y modifier.

```sh
python3 trainer.py                 # interface
python3 trainer.py --taille 9      # démarrer sur un plateau 9 × 9
```

- **Vos réseaux** se déposent dans `reseaux/<nom>/` : voir [reseaux/README.md](reseaux/README.md).
  `reseaux/exemple_mlp` est un réseau complet en numpy, `reseaux/_modele` un squelette à copier.
- **Tailles** : 5, 7, 9, 11, ou toute taille impaire de 3 à 25.
- **Adversaires** : *autojeu* (le réseau contre lui-même), *ordi à règles* (celui du jeu,
  en version rapide), *aléatoire*, ou *parties* enregistrées dans `parties/*.jsonl`.
- **Vitesse** : *Turbo* joue des centaines de parties par seconde dans un processus séparé ;
  *Rapide* et *Lent* montrent chaque coup.
- **Raisonnement** : le plateau teinte en vert les cases que le réseau veut jouer, une jauge
  donne qui gagne selon lui, et la colonne centrale montre ses couches. Survolez un neurone
  pour voir ses connexions et les cases qu’il regarde. Le **bac à sable** permet de poser
  des pierres soi-même et de voir le réseau réagir.
- Poids et courbes sont sauvegardés dans `reseaux/<nom>/sauvegardes/taille_<n>/` :
  l’entraînement reprend où il s’était arrêté.

Sans interface, à pleine vitesse :

```sh
python3 trainer.py --console --reseau exemple_mlp --taille 7 --adversaire regles --duree 600
python3 trainer.py --console --adversaire parties --fichier parties/regles_9.jsonl --taille 9
python3 trainer.py --generer 500 --taille 9      # parties pré-jouées par l’ordi à règles
python3 trainer.py --liste                       # réseaux et fichiers de parties
```

Format d’un fichier de parties (une partie par ligne, notation de l’interface) :

```json
{"taille": 7, "coups": ["D4", "echange", "C3", "E2"], "gagnant": "rouge", "joueurs": ["regles", "regles"]}
```

Tests : `python3 -m unittest`.
