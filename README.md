# Hex

- **À deux** — deux joueurs se relaient sur le même écran.
- **Ordi à règles** — jouez contre l’ordinateur, qui choisit ses coups à l’aide de règles tactiques et d’une estimation des chemins.

## Lancer le jeu

Depuis le dossier `jeu-seul` :

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
