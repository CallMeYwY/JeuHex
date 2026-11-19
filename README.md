# Hex

- **À deux** — deux joueurs se relaient sur le même écran.
- **Ordi à règles** — jouez contre l’ordinateur, qui choisit ses coups à l’aide de règles tactiques et d’une estimation des chemins.
- **Duel IA** — cliquez sur les deux noms pour choisir deux IA différentes, puis lancez une partie. Les IA jouent automatiquement.
- **Tournoi IA** — lancez une partie pour démarrer les demi-finales croisées (`user1` contre `user4`, puis `user2` contre `user3`). Passez au match suivant depuis la barre latérale. Les deux gagnants jouent la finale et le champion s’affiche sous le plateau.

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
Sous les tailles prédéfinies, le champ de taille libre accepte les entiers impairs à partir de 3. Cliquez sur le champ, saisissez par exemple 31 ou 51, puis cliquez sur **Nouvelle partie**.

Le bouton **Réflexion des IA** ouvre une deuxième fenêtre pendant que le match se poursuit. Pour chaque IA engagée, elle montre les règles consultées lors de son dernier tour et entoure celle qui a déterminé le coup. Si l’IA compare les chemins, la fenêtre affiche aussi les cinq meilleurs scores de cases. Si vous personnalisez une IA sans utiliser la stratégie à règles fournie, la fenêtre affiche toujours le coup joué.

Le bouton **Délai IA** ouvre le réglage du rythme. Saisissez un délai en millisecondes et cliquez sur **Appliquer** pour reprendre avec le nouveau réglage. Par défaut, le délai est de 400 ms. Saisissez 0 pour avancer dès que l’IA a trouvé son coup. Le temps de calcul dépend de la stratégie et de la taille du plateau.

Rouge relie le haut au bas, bleu relie la gauche à la droite. Après le premier coup, le second joueur peut échanger les couleurs. La première chaîne continue reliant les deux bords de sa couleur gagne.

La barre d’évaluation au-dessus du plateau se met à jour après chaque coup, dans tous les modes. Un score positif favorise les rouges, un score négatif favorise les bleus et zéro indique une position estimée équilibrée. L’indice compare le nombre de cases vides restant sur le chemin le plus court de chaque couleur. Il indique un avantage de position et ne représente pas une probabilité de victoire. Lors d’un échange, l’avantage reste attaché à la couleur qui le possède.

Raccourcis clavier : **N** pour une nouvelle partie, **U** pour annuler le dernier tour, **E** pour échanger les couleurs et **H** pour l’aide.

## Personnaliser les IA

Le dossier `IA` contient les quatre stratégies de départ, dans `user1.py` à `user4.py`.
Remplacez la fonction `choose_action(game)` du participant à personnaliser, puis relancez le jeu.
Elle reçoit une copie de la partie et joue pour `game.current_player`, dont la couleur est `game.turn`.
Elle doit renvoyer une case libre sous forme de tuple `(ligne, colonne)` d’entiers indexés à partir de zéro, ou `"swap"` si `game.can_swap` autorise l’échange.
`game.legal_moves()` donne la liste des cases libres jouables.
