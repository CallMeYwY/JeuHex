# Dossier des réseaux

Chaque sous-dossier est un réseau que l'entraîneur détecte tout seul :

```
reseaux/
├── exemple_mlp/          réseau d'exemple (numpy pur), prêt à entraîner
│   ├── __init__.py
│   └── sauvegardes/      créé à l'entraînement : un dossier par taille
│       └── taille_7/     poids + journal.json (courbes, nombre de parties)
├── _modele/              squelette vide à copier (ignoré : commence par « _ »)
└── mon_reseau/           ← le vôtre
```

## Ajouter un réseau

```sh
cp -r reseaux/_modele reseaux/mon_reseau
python3 trainer.py                 # il apparaît dans « Réseau », flèches < >
```

Le dossier doit contenir un `__init__.py` qui expose **une classe `Reseau(size)`**
(ou une fonction `create(size)`). Vous pouvez y mettre d'autres fichiers
(`modele.py`, `couches.py`…) et les importer en relatif (`from .modele import …`).
Le bouton **Recharger le code** relit vos fichiers sans relancer l'appli.

Une instance est créée **par taille de plateau** : `self.size` vaut 5, 7, 9, 11…
Les poids sont sauvegardés séparément pour chaque taille.

## Le contrat (`hextrainer/api.py`)

| Méthode | Obligatoire | Rôle |
|---|---|---|
| `act(obs)` | oui | renvoie le coup : `Decision(coup, politique, valeur)`, ou simplement `(ligne, colonne)` / `"swap"` |
| `learn(samples)` | non | apprend d'un lot de décisions, renvoie des mesures à tracer : `{"perte": 0.42}` |
| `explain(obs)` | non | décrit les couches pour l'affichage (`Explanation`) |
| `save(dossier)` / `load(dossier)` | non | écrit / relit les poids ; `load` renvoie `False` s'il n'y a rien |

### Ce que voit le réseau : `Observation`

- `obs.board` : tableau n×n (0 vide, 1 rouge, 2 bleu), orientation réelle ;
- `obs.to_play` : couleur au trait (1 rouge relie haut↔bas, 2 bleu gauche↔droite) ;
- `obs.legal` : cases jouables ; `obs.can_swap` : l'échange est permis ;
- `obs.training` : vrai pendant l'entraînement (pensez à explorer), faux en évaluation.

**Vue canonique** — `obs.planes()` renvoie un tableau `(3, n, n)` :
*mes pierres, pierres adverses, cases vides*, toujours vu par le joueur au trait
**comme s'il reliait le haut au bas**. Pour bleu, le plateau est transposé
(ligne ↔ colonne), ce qui respecte les voisinages du Hex. Un seul réseau joue
donc les deux couleurs. Outils de conversion :
`obs.legal_canonical()`, `obs.to_real(grille)`, `obs.cell_to_real(case)`,
`obs.cell_to_canonical(case)`.

### Ce qu'il apprend : `Sample`

`learn` reçoit toutes les décisions des parties jouées depuis le dernier appel
(16 parties en Turbo, une seule au ralenti) :

- `obs`, `move` (orientation réelle, ou `"swap"`) ;
- `outcome` : +1 si le **joueur** qui a pris cette décision a gagné ;
- `color_outcome` : +1 si la **couleur au trait** a gagné → cible d'une tête « valeur » ;
- `actor` : `"reseau"`, `"regles"`, `"aleatoire"` ou `"partie"` (fichier) ;
- `policy` : la politique que votre réseau avait renvoyée, si c'est lui qui jouait ;
- `game`, `ply` : numéro de la partie et rang du coup, dans l'ordre de jeu.

**Pourquoi par lots de parties ?** Chaque partie est bien jouée coup par coup :
`act` est appelé à chaque coup. Mais on ne sait qui a gagné qu'à la fin, et c'est
ce résultat qui dit si un coup était bon : `learn` reçoit donc les parties une
fois terminées. Grouper plusieurs parties rend aussi chaque mise à jour plus
stable et plus rapide. Si vous voulez apprendre coup par coup (méthode TD :
corriger la valeur d'une position avec celle de la position suivante), `game`
et `ply` vous donnent l'ordre exact des positions.

Selon l'adversaire choisi, les décisions viennent du réseau contre lui-même
(autojeu), du réseau et de l'ordi à règles, du réseau et du hasard, ou de
parties enregistrées. À vous de choisir lesquelles imiter.

### Montrer son raisonnement : `Explanation`

```python
Explanation(
    layers=[
        Layer("Entrée", obs.planes(), channels=("mes pierres", "adversaire", "vides")),
        Layer("Cachée 1", h1, weights=W1.T),            # (nb_neurones, nb_entrées)
        Layer("Politique", probas.reshape(n, n), weights=Wp.T),
        Layer("Valeur", v, weights=Wv.T, source="Cachée 1"),  # branchée ailleurs
    ],
    policy=politique_reelle_nxn,   # teinte le plateau en vert
    value=v,                       # jauge « qui gagne ? »
)
```

Une couche 1D se dessine en grille de neurones, une couche 2D `(h, w)` ou 3D
`(c, h, w)` en petits plateaux. Avec `weights` (forme `(sorties, entrées)` après
aplatissement), l'entraîneur trace les connexions les plus fortes et, au survol
d'un neurone, montre **quelles cases il regarde** (ses poids sur le plateau).

## Sans numpy ? Avec PyTorch ?

Tout passe par des tableaux numpy, rien n'impose une bibliothèque. Avec PyTorch,
convertissez dans `act`/`learn` (`torch.from_numpy(obs.planes())`) et renvoyez
des tableaux numpy dans `explain` (`tensor.detach().numpy()`).
