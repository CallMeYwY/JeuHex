"""Réseau d'exemple : un perceptron multicouche en numpy pur.

    entrée 3×n×n (mes pierres, adversaire, vides — vue canonique)
      → cachée 1 : 128 neurones ReLU
      → cachée 2 : 64 neurones ReLU
      → politique : n×n logits (softmax sur les cases libres)
      → valeur : 1 neurone tanh (+1 : le joueur au trait va gagner)

Apprentissage (`learn`) :
  * politique : imite les coups du camp gagnant, et tous ceux d'un « professeur »
    (ordi à règles, parties enregistrées) ;
  * valeur : prédit l'issue de la partie.
Le plateau tourné de 180° est équivalent au Hex : chaque exemple compte double.

Copiez ce dossier pour démarrer votre propre réseau.
"""

import os

import numpy as np

from hextrainer.api import SWAP, Brain, Decision, Explanation, Layer


def _masked_softmax(logits, mask):
    z = np.where(mask, logits, -1e9)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


class Reseau(Brain):
    name = "Exemple MLP"
    description = "Perceptron 128-64 en numpy : imite les coups gagnants, estime qui gagne."
    hidden = (128, 64)
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, size):
        super().__init__(size)
        rng = np.random.default_rng(0)
        n_in, n_out = 3 * size * size, size * size
        h1, h2 = self.hidden

        def he(fan_in, fan_out, scale=1.0):
            return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2 / fan_in) * scale
                    ).astype(np.float32)

        self.params = {
            "W1": he(n_in, h1), "b1": np.zeros(h1, np.float32),
            "W2": he(h1, h2), "b2": np.zeros(h2, np.float32),
            "Wp": he(h2, n_out, 0.1), "bp": np.zeros(n_out, np.float32),
            "Wv": he(h2, 1, 0.1), "bv": np.zeros(1, np.float32),
        }
        self.adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_t = 0
        self.rng = np.random.default_rng()

    # --- Calcul -----------------------------------------------------------------
    def forward(self, x):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        logits = a2 @ p["Wp"] + p["bp"]
        value = np.tanh(a2 @ p["Wv"] + p["bv"])[:, 0]
        return {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "logits": logits, "v": value}

    def _evaluate(self, obs):
        cache = self.forward(obs.planes().reshape(1, -1))
        mask = obs.legal_canonical().reshape(1, -1)
        probs = _masked_softmax(cache["logits"], mask)[0]
        return cache, probs, float(cache["v"][0])

    # --- Jouer -------------------------------------------------------------------
    def act(self, obs):
        _, probs, value = self._evaluate(obs)
        n = self.size
        policy = obs.to_real(probs.reshape(n, n))
        if obs.can_swap:
            # Échanger = prendre la place de l'adversaire : utile si notre position est mauvaise.
            swap = value < -0.1
            if obs.training and self.rng.random() < 0.1:
                swap = not swap
            if swap:
                return Decision(SWAP, policy, value)
        if obs.training:
            index = int(self.rng.choice(probs.size, p=probs))
        else:
            index = int(np.argmax(probs))
        move = obs.cell_to_real(divmod(index, n))
        return Decision(move, policy, value)

    # --- Apprendre ---------------------------------------------------------------
    def learn(self, samples):
        n = self.size
        xs, masks, targets, weights, values = [], [], [], [], []
        for s in samples:
            planes = s.obs.planes()
            legal = s.obs.legal_canonical()
            # Valeur : la couleur au trait a-t-elle gagné ? (vrai aussi après un échange)
            value = s.color_outcome
            if s.move == SWAP:
                cell, weight = (0, 0), 0.0  # Pas de case à imiter.
            else:
                cell = s.obs.cell_to_canonical(s.move)
                teacher = s.actor in ("regles", "partie")
                weight = 1.0 if (s.outcome > 0 or teacher) else 0.0
            for flip in (False, True):
                if flip:  # Rotation de 180° : même partie, vue autrement.
                    planes, legal = planes[:, ::-1, ::-1], legal[::-1, ::-1]
                    cell = (n - 1 - cell[0], n - 1 - cell[1])
                xs.append(planes.reshape(-1))
                masks.append(legal.reshape(-1))
                targets.append(cell[0] * n + cell[1])
                weights.append(weight)
                values.append(value)
        if not xs:
            return {}
        x = np.asarray(xs, np.float32)
        mask = np.asarray(masks, bool)
        target = np.asarray(targets)
        weight = np.asarray(weights, np.float32)
        value = np.asarray(values, np.float32)
        order = self.rng.permutation(len(x))
        totals = {"perte_politique": 0.0, "perte_valeur": 0.0, "precision": 0.0}
        batches = 0
        for start in range(0, len(x), self.batch_size):
            idx = order[start:start + self.batch_size]
            stats = self._step(x[idx], mask[idx], target[idx], weight[idx], value[idx])
            for key in totals:
                totals[key] += stats[key]
            batches += 1
        return {k: v / batches for k, v in totals.items()}

    def _step(self, x, mask, target, weight, value):
        p = self.params
        c = self.forward(x)
        batch = len(x)
        probs = _masked_softmax(c["logits"], mask)
        rows = np.arange(batch)
        picked = probs[rows, target]
        wsum = max(weight.sum(), 1.0)
        policy_loss = float(-(weight * np.log(picked + 1e-9)).sum() / wsum)
        value_loss = float(((c["v"] - value) ** 2).mean())
        accuracy = float((weight * (probs.argmax(1) == target)).sum() / wsum)

        dlogits = probs.copy()
        dlogits[rows, target] -= 1
        dlogits *= (weight / wsum)[:, None]
        dv = ((c["v"] - value) * (1 - c["v"] ** 2) / batch)[:, None]
        grads = {"Wp": c["a2"].T @ dlogits, "bp": dlogits.sum(0),
                 "Wv": c["a2"].T @ dv, "bv": dv.sum(0)}
        da2 = dlogits @ p["Wp"].T + dv @ p["Wv"].T
        dz2 = da2 * (c["z2"] > 0)
        grads["W2"], grads["b2"] = c["a1"].T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ p["W2"].T) * (c["z1"] > 0)
        grads["W1"], grads["b1"] = c["x"].T @ dz1, dz1.sum(0)
        self._adam(grads)
        return {"perte_politique": policy_loss, "perte_valeur": value_loss,
                "precision": accuracy}

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.adam_t += 1
        t = self.adam_t
        for k, g in grads.items():
            m = self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * g
            v = self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * g * g
            step = self.learning_rate * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
            self.params[k] -= step.astype(np.float32)

    # --- Montrer son raisonnement ----------------------------------------------
    def explain(self, obs):
        cache, probs, value = self._evaluate(obs)
        p, n = self.params, self.size
        return Explanation(
            layers=[
                Layer("Entrée", obs.planes(), channels=("mes pierres", "adversaire", "vides"),
                      note="Vue du joueur au trait, qui relie le haut au bas."),
                Layer("Cachée 1", cache["a1"][0], weights=p["W1"].T, note="ReLU"),
                Layer("Cachée 2", cache["a2"][0], weights=p["W2"].T, note="ReLU"),
                Layer("Politique", probs.reshape(n, n), weights=p["Wp"].T,
                      note="Probabilité de jouer chaque case (vue canonique)."),
                Layer("Valeur", cache["v"], weights=p["Wv"].T, source="Cachée 2",
                      note="tanh : +1 = victoire du joueur au trait"),
            ],
            policy=obs.to_real(probs.reshape(n, n)),
            value=value,
        )

    # --- Sauvegarde ---------------------------------------------------------------
    def save(self, folder):
        arrays = dict(self.params)
        arrays.update({"m_" + k: v for k, v in self.adam_m.items()})
        arrays.update({"v_" + k: v for k, v in self.adam_v.items()})
        arrays["adam_t"] = np.array(self.adam_t)
        np.savez(os.path.join(folder, "poids.npz"), **arrays)

    def load(self, folder):
        path = os.path.join(folder, "poids.npz")
        if not os.path.exists(path):
            return False
        with np.load(path) as data:
            if any(data[k].shape != v.shape for k, v in self.params.items()):
                return False  # Architecture modifiée : on repart de zéro.
            for k in self.params:
                self.params[k] = data[k].astype(np.float32)
                self.adam_m[k] = data["m_" + k]
                self.adam_v[k] = data["v_" + k]
            self.adam_t = int(data["adam_t"])
        return True
