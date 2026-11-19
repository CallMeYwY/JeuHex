"""Découverte des réseaux déposés dans `reseaux/`.

Chaque sous-dossier `reseaux/<nom>/` contenant un `__init__.py` est un réseau,
sauf ceux dont le nom commence par « _ » (modèles, brouillons). Le module doit
exposer `create(size)` ou une classe `Reseau(size)`.
"""

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETWORKS_DIR = ROOT / "reseaux"
DATASETS_DIR = ROOT / "parties"


def discover():
    if not NETWORKS_DIR.is_dir():
        return []
    return sorted(p.name for p in NETWORKS_DIR.iterdir()
                  if p.is_dir() and (p / "__init__.py").exists()
                  and not p.name.startswith(("_", ".")))


def load_module(name, reload=False):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module_name = "reseaux." + name
    if reload and module_name in sys.modules:
        importlib.invalidate_caches()
        # Recharge aussi les sous-modules du réseau, puis le paquet lui-même.
        for key in sorted((k for k in sys.modules if k.startswith(module_name + ".")),
                          reverse=True):
            importlib.reload(sys.modules[key])
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def create(name, size, reload=False):
    module = load_module(name, reload)
    if hasattr(module, "create"):
        brain = module.create(size)
    elif hasattr(module, "Reseau"):
        brain = module.Reseau(size)
    else:
        raise AttributeError("reseaux/{}/__init__.py doit définir create(size) "
                             "ou une classe Reseau.".format(name))
    if not hasattr(brain, "act"):
        raise TypeError("Le réseau {} n'a pas de méthode act(obs).".format(name))
    return brain


def display_name(brain, name):
    return getattr(brain, "name", None) or name


def checkpoint_dir(name, size):
    """reseaux/<nom>/sauvegardes/taille_<n>, ou $HEXTRAINER_SAUVEGARDES/<nom>/taille_<n>."""
    override = os.environ.get("HEXTRAINER_SAUVEGARDES")
    if override:
        return Path(override) / name / "taille_{}".format(size)
    return NETWORKS_DIR / name / "sauvegardes" / "taille_{}".format(size)


def save_checkpoint(brain, name, size, extra_files=None):
    """Sauvegarde atomique : dossier temporaire puis renommage."""
    target = checkpoint_dir(name, size)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    old = target.with_name(target.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    if hasattr(brain, "save"):
        brain.save(str(tmp))
    for filename, text in (extra_files or {}).items():
        (tmp / filename).write_text(text, encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    tmp.rename(target)
    shutil.rmtree(old, ignore_errors=True)
    return target


def load_checkpoint(brain, name, size):
    folder = checkpoint_dir(name, size)
    if not folder.is_dir() or not hasattr(brain, "load"):
        return False
    return bool(brain.load(str(folder)))
