"""Processus d'entraînement séparé de l'interface.

L'interface reste fluide pendant que ce processus joue et apprend à pleine
vitesse. Ils communiquent par deux files :
  commandes (interface → processus) : ("stop",), ("delay", s), ("save",)
  évènements (processus → interface) : (type, données)
"""

import multiprocessing
import queue
import traceback

from .session import Session, TrainConfig


def _drain(commands, session):
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            return
        if command[0] == "stop":
            session.stop()
        elif command[0] == "delay":
            session.cfg.move_delay = float(command[1])
        elif command[0] == "save":
            session.save()


def train_process(config, events, commands):
    emit = lambda kind, data: events.put((kind, data))
    try:
        session = Session(TrainConfig(**config), emit)
        session.run(poll=lambda s: _drain(commands, s))
    except Exception:
        emit("error", traceback.format_exc())
        emit("stopped", {})


def generate_process(size, count, events, commands):
    from .dataset import generate
    stop = [False]

    def should_stop():
        try:
            while True:
                if commands.get_nowait()[0] == "stop":
                    stop[0] = True
        except queue.Empty:
            pass
        return stop[0]

    try:
        path, written = generate(size, count, progress=lambda done, total: events.put(
            ("progress", {"fait": done, "total": total})), should_stop=should_stop)
        events.put(("generated", {"fichier": str(path), "parties": written}))
    except Exception:
        events.put(("error", traceback.format_exc()))
    events.put(("stopped", {}))


class Worker:
    """Pilote un processus (entraînement ou génération) depuis l'interface."""

    def __init__(self, target, *args):
        ctx = multiprocessing.get_context("spawn")
        self.events = ctx.Queue()
        self.commands = ctx.Queue()
        self.process = ctx.Process(target=target, args=(*args, self.events, self.commands),
                                   daemon=True)
        self.process.start()
        self.finished = False

    @classmethod
    def train(cls, config):
        return cls(train_process, config.to_dict())

    @classmethod
    def generate(cls, size, count):
        return cls(generate_process, size, count)

    def send(self, *command):
        if self.process.is_alive():
            self.commands.put(command)

    def poll(self, limit=500):
        """Évènements reçus depuis le dernier appel (au plus `limit`)."""
        received = []
        for _ in range(limit):
            try:
                received.append(self.events.get_nowait())
            except queue.Empty:
                break
        if any(kind == "stopped" for kind, _ in received):
            self.finished = True
        elif not received and not self.process.is_alive():
            self.finished = True
        return received

    def stop(self, wait=10.0):
        """Demande l'arrêt (le processus sauvegarde d'abord) et renvoie les derniers évènements.

        La file est vidée pendant l'attente : un processus dont la file est
        pleine ne peut pas se terminer."""
        import time
        self.send("stop")
        received = []
        deadline = time.monotonic() + wait
        while self.process.is_alive() and time.monotonic() < deadline:
            received.extend(self.poll())
            self.process.join(0.05)
        received.extend(self.poll())
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
        self.finished = True
        return received
