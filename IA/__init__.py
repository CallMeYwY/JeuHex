"""Participants de l'arène Hex."""

from . import user1, user2, user3, user4

PLAYERS = {
    "user1": user1.choose_action,
    "user2": user2.choose_action,
    "user3": user3.choose_action,
    "user4": user4.choose_action,
}
SEMIFINALS = (("user1", "user4"), ("user2", "user3"))
