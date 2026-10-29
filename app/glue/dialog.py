"""The Helper dialog's catalogue (glue; no Dash): the models the Helper button offers on each view, their labels, the
sublevels each reaches from there, the note next to Confirm, and the run order. The viewer draws the dialog from
these options (project.services.actions.ActionOption); Helper.run (button.py) runs the checked ones.

    | View   | Models offered                             | Note while checked                        |
    |--------|--------------------------------------------|-------------------------------------------|
    | decks  | Topics, Positions, Connections, Difficulty | Topics: the topics and cards views; the   |
    |        |                                            | others: the cards view                    |
    | topics | Topics, Positions, Connections, Difficulty | every model: the cards view               |
    | cards  | Positions, Connections, Difficulty         | none (no sublevel)                        |

* Topics (T1) gives the decks of the view the model's topics, so the cards' membership follows. It is not offered in
  the cards view: T1 decides a deck's topics, not one topic's.
* Positions places the decks (the decks view) and every checked topic's cards on their spheres; the topic dots keep
  the viewer's positions.
* Connections writes the model's connections and the cut-off presets: the decks' and every checked topic's.
* Difficulty fills the difficulty and predicted groups of the view's cards (decks are not rated).

The run order is Topics, Positions, Connections, Difficulty: T1 decides the membership the layouts and the trees
are made per, the preset needs the connections, the groups are independent. Every box is checked when the dialog
opens, so Confirm does what the one click did (with the groups in the decks view).
"""

from typing import Iterable

from project.services.actions import ActionOption
from project.services.views import CARDS, DECKS, TOPICS

MODEL_TOPICS = "topics"
MODEL_POSITIONS = "positions"
MODEL_CONNECTIONS = "connections"
MODEL_DIFFICULTY = "difficulty"
RUN_ORDER = (MODEL_TOPICS, MODEL_POSITIONS, MODEL_CONNECTIONS, MODEL_DIFFICULTY)
LABELS = {MODEL_TOPICS: "Topics", MODEL_POSITIONS: "Positions", MODEL_CONNECTIONS: "Connections", MODEL_DIFFICULTY: "Difficulty"}
# The models each view offers, in option order (the run order)
OFFERED = {DECKS: RUN_ORDER, TOPICS: RUN_ORDER, CARDS: (MODEL_POSITIONS, MODEL_CONNECTIONS, MODEL_DIFFICULTY)}
VIEW_NAMES = {DECKS: "decks", TOPICS: "topics", CARDS: "cards"}


def sublevels(view: str, model: str) -> tuple[str, ...]:
    """The views below `view` whose objects a model started there changes: T1 on the decks view changes the topics
    and the cards; every other model on the decks or topics view the cards; nothing reaches below the cards view."""
    if view == DECKS and model == MODEL_TOPICS:
        return (TOPICS, CARDS)
    if view in (DECKS, TOPICS):
        return (CARDS,)
    return ()


def note(view: str, model: str) -> str:
    """The sentence next to Confirm while the model is checked; empty without a sublevel. The plural is written
    out ("the cards view", "the topics and cards views")."""
    below = [VIEW_NAMES[level] for level in sublevels(view, model)]
    if not below:
        return ""
    views = f"{below[0]} view" if len(below) == 1 else f"{', '.join(below[:-1])} and {below[-1]} views"
    return f"Starting model {LABELS[model]} on the {VIEW_NAMES[view]} view affects all objects on the {views}"


def options_for(view: str) -> list[ActionOption]:
    """The dialog's checklist on a view: the models it offers, each checked, with its note."""
    return [ActionOption(model, LABELS[model], True, note(view, model)) for model in OFFERED.get(view, ())]


def run_order(view: str, checked: Iterable[str]) -> list[str]:
    """The checked models the view offers, in the run order (anything else is ignored)."""
    wanted = set(checked)
    return [model for model in OFFERED.get(view, ()) if model in wanted]
