"""The viewer's position sources over the model's points (glue). Each of the viewer's two states takes a position
source with the signature (ids, seed) -> list[CardPosition] (project.store.state_core); it places the dots without
a stored position. The sources here are deterministic for the ids the model has a point for and ignore the seed;
any other id (a card of a one-card topic, a card of a deck left out, an id that is not in the course the model was
fitted on) gets the viewer's random position on the surface (project.services.positions.surface_positions), with
the seed.

* The decks state keeps the viewer's default: the Helper button sets the model's deck positions itself, and the
  helper passes no `deck_positions` to compose (deck_positions is here for the record and the tests).
* The card state takes `card_positions`, limited to the decks with the model's topics (Helper.card_positions), so a
  deck checked again gets the model's positions back and every other deck the viewer's random ones.
* random_points draws the viewer's random positions as plain points, for the evaluation's random floor and random
  layouts (evaluate.py).
"""

from typing import Collection, Mapping, Optional, Sequence

import numpy as np

from project.models import CardPosition
from project.services.positions import surface_positions

from ..models.course_model import CourseModel, Point


def positions_from(ids: Sequence[str], known: Mapping[str, Point], seed: int = 42) -> list[CardPosition]:
    """The known points as the viewer's positions, in the order of `ids`; the unknown ids at random on the surface."""
    unknown = [card_id for card_id in ids if card_id not in known]
    random_positions = {position.comb_id: position for position in surface_positions(unknown, seed)} if unknown else {}
    positions = []
    for card_id in ids:
        if card_id in known:
            x, y, z = known[card_id]
            positions.append(CardPosition(comb_id=card_id, x=x, y=y, z=z))
        else:
            positions.append(random_positions[card_id])
    return positions


def deck_positions(model: CourseModel, ids: Sequence[str], seed: int = 42) -> list[CardPosition]:
    """A PositionSource of the decks state: the model's deck points; unknown ids are placed at random on the
    surface."""
    return positions_from(ids, model.deck_points, seed)


def card_positions(
    model: CourseModel, ids: Sequence[str], seed: int = 42, decks: Optional[Collection[str]] = None
) -> list[CardPosition]:
    """A PositionSource of the card state's cards: the model's directions (radius 1) for the cards of `decks` (deck
    ids; None for every deck); the other ids, unknown ids and the cards of a one-card topic are placed at random on
    the surface."""
    return positions_from(ids, model.card_points_of(decks), seed)


def random_points(ids: Sequence[str], seed: int = 42) -> np.ndarray:
    """The viewer's random positions on the surface as one (x, y, z) row per id."""
    return np.array([[position.x, position.y, position.z] for position in surface_positions(list(ids), seed)]).reshape(len(ids), 3)
