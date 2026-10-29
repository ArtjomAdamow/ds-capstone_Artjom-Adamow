"""The course model (no Dash, no state, nothing of the viewer): the pipeline fitted once on all cards, and from it the
topics, positions and connections the Helper button puts on the viewer's map.

* Decks view: one unit vector per deck (similar.deck_vectors); the layout step (models.layout, k = 3) places all
  decks on the unit sphere at once. The points are keyed by deck id, the dots of the viewer's decks state. The
  clustering of the deck vectors (models.clustering) gives the deck connections.
* Topics: the topic model T1 (models.topic_model) takes the scripted topics of the course file as its input and
  decides every card's topic and margin; the glue writes them into the derived course file the viewer reads once
  the Helper button has given a deck the model's topics (the topic dots stay the viewer's own, at random around the
  deck's centroid; the model places none).
* Cards view, for each topic of the model: the layout step on the topic's card vectors, centred on the topic's mean
  (the start's own centring), with the neighbours within the topic (k = min(5, n - 1)), gives one unit direction per
  card, and that direction is the position: every card sits on the surface of its topic's sphere (radius 1), as the
  viewer's hard rule wants it. The clustering of the topic's card vectors gives its connections (n - 1). A topic
  with fewer than two cards gets no layout and no connections. A scripted topic (in a deck without the model's
  topics) is laid out the same way on request (layout_of).
* The points are plain (x, y, z) tuples, deterministic. The viewer's position sources with their random fallback on
  the surface are the glue's (glue.positions): any id the model has no point for (a card of a one-card topic, a
  card of a deck left out, an id that is not in the course the model was fitted on) gets the viewer's random
  position there.
"""

import math
from dataclasses import dataclass
from typing import Collection, Mapping, Optional, Sequence

import numpy as np

from ..cards import Card
from . import clustering, layout, similar
from .topic_model import FOCUS_MARGIN, TopicModel, TopicSeed

DECK_NEIGHBOURS = 3
CARD_NEIGHBOURS = 5
# The force layout of the decks runs with a smaller step and a weaker push than the card layouts (the defaults of
# models.layout): on the course both measures of the deck layout are better that way (results.csv)
DECK_LAYOUT = {"iterations": 300, "step": 0.05, "repulsion": 0.5}

Point = tuple[float, float, float]
Key = tuple[str, str]  # (deck id, topic id)


@dataclass
class TopicLayout:
    """One topic's cards on the surface of the topic's sphere."""

    deck: str  # the deck name
    deck_id: str
    topic_id: str
    name: str
    ids: list[str]
    positions: dict[str, Point]  # unit directions; empty below two cards
    links: list[tuple[str, str]]  # n - 1 connections in merge order; empty below two cards
    overlap: float  # top-k overlap of the 3D directions with the within-topic neighbours (NaN below two cards)
    trustworthiness: float


class CourseModel:
    def __init__(
        self,
        cards: Sequence[Card],
        deck_ids: Mapping[str, str],
        topics: Sequence[TopicSeed],
        vectoriser=None,
        index: Optional[similar.SimilarityIndex] = None,
        margin: float = FOCUS_MARGIN,
    ) -> None:
        """Fit on all cards. `deck_ids` maps deck names to the deck ids of the course file (the decks state's dot
        ids); a deck without an id keeps its name as id. `topics` are the scripted topics (TopicSeed), the input of
        T1."""
        self.index = index if index is not None else similar.SimilarityIndex.fit(cards, vectoriser)
        self.deck_ids = {name: deck_ids.get(name, name) for name in dict.fromkeys(self.index.decks)}
        self.deck_of: dict[str, str] = {
            card_id: self.deck_ids[deck] for card_id, deck in zip(self.index.ids, self.index.decks)
        }  # card id -> deck id
        names, vectors = similar.deck_vectors(self.index.decks, self.index.vectors)
        order = sorted(range(len(names)), key=lambda row: self.deck_ids[names[row]])  # deck id order
        self.deck_names: list[str] = [names[row] for row in order]
        self.deck_vectors: np.ndarray = vectors[order] if order else vectors
        points, _start = layout.sphere_layout(self.deck_vectors, DECK_NEIGHBOURS, **DECK_LAYOUT)
        self.deck_points: dict[str, Point] = {
            self.deck_ids[name]: tuple(float(value) for value in points[row]) for row, name in enumerate(self.deck_names)
        }
        self.deck_trustworthiness = layout.trustworthiness_score(self.deck_vectors, points, DECK_NEIGHBOURS)
        self.deck_overlap = layout.neighbour_overlap(self.deck_vectors, points, DECK_NEIGHBOURS)
        self.deck_links: list[tuple[str, str]] = clustering.merge_links(
            [self.deck_ids[name] for name in self.deck_names], self.deck_vectors
        )
        self.topic_model = TopicModel(self.index, topics, margin)
        self._row_of = {card_id: row for row, card_id in enumerate(self.index.ids)}
        self.topics: dict[Key, TopicLayout] = {}
        for seed in self.topic_model.seeds:
            placed = self._lay_out(
                seed.deck, self.deck_ids.get(seed.deck, seed.deck), seed.topic_id, seed.name,
                self.topic_model.members.get(seed.key, []),
            )
            self.topics[(placed.deck_id, placed.topic_id)] = placed
        self.card_points: dict[str, Point] = {
            card_id: point for placed in self.topics.values() for card_id, point in placed.positions.items()
        }
        self._laid_out: dict[tuple, TopicLayout] = {}  # layout_of's other memberships, by deck, topic and cards

    def _lay_out(self, deck: str, deck_id: str, topic_id: str, name: str, card_ids: Sequence[str]) -> TopicLayout:
        ids = [card_id for card_id in card_ids if card_id in self._row_of]
        if len(ids) < 2:  # no layout and no connections below two cards (the user's edge case)
            return TopicLayout(deck, deck_id, topic_id, name, ids, {}, [], math.nan, math.nan)
        vectors = layout.as_dense(self.index.vectors[[self._row_of[card_id] for card_id in ids]])
        k = min(CARD_NEIGHBOURS, len(ids) - 1)
        directions, _start = layout.sphere_layout(vectors, k)  # the start centres the vectors on the topic's mean
        positions = {card_id: tuple(float(value) for value in directions[row]) for row, card_id in enumerate(ids)}
        return TopicLayout(
            deck=deck,
            deck_id=deck_id,
            topic_id=topic_id,
            name=name,
            ids=ids,
            positions=positions,
            links=clustering.merge_links(ids, vectors),
            overlap=layout.neighbour_overlap(vectors, directions, k),
            trustworthiness=layout.trustworthiness_score(vectors, directions, k),
        )

    # --- the points ------------------------------------------------------------

    def card_points_of(self, decks: Optional[Collection[str]] = None) -> dict[str, Point]:
        """The model's card points (unit directions on the topic's sphere) of the cards of these decks (deck ids;
        None for every deck). A card of a one-card topic has none."""
        if decks is None:
            return dict(self.card_points)
        return {card_id: point for card_id, point in self.card_points.items() if self.deck_of.get(card_id) in decks}

    # --- the topics, connections and neighbours ------------------------------

    @property
    def hidden(self) -> set[str]:
        """The cards T1 would hide from focus (a margin below the threshold): a measure, the app hides none."""
        return self.topic_model.hidden

    def topics_of_deck(self, deck_id: str) -> list[TopicLayout]:
        """The deck's topics in file order."""
        return [placed for placed in self.topics.values() if placed.deck_id == deck_id]

    def card_links(self, deck_id: str, topic_id: str) -> list[tuple[str, str]]:
        """The topic's connections in merge order; empty for an unknown topic or one with fewer than two cards."""
        placed = self.topics.get((deck_id, topic_id))
        return list(placed.links) if placed else []

    def layout_of(self, deck_id: str, topic_id: str, card_ids: Sequence[str], name: str = "") -> TopicLayout:
        """The layout of a topic with these cards: the model's own when the membership is the model's (a deck with
        the model's topics), else the layout step and the clustering on these cards (a scripted topic), kept for
        the next call. Cards the model was not fitted on are left out."""
        placed = self.topics.get((deck_id, topic_id))
        if placed is not None and set(placed.ids) == {card_id for card_id in card_ids if card_id in self._row_of}:
            return placed
        key = (deck_id, topic_id, tuple(card_ids))
        if key not in self._laid_out:
            deck = next((deck for deck, known in self.deck_ids.items() if known == deck_id), deck_id)
            self._laid_out[key] = self._lay_out(deck, deck_id, topic_id, name, card_ids)
        return self._laid_out[key]

    def related(self, card_id: str) -> list[tuple[str, float]]:
        """The neighbours of a card across the whole course, best first."""
        return self.index.related(card_id)
