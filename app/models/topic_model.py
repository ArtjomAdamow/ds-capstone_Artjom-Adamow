"""The topic model T1 (no Dash, no state, nothing of the viewer): nearest centroid on the similarity index, seeded by
the scripted topics of the course file (docs/models/topic.md).

The topics of the course file are scripted (scripts/generate_topics.py groups the cards by the headings and sources
of the lectures), so they are the model's input, not the truth: the model takes them in and decides every card's topic
for the state. The raw scripted topics stay in the course file, so nothing is ever removed.

    unit rows of the cards (models.similar) ─────┐   per deck: a centroid per scripted topic (the mean of its cards'
                                                 ├→  unit rows, normalised) → every card's cosine to the deck's centroids,
    scripted topics (TopicSeed, from the file) ──┘   its own topic's without itself (leave-one-out) → the nearest
                                                     centroid is the card's topic, the best cosine minus the second
                                                     best its margin

* The decision for the state: the membership is the nearest centroid, for every card (the app shows them all); a
  topic is never removed, and an emptied one keeps its dot. The margin is kept as a measure: a card below
  FOCUS_MARGIN sits between two topics (`hidden`, for the evaluation), and the runner-up topic next to it
  (`second_of`, for the models view's card panel). The models view's margin table (and, until step 33, the viewer's
  margin slider, the glue's focus of step 28) runs on
  a share scale: `share_threshold` gives the margin that keeps a share of the cards, the highest margins first, so
  100 % keeps every card and 1 % the eleven with the highest margins on the course (models/topic.md, "The margin on
  a share scale").
* The derived course file the viewer reads (the raw file with the cards' "topic" set by the model) is the plugin's
  (plugin.derived.write_derived_course): it is written in the viewer's format with the viewer's parser.
* The measures: the agreement with the scripted topics (the share of cards whose nearest centroid is their scripted
  topic, one minus the lessons' Hamming loss), the coherence per topic (the share of its scripted cards nearest to
  itself), the margin table (the cards kept at a threshold and their agreement), the moved cards and the emptied
  topics; evaluate.py --part topics records them.
* T1 is deterministic (the vectoriser and the means): the derived file and the viewer's dot ids are the same at
  every start; a changed course or vectoriser changes them, and the state starts fresh (the viewer's rule).
* As written (`moves=False`, the demo's deck 00, docs/demo/architecture.md): every card keeps its scripted topic,
  no card is moved; the margins and the runner-ups are measured as above.

Edge cases: a topic with one scripted card has no centroid under leave-one-out, so its card
goes to the nearest other topic and the topic is emptied; a card with a single centroid to compare with (its own
topic the deck's only one with other cards) takes it with the full margin; an empty topic has no centroid and no
coherence. Ties go to the topic that comes first in the file.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import similar
from .layout import as_dense

# A card whose margin (the best cosine minus the second best) is below this sits between two topics: at 0.05, 70 %
# of the cards are above it at 0.77 agreement with the scripted topics (models/topic.md). A measure; the app hides none
FOCUS_MARGIN = 0.05
FULL_MARGIN = 1.0  # the margin of a card with nothing to compare its topic with: it is kept
MODEL_NAME = "T1 nearest centroid"

Key = tuple[str, str]  # (deck name, topic id)


@dataclass(frozen=True)
class TopicSeed:
    """A scripted topic of the course file, the model's input: its deck (by name), its id within the deck, its
    name, whether it is the rest topic, its scripted cards in deck order, and the course file's headings (the
    lecture's H2 and H3 titles, P1's text route) and sources (the lecture notebook and the exercise files, a
    source's files joined by "; " as the file lists them)."""

    deck: str
    topic_id: str
    name: str
    rest: bool
    card_ids: tuple[str, ...]
    headings: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def key(self) -> Key:
        return (self.deck, self.topic_id)


def unit(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to length 1 (a zero vector stays zero)."""
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector


def share_threshold(margins: Sequence[float], share: float) -> float:
    """The margin that keeps `share` (0 to 1) of these cards in focus, the highest margins first: 0.0 at a share of
    1 or more (every margin is 0 or above), the k-th highest margin for a share that rounds (half up) to k cards,
    and above the highest margin for a share that rounds to no card; 0.0 without margins."""
    ordered = sorted((float(margin) for margin in margins), reverse=True)
    if not ordered:
        return 0.0
    kept = max(0, min(len(ordered), int(share * len(ordered) + 0.5)))  # rounded half up: 50 % of 5 cards keeps 3
    if kept >= len(ordered):
        return 0.0
    if kept == 0:
        return ordered[0] + 1.0
    return ordered[kept - 1]


class TopicModel:
    """T1 fitted on the index: every card's topic and margin, the model's membership per topic, the coherence per
    scripted topic, and the measures against the scripted topics."""

    def __init__(
        self, index: similar.SimilarityIndex, seeds: Sequence[TopicSeed], margin: float = FOCUS_MARGIN, moves: bool = True,
    ) -> None:
        """`moves=False` keeps every card in its scripted topic (the membership as written), the margins measured."""
        self.seeds: list[TopicSeed] = list(seeds)
        self.margin = margin
        self.moves = moves
        self.scripted_of: dict[str, str] = {card_id: seed.topic_id for seed in self.seeds for card_id in seed.card_ids}
        self.topic_of: dict[str, str] = {}  # card id -> the model's topic id (within the card's deck)
        self.margin_of: dict[str, float] = {}  # card id -> the best cosine minus the second best
        self.second_of: dict[str, str] = {}  # card id -> the runner-up topic id ("" with nothing to compare with)
        self.members: dict[Key, list[str]] = {seed.key: [] for seed in self.seeds}  # the model's membership, deck order
        self.coherence: dict[Key, float] = {}  # the share of a topic's scripted cards nearest to itself (NaN: no cards)
        self._fit(index)

    def _fit(self, index: similar.SimilarityIndex) -> None:
        by_deck: dict[str, list[TopicSeed]] = {}
        for seed in self.seeds:
            by_deck.setdefault(seed.deck, []).append(seed)
        for deck, seeds in by_deck.items():
            rows = index.rows_of_deck(deck)
            ids = [index.ids[row] for row in rows]
            position = {card_id: place for place, card_id in enumerate(ids)}
            vectors = as_dense(index.vectors[rows]) if rows else np.zeros((0, 0))
            sums: dict[str, np.ndarray] = {}
            counts: dict[str, int] = {}
            for seed in seeds:
                own = [position[card_id] for card_id in seed.card_ids if card_id in position]
                sums[seed.topic_id] = vectors[own].sum(axis=0) if own else np.zeros(vectors.shape[1] if rows else 0)
                counts[seed.topic_id] = len(own)
            for place, card_id in enumerate(ids):
                own_topic = self.scripted_of.get(card_id)
                scores = []
                for seed in seeds:
                    total, count = sums[seed.topic_id], counts[seed.topic_id]
                    if seed.topic_id == own_topic:
                        total, count = total - vectors[place], count - 1  # leave-one-out: never its own vote
                    if count <= 0:
                        continue
                    scores.append((float(vectors[place] @ unit(total)), seed.topic_id))
                second = ""
                if not scores:
                    if own_topic is None:
                        continue  # not a card of this model: no topic to keep
                    topic, margin = own_topic, FULL_MARGIN
                elif len(scores) == 1:
                    topic, margin = scores[0][1], FULL_MARGIN
                else:
                    ordered = sorted(scores, key=lambda item: -item[0])  # a stable sort: ties keep the file order
                    topic, margin, second = ordered[0][1], ordered[0][0] - ordered[1][0], ordered[1][1]
                if not self.moves and own_topic is not None:
                    topic = own_topic  # as written: no card moved
                self.topic_of[card_id] = topic
                self.margin_of[card_id] = margin
                self.second_of[card_id] = second
                self.members[(deck, topic)].append(card_id)
            for seed in seeds:
                judged = [card_id for card_id in seed.card_ids if card_id in self.topic_of]
                self.coherence[seed.key] = (
                    sum(self.topic_of[card_id] == seed.topic_id for card_id in judged) / len(judged) if judged else math.nan
                )

    # --- the decision for the state ------------------------------------------

    @property
    def hidden(self) -> set[str]:
        """The cards between two topics: a margin below the threshold (a measure; the app shows them)."""
        return {card_id for card_id, margin in self.margin_of.items() if margin < self.margin}

    def topics_of_deck(self, deck: str) -> list[TopicSeed]:
        return [seed for seed in self.seeds if seed.deck == deck]

    # --- the measures against the scripted topics ----------------------------

    def agreement_of(self, card_ids: Iterable[str]) -> float:
        """The share of these cards whose model topic is their scripted one (NaN without cards)."""
        judged = [card_id for card_id in card_ids if card_id in self.topic_of and card_id in self.scripted_of]
        if not judged:
            return math.nan
        return sum(self.topic_of[card_id] == self.scripted_of[card_id] for card_id in judged) / len(judged)

    @property
    def agreement(self) -> float:
        return self.agreement_of(self.topic_of)

    def kept(self, threshold: float) -> tuple[float, float]:
        """The margin table's row: (the share of the cards with a margin of at least `threshold`, their agreement)."""
        if not self.topic_of:
            return math.nan, math.nan
        kept = [card_id for card_id, margin in self.margin_of.items() if margin >= threshold]
        return len(kept) / len(self.topic_of), self.agreement_of(kept)

    def threshold_for_share(self, share: float) -> float:
        """The margin the slider's share stands for over the model's cards (share_threshold)."""
        return share_threshold(list(self.margin_of.values()), share)

    @property
    def moved(self) -> int:
        """The cards whose model topic is not their scripted one."""
        return sum(self.topic_of[card_id] != self.scripted_of.get(card_id) for card_id in self.topic_of)

    @property
    def emptied(self) -> list[Key]:
        """The topics that had scripted cards and keep none under the model (they keep their dots)."""
        return [seed.key for seed in self.seeds if seed.card_ids and not self.members[seed.key]]
