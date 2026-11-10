"""The navigation prompt P1 (no Dash, no state): a request typed by the learner, scored
against the topics by three routes over the index's vectors and answered with a target or "no match".

    request ──→ the index's vectoriser (SimilarityIndex.queries, one transform) ──→ one unit row
                │
                ├─ card route (few-shot, the nearest example): the cosine to every card → a topic's best member
                ├─ centroid route (few-shot, the mean of the examples): the cosine to a topic's full centroid
                └─ text route (zero-shot, the category named): the cosine to a topic's headings
                │
                each route normalised at its own floor, (s − floor) / (1 − floor); the variant's mean ranks the topics
                │
                decision: the first topic by the mean is entered when the variant's few-shot route (the card route;
                the centroid route without it; the text route alone its own) clears its floor there; the card route's
                top card, the request's nearest card, is selected when it lies in that topic; else "no match"

* The topics are T1's membership (TopicModel.members, every deck of the index), the targets as the decks have
  them under the model: a topic's full centroid is the mean of its members' unit rows at unit length, its text
  its headings joined (TopicSeed.headings, the course file's), one transform at the fit. A topic without members
  scores 0 on the card and the centroid route and has no card; a topic without headings scores 0 on the text route.
* The floors come from the self set before any request: each card's question alone, its `Source:` line
  cut, as a request with the card held out (its row masked for the card route, its own topic's leave-one-out
  centroid for the centroid route; the text route has nothing to mask), its truth T1's topic. The card route's
  floor is the 10th percentile of a question's first neighbour's cosine among the other cards, the centroid
  route's floor the 10th percentile of a question's cosine to its own topic's leave-one-out
  centroid, the text route's the 10th percentile of a question's cosine to its own topic's text.
  A score clears a floor when it lies above it: a request with no term in common never
  clears the text route's floor of 0 under TF-IDF. No card bar: the nearest card is selected whenever it lies in the
  entered topic (the a priori median withheld a right card more often than not).
* ROUTES are the evaluation's five variants; DEFAULT_ROUTES the card route, which the choice rule keeps until
  another variant wins on the judged requests; a variant's label names the
  vectoriser and the routes (variant_label, the rows of prompt.csv). The model takes any index:
  the evaluation fits one on character n-grams for the card route's eleventh row.
* Deterministic: the same index, membership and request give the same answer; ties go to the topic first in the
  file, and among a topic's cards to the first in deck order.

The label of a set of cards (`label`), the text route turned around:

    card ids ──→ their unit rows in the index's space ──→ the full centroid (their mean, at unit length)
                 │
                 the cosine to every topic's headings (the text route's vectors) ──→ the nearest three topics
                 │
                 the label: the three topics' names, each its first heading, the longer ones cut with "…" to one
                 length until the three fit LABEL_TARGET, joined by LABEL_SEPARATOR

* A topic counts only with a cosine above 0, so a set that shares nothing with any topic's headings gets no label
  (an empty text), and one that shares something with one or two topics gets one or two names. Cards the index
  does not hold are left out; none left, no label. Ties go to the topic first in the file.
* A name's first heading is the part before its first ";" (a topic's name is made of its headings, two at most
  joined by "; ", a " …" where it has more), `first_heading`. The cut (`label_text`) gives every name the room of
  the longest length all three can keep, so a short name stays whole and only the longer ones lose their ends.
"""

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse

from .cards import Card
from . import embeddings, similar
from .topic_model import Key, TopicModel
from .zero_shot import SOURCE_LINE

CARD, CENTROID, TEXT = "card", "centroid", "text"
ROUTE_NAMES = (CARD, CENTROID, TEXT)
# The variants: card, centroid, text, card + text and all three
ROUTES: dict[str, tuple[str, ...]] = {
    "card": (CARD,), "centroid": (CENTROID,), "text": (TEXT,), "card+text": (CARD, TEXT), "all": (CARD, CENTROID, TEXT),
}
DEFAULT_VARIANT = "card"  # the variant the app runs, until the choice rule names another
DEFAULT_ROUTES = ROUTES[DEFAULT_VARIANT]
FLOOR_PERCENTILE = 10  # every route's floor: this percentile of its own scores over the self set
NEAREST = 3  # the targets an answer lists, the decision's own first
HEADING_SEPARATOR = " | "  # a topic's text: its headings joined, as the course table's cell holds them
MODEL_NAME = "P1 navigation prompt"
LABEL_TARGET = 70  # the label's length at most, in characters: the three names cut until they fit
LABEL_SEPARATOR = " · "  # between the label's names
CUT_MARK = "…"  # the end of a cut name
NAME_PARTS = ";"  # between the headings of a topic's name

SELECTED = "card selected"
ENTERED = "topic entered"
NO_MATCH = "no match"
MESSAGES = {SELECTED: SELECTED, ENTERED: ENTERED, NO_MATCH: "no match, the three nearest below the floor"}


def family_of(vectoriser) -> str:
    """A vectoriser's name in a variant's label: "TF-IDF", "minilm", "hybrid-minilm"."""
    names = {model_name: name for name, model_name in embeddings.ENCODERS.items()}
    if isinstance(vectoriser, embeddings.HybridVectoriser):
        return f"hybrid-{names.get(vectoriser.second.model_name, vectoriser.second.model_name)}"
    if isinstance(vectoriser, embeddings.SentenceEncoder):
        return names.get(vectoriser.model_name, vectoriser.model_name)
    return "TF-IDF"


def variant_label(vectoriser, name: str = DEFAULT_VARIANT) -> str:
    """A variant's label in prompt.csv: the vectoriser's name and the routes' ("hybrid-minilm: card")."""
    return f"{family_of(vectoriser)}: {name}"


def gate_of(routes: Sequence[str]) -> str:
    """The route that decides a variant's entry: the card route, the centroid route without it, else the text route."""
    return next(route for route in (CARD, CENTROID, TEXT) if route in routes)


def checked(routes: Sequence[str]) -> tuple[str, ...]:
    """The variant's routes as a tuple, refused when empty or naming an unknown route."""
    routes = tuple(routes)
    unknown = [route for route in routes if route not in ROUTE_NAMES]
    if not routes or unknown:
        raise ValueError(f"a variant is one or more of {', '.join(ROUTE_NAMES)}, not {routes!r}")
    return routes


def request_text(question: str) -> str:
    """A card's question as a proxy request: its `Source:` line cut (it names the notebook, a template shortcut)."""
    return SOURCE_LINE.sub("", question or "").strip()


def first_heading(name: str) -> str:
    """A topic's name to its first heading: the part before its first ";", without the " …" that marks more
    headings; the whole name, stripped, where that part is empty."""
    name = " ".join((name or "").split())
    first = name.split(NAME_PARTS, 1)[0].rstrip(f" {CUT_MARK}").strip()
    return first or name


def cut(text: str, length: int) -> str:
    """The text, or its start with CUT_MARK at its end where it is longer than `length` (at least the mark)."""
    if len(text) <= length:
        return text
    return text[: max(length - 1, 0)].rstrip() + CUT_MARK


def label_text(names: Sequence[str], target: int = LABEL_TARGET) -> str:
    """The names joined by LABEL_SEPARATOR within `target` characters: every name keeps at most the longest length
    all of them can keep together, so the short ones stay whole and the longer ones are cut with CUT_MARK; empty
    without names. A target too small for a mark per name gives every name its mark alone."""
    names = [name for name in names if name]
    if not names:
        return ""
    room = target - len(LABEL_SEPARATOR) * (len(names) - 1)
    longest = max(len(name) for name in names)
    length = next(
        (length for length in range(longest, 0, -1) if sum(min(len(name), length) for name in names) <= room), 1
    )
    return LABEL_SEPARATOR.join(cut(name, length) for name in names)


@dataclass(frozen=True)
class Label:
    """The label of a set of cards: the nearest topics in rank order (three at most, none without a cosine above
    0) and the text, their names within the target length; no keys and an empty text for no label."""

    keys: tuple[Key, ...] = ()
    text: str = ""


@dataclass(frozen=True)
class Floors:
    """The floors of the three routes, percentiles over the self set."""

    card: float
    centroid: float
    text: float

    def of(self, route: str) -> float:
        return {CARD: self.card, CENTROID: self.centroid, TEXT: self.text}[route]


@dataclass(frozen=True)
class TopicScore:
    """A topic's place in a request's ranking: its key, the variant's mean of the normalised scores (the ranking),
    every route's cosine (all three, whatever the variant) and its best card by the card route."""

    key: Key  # (deck name, topic id)
    mean: float
    scores: Mapping[str, float] = field(hash=False)
    card: Optional[str]  # the member nearest to the request; None for a topic without members


@dataclass(frozen=True)
class Answer:
    """The decision on a request: every topic in rank order (the variant's mean), the decision and the selected
    card; the targets are the nearest three, the decision's own first."""

    text: str
    routes: tuple[str, ...]
    decision: str  # SELECTED, ENTERED or NO_MATCH
    ranking: tuple[TopicScore, ...]
    card: Optional[str] = None  # the selected card (SELECTED only)

    @property
    def navigate(self) -> bool:
        return self.decision != NO_MATCH

    @property
    def topic(self) -> Optional[Key]:
        """The entered topic; None for "no match"."""
        return self.ranking[0].key if self.navigate and self.ranking else None

    @property
    def targets(self) -> tuple[TopicScore, ...]:
        return self.ranking[:NEAREST]

    @property
    def message(self) -> str:
        return MESSAGES[self.decision]


@dataclass(frozen=True)
class SelfCase:
    """A card's question as a request with the card held out: its truth (T1's topic), the nearest three topics, the
    decision and the selected card."""

    card_id: str
    key: Key
    nearest: tuple[Key, ...]
    decision: str
    card: Optional[str]


@dataclass(frozen=True)
class _Scores:
    """The raw route scores of a batch of requests: the cosine to every card (held-out rows at -inf), and per topic
    the card route's best member's cosine, the centroid route's and the text route's."""

    cards: np.ndarray  # (requests, cards)
    routes: dict[str, np.ndarray]  # route -> (requests, topics)


class PromptModel:
    """P1 fitted on the index and T1's membership: the topics' full centroids and texts, the floors from the
    self set; `find` answers a request, `self_set` the cards' own questions for the evaluation, `label` names a set
    of cards by the nearest three topics. `cards` are the index's cards (their questions make the self set)."""

    def __init__(self, index: similar.SimilarityIndex, topic_model: TopicModel, cards: Sequence[Card]) -> None:
        self.index = index
        self.topic_model = topic_model
        self.seeds = list(topic_model.seeds)
        self.keys: list[Key] = [seed.key for seed in self.seeds]
        self._place = {key: place for place, key in enumerate(self.keys)}
        self._row_of = {card_id: row for row, card_id in enumerate(index.ids)}
        self.members: list[list[int]] = [
            [self._row_of[card_id] for card_id in topic_model.members.get(key, []) if card_id in self._row_of]
            for key in self.keys
        ]
        self.topic_of_row: dict[int, int] = {row: place for place, rows in enumerate(self.members) for row in rows}
        width = index.vectors.shape[1]
        self._sums = np.zeros((len(self.keys), width))  # the members' unit rows summed: the full and the LOO centroids
        for place, rows in enumerate(self.members):
            if rows:
                self._sums[place] = np.asarray(index.vectors[rows].sum(axis=0)).ravel()
        lengths = np.linalg.norm(self._sums, axis=1)
        self.centroids = self._sums / np.where(lengths > 0, lengths, 1.0)[:, None]  # a topic without members: zeros
        self.texts = [HEADING_SEPARATOR.join(seed.headings) for seed in self.seeds]
        self.text_vectors = index.queries(self.texts) if self.keys else sparse.csr_matrix((0, width))
        self.label_names = [first_heading(seed.name) for seed in self.seeds]  # the names a label is made of
        # The self set: every card of T1's membership whose question is not empty without its Source: line
        by_id = {card.id: card for card in cards}
        self.self_ids = [
            index.ids[row] for row in sorted(self.topic_of_row)
            if index.ids[row] in by_id and request_text(by_id[index.ids[row]].question)
        ]
        self.self_questions = [request_text(by_id[card_id].question) for card_id in self.self_ids]
        self._self_rows = [self._row_of[card_id] for card_id in self.self_ids]
        self._self = self._score(index.queries(self.self_questions), self._self_rows) if self.self_ids else None
        self.floors = self._floors()

    # --- the routes -------------------------------------------------------------

    def _score(self, requests: sparse.csr_matrix, held_out: Optional[Sequence[int]] = None) -> _Scores:
        """The raw scores of the requests' unit rows; `held_out` the card row each request stands in for."""
        count = requests.shape[0]
        cards = np.asarray((requests @ self.index.vectors.T).todense(), dtype=float).reshape(count, len(self.index.ids))
        centroid = np.asarray(requests @ self.centroids.T, dtype=float).reshape(count, len(self.keys))
        if held_out is not None:
            for number, row in enumerate(held_out):
                cards[number, row] = -np.inf
                place = self.topic_of_row.get(row)
                if place is not None:  # its own topic's leave-one-out centroid
                    rest = self._sums[place] - np.asarray(self.index.vectors[row].todense()).ravel()
                    length = float(np.linalg.norm(rest))
                    own = np.asarray(requests[number] @ rest).ravel()[0]
                    centroid[number, place] = own / length if length > 1e-9 else 0.0
        text = np.asarray((requests @ self.text_vectors.T).todense(), dtype=float).reshape(count, len(self.keys))
        best = np.zeros((count, len(self.keys)))
        for place, rows in enumerate(self.members):
            if rows:
                best[:, place] = cards[:, rows].max(axis=1)
        best[~np.isfinite(best)] = 0.0  # a topic whose only member is held out: no example left
        return _Scores(cards, {CARD: best, CENTROID: centroid, TEXT: text})

    def _floors(self) -> Floors:
        if self._self is None:
            return Floors(0.0, 0.0, 0.0)
        cards = self._self.cards
        first = cards.max(axis=1)
        first = first[np.isfinite(first)]
        own = [self.topic_of_row[row] for row in self._self_rows]
        numbers = np.arange(len(own))
        centroid = self._self.routes[CENTROID][numbers, own]
        text = self._self.routes[TEXT][numbers, own]

        def percentile(values: np.ndarray, rank: float) -> float:
            return float(np.percentile(values, rank)) if len(values) else 0.0

        return Floors(
            percentile(first, FLOOR_PERCENTILE), percentile(centroid, FLOOR_PERCENTILE),
            percentile(text, FLOOR_PERCENTILE),
        )

    def _means(self, scores: _Scores, routes: tuple[str, ...]) -> np.ndarray:
        """The variant's mean of the routes' scores, each normalised at its own floor: (requests, topics)."""
        total = np.zeros_like(scores.routes[CARD])
        for route in routes:
            floor = self.floors.of(route)
            total += (scores.routes[route] - floor) / max(1.0 - floor, 1e-9)
        return total / len(routes)

    def _decide(
        self, scores: _Scores, means: np.ndarray, number: int, routes: tuple[str, ...]
    ) -> tuple[list[int], str, Optional[str]]:
        """A request's ranking (topic places, by the mean; ties in file order), its decision and its selected card."""
        order = list(np.argsort(-means[number], kind="stable"))
        if not order:
            return order, NO_MATCH, None
        first = order[0]
        gate = gate_of(routes)
        if not scores.routes[gate][number, first] > self.floors.of(gate):  # clears: above it, so 0 never clears 0
            return order, NO_MATCH, None
        if CARD in routes:
            row = int(np.argmax(scores.cards[number]))  # the nearest card: selected when it lies in the entered topic
            if self.topic_of_row.get(row) == first:
                return order, SELECTED, self.index.ids[row]
        return order, ENTERED, None

    def _best_card(self, scores: _Scores, number: int, place: int) -> Optional[str]:
        rows = self.members[place]
        if not rows:
            return None
        values = scores.cards[number, rows]
        if not np.isfinite(values).any():
            return None
        return self.index.ids[rows[int(np.argmax(values))]]

    def score(self, text: str, routes: Sequence[str] = DEFAULT_ROUTES) -> list[TopicScore]:
        """Every topic for the request, by the variant's mean (ties in file order), with every route's cosine."""
        return list(self.find(text, routes).ranking)

    def find(self, text: str, routes: Sequence[str] = DEFAULT_ROUTES) -> Answer:
        """The answer to a request: the ranking, the decision (the first topic entered where the variant's
        few-shot route clears its floor, its nearest card selected where the card route's top card lies in it) and
        the selected card."""
        routes = checked(routes)
        scores = self._score(self.index.queries([text or ""]))
        means = self._means(scores, routes)
        order, decision, card = self._decide(scores, means, 0, routes)
        ranking = tuple(
            TopicScore(
                self.keys[place], float(means[0, place]),
                {route: float(scores.routes[route][0, place]) for route in ROUTE_NAMES}, self._best_card(scores, 0, place),
            )
            for place in order
        )
        return Answer(text, routes, decision, ranking, card)

    def label(self, card_ids: Iterable[str], target: int = LABEL_TARGET) -> Label:
        """The label of a set of cards: their full centroid (the mean of their unit rows at unit length) against
        every topic's headings, the nearest three topics with a cosine above 0 (ties in file order) and their names,
        each its first heading, cut to fit `target` (label_text). Cards the index does not hold are left out; none
        left, or no topic above 0, is no label."""
        rows = [self._row_of[card_id] for card_id in dict.fromkeys(card_ids) if card_id in self._row_of]
        if not rows or not self.keys:
            return Label()
        centroid = np.asarray(self.index.vectors[rows].sum(axis=0), dtype=float).ravel()
        length = float(np.linalg.norm(centroid))
        if length <= 1e-12:
            return Label()
        scores = np.asarray(self.text_vectors @ (centroid / length), dtype=float).ravel()
        order = [int(place) for place in np.argsort(-scores, kind="stable") if scores[place] > 0][:NEAREST]
        return Label(
            tuple(self.keys[place] for place in order), label_text([self.label_names[place] for place in order], target)
        )

    def self_set(self, routes: Sequence[str] = DEFAULT_ROUTES) -> list[SelfCase]:
        """Every card's question (its Source: line cut) as a request with the card held out, for the evaluation."""
        routes = checked(routes)
        if self._self is None:
            return []
        means = self._means(self._self, routes)
        cases = []
        for number, card_id in enumerate(self.self_ids):
            order, decision, card = self._decide(self._self, means, number, routes)
            truth = self.keys[self.topic_of_row[self._self_rows[number]]]
            cases.append(SelfCase(card_id, truth, tuple(self.keys[place] for place in order[:NEAREST]), decision, card))
        return cases
