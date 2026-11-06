"""Offline measures of the models (no Dash, no state, nothing of the viewer): how good the neighbours are, judged by
an independent label (the deck a card belongs to) and by hand marks, the random floor the map shows without a
model, what a layout keeps of the neighbourhoods against that floor, and the judges of the topic model T1
(docs/draft/models/topic.md). All of it for evaluate.py.

* same_deck_precision: of a card's top-5 neighbours, how many come from the card's own deck, over all cards.
  Empty slots (fewer than 5 neighbours above the minimum score) count as misses. Random neighbours score about 2 %.
* kept_precision: the same over the kept neighbours only (what the user would actually see).
* coverage: the share of cards with at least one neighbour, and the number of different cards that appear as a
  neighbour at all (hub cards show up everywhere; a low number means they do).
* judged_precision: every judged card's hand precision@k, the share of its first k neighbours marked relevant;
  paired_bootstrap: the mean of per-card differences (a variant against the chosen one) with the percentile interval
  of the mean over resamples of the cards, so two variants are compared on the same cards, not by their two means.
* random_directions: points drawn uniformly on the unit sphere, the models' own draw for the random floor and the
  random layouts (the same distribution as the viewer's random positions on the surface, other digits).
* random_floor (B0): the same-deck precision of the k nearest cards at given random positions.
* mean_overlap and lift: the within-topic neighbour overlap at a fixed small k over the topics large enough for it,
  and how far a layout's overlap lies above the random floor on the way to 1.
* The topic judges, each on one deck's rows and scripted labels: agreement (the share of predicted labels that are
  the scripted ones, one minus the lessons' Hamming loss), nearest_labels (the label of the most similar other card,
  the helper's neighbour rule), leave_one_out_logistic (logistic regression per deck, every card predicted by a
  model fitted on the others) and chance_by_sizes (an assignment by the topic sizes).
"""

import math
from typing import Collection, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .layout import DIMENSIONS, as_dense, nearest_in_3d, neighbour_overlap
from .similar import TOP_K

LOGISTIC_ITERATIONS = 2000  # the per-deck logistic regression of the T1 measurement
BOOTSTRAP_DRAWS = 5000  # resamples of the judged cards for the paired difference's interval
INTERVAL = 0.95  # the interval's level


def _card_id(item) -> str:
    return item[0] if isinstance(item, tuple) else item


def same_deck_precision(neighbours: Mapping[str, Sequence], deck_of: Mapping[str, str], k: int = TOP_K) -> float:
    """Mean over all cards of (same-deck neighbours among the first k) / k. NaN without cards."""
    if not neighbours:
        return float("nan")
    hits = [
        sum(deck_of.get(_card_id(item)) == deck_of.get(card_id) for item in list(items)[:k]) / k
        for card_id, items in neighbours.items()
    ]
    return float(np.mean(hits))


def kept_precision(neighbours: Mapping[str, Sequence], deck_of: Mapping[str, str], k: int = TOP_K) -> float:
    """The share of the kept neighbour slots (up to k per card) that are same-deck. NaN without neighbours."""
    kept = [(card_id, _card_id(item)) for card_id, items in neighbours.items() for item in list(items)[:k]]
    if not kept:
        return float("nan")
    return float(np.mean([deck_of.get(other) == deck_of.get(card_id) for card_id, other in kept]))


def coverage(neighbours: Mapping[str, Sequence]) -> tuple[float, int]:
    """(share of cards with at least one neighbour, number of distinct cards that are someone's neighbour)."""
    if not neighbours:
        return float("nan"), 0
    with_neighbour = float(np.mean([len(items) > 0 for items in neighbours.values()]))
    distinct = {_card_id(item) for items in neighbours.values() for item in items}
    return with_neighbour, len(distinct)


def judged_precision(neighbours: Mapping[str, Sequence], marks: Mapping[str, Collection[str]], k: int = TOP_K) -> dict[str, float]:
    """Every judged card's hand precision@k: the share of its first k neighbours that are marked relevant (`marks`:
    card id -> the ids marked relevant, for the judged cards; a missing slot counts as a miss)."""
    return {
        card_id: sum(_card_id(item) in relevant for item in list(neighbours.get(card_id, []))[:k]) / k
        for card_id, relevant in marks.items()
    }


def paired_bootstrap(
    differences: Sequence[float], draws: int = BOOTSTRAP_DRAWS, seed: int = 42, level: float = INTERVAL
) -> tuple[float, float, float]:
    """The mean of the per-item differences and the percentile interval of that mean over `draws` resamples of the
    items with replacement (the paired bootstrap): (mean, low, high); NaN without items. Deterministic by the seed."""
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        return math.nan, math.nan, math.nan
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    tail = 100 * (1.0 - level) / 2
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def random_directions(count: int, seed: int = 42) -> np.ndarray:
    """`count` points drawn uniformly on the unit sphere, one (x, y, z) row each, by the seed: normalised Gaussian
    vectors, the same distribution as the viewer's random positions on the surface (other digits)."""
    generator = np.random.default_rng(seed)
    points = generator.standard_normal((count, DIMENSIONS))
    lengths = np.linalg.norm(points, axis=1, keepdims=True)
    while (lengths <= 1e-9).any():  # a vector at the origin has no direction: drawn again
        short = (lengths <= 1e-9).ravel()
        points[short] = generator.standard_normal((int(short.sum()), DIMENSIONS))
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
    return points / lengths


def nearest_by_position(ids: Sequence[str], points: np.ndarray, k: int = TOP_K) -> dict[str, list[str]]:
    """Every card's k nearest cards on a map (euclidean), as the map shows them."""
    nearest = nearest_in_3d(np.asarray(points, dtype=float), k)
    return {card_id: [ids[column] for column in nearest[row]] for row, card_id in enumerate(ids)}


def random_floor(ids: Sequence[str], deck_of: Mapping[str, str], points, k: int = TOP_K) -> float:
    """B0: the same-deck precision of the k nearest cards when all cards sit at the given random positions
    (`points`: one (x, y, z) row per id, in the order of `ids`)."""
    return same_deck_precision(nearest_by_position(list(ids), np.asarray(points, dtype=float), k), deck_of, k)


# --- the layouts against the random floor -----------------------------------------


def mean_overlap(layouts: Sequence[tuple], k: int) -> tuple[float, int]:
    """The mean neighbour overlap at a fixed k over the (vectors, points) pairs with at least k + 2 rows, and how
    many pairs that is; (NaN, 0) without one. With k + 1 rows every other row is a neighbour and the overlap is 1
    for any layout, so those pairs say nothing and are left out."""
    values = [neighbour_overlap(vectors, points, k) for vectors, points in layouts if len(as_dense(vectors)) >= k + 2]
    return (float(np.mean(values)) if values else math.nan), len(values)


def lift(value: float, floor: float) -> float:
    """How far a layout's overlap lies above the random floor on the way to 1: (value - floor) / (1 - floor), 0 for
    the floor itself and 1 for a layout that keeps every neighbour; NaN at a floor of 1 or without a value."""
    if math.isnan(value) or math.isnan(floor) or floor >= 1.0:
        return math.nan
    return (value - floor) / (1.0 - floor)


# --- the judges of the topic model ----------------------------------------------


def agreement(predicted: Sequence, scripted: Sequence) -> float:
    """The share of predictions that equal the scripted label (NaN without cards)."""
    if not len(scripted):
        return math.nan
    return float(np.mean([left == right for left, right in zip(predicted, scripted)]))


def nearest_labels(vectors, labels: Sequence) -> list:
    """The neighbour rule: every row takes the label of its most similar other row (by cosine; a tie goes to the
    lowest row). A single row keeps its own label."""
    matrix = as_dense(vectors)
    count = len(matrix)
    if count < 2:
        return list(labels)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    units = np.divide(matrix, lengths, out=np.zeros_like(matrix), where=lengths > 0)
    cosines = units @ units.T
    np.fill_diagonal(cosines, -np.inf)
    return [labels[int(np.argmax(cosines[row]))] for row in range(count)]


def leave_one_out_logistic(vectors, labels: Sequence, max_iter: int = LOGISTIC_ITERATIONS) -> list:
    """Logistic regression per deck, leave-one-out: every row's label predicted by a model fitted on the other rows
    (the only label left when they all share one)."""
    matrix = as_dense(vectors)
    labels = list(labels)
    count = len(matrix)
    predicted = []
    for row in range(count):
        others = [other for other in range(count) if other != row]
        classes = {labels[other] for other in others}
        if len(classes) < 2:
            predicted.append(next(iter(classes)) if classes else labels[row])
            continue
        model = LogisticRegression(max_iter=max_iter).fit(matrix[others], [labels[other] for other in others])
        predicted.append(model.predict(matrix[row : row + 1])[0])
    return predicted


def chance_by_sizes(labels: Sequence) -> float:
    """The agreement of an assignment drawn by the label sizes: the sum of the squared shares (NaN without cards)."""
    labels = list(labels)
    if not labels:
        return math.nan
    return float(sum((labels.count(label) / len(labels)) ** 2 for label in set(labels)))
