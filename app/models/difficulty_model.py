"""The difficulty model D1 (no Dash, no state, nothing of the viewer): the course's difficulty (easy, medium, hard) of
the flattened quizzes that have none, learned from the quizzes that have one (docs/draft/helper/roadmap.md, step 10).

    rated quizzes (stack 3) ───┐
                               ├→ features → logistic regression → score 0..2 ─┬→ cross-validation: deck shares → metrics ± sd
    unrated quizzes (stack 2) ─┘                                                └→ test: deck shares → predicted levels
    plain flashcards (stack 1, and the quiz file's flashcards, stack 4): not in the first run (another kind of card,
    rated differently later)

* Features come from a card's own text and the fitted similarity index only: text statistics, its centrality and
  neighbour scores, and (as a variant) its TF-IDF row. Never the card order, the ids, the deck or the stack: the
  course lists every deck's rated quizzes easy → medium → hard, so the order would score perfectly in training and
  mean nothing for a new card. The rated cards are put into an order of their texts before the cross-validation, so
  not even the folds depend on the course order.
* A difficulty is relative to its deck (every deck's 12 rated quizzes are 8 easy, 3 medium and 1 hard), so labels
  come from each deck's shares: the cross-validation ranks each deck's out-of-fold scores and cuts them by the deck's
  own counts; the unrated quizzes of a deck are ranked among themselves, and the card at rank i of n takes the band
  its midpoint (i - 1/2)/n falls into (hard in the top share, medium in the next, easy below).
* Metrics are out of fold (repeated stratified k-fold), as mean ± sd over the repeats, next to two baselines: always
  "easy", and the similarity baseline as a classifier (the nearest rated cards by cosine). The same cross-validation
  also gives every deck's own metrics (its rated quizzes only, under the course-wide model) for the topics view, and
  a topic's part counts the topic's quizzes with the deck's metrics for the cards view (DifficultyReport.for_topic).
"""

import dataclasses
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..cards import STACK_QUIZ, Card
from . import similar

LEVELS = ("easy", "medium", "hard")
LEVEL_OF = {level: number for number, level in enumerate(LEVELS)}
# The shares of a deck's 12 rated quizzes (8 easy, 3 medium, 1 hard): the bands of a deck without rated quizzes
COURSE_SHARES = (8 / 12, 3 / 12, 1 / 12)
N_SPLITS = 5
N_REPEATS = 3
SEED = 42
NEAREST = 10  # the similarity baseline: the rated cards it votes with
FEATURE_SETS = ("text", "text+course", "text+course+tfidf")
DEFAULT_FEATURES = "text+course"
TEXT_FEATURES = (
    "question words (log)", "answer words (log)", "explanation words (log)", "code in the question",
    "digits in the question", "scenario words", "commas in the question",
)
COURSE_FEATURES = ("centrality", "first neighbour", "fifth neighbour")
SCENARIO = re.compile(r"\b(why|which|best|most likely|scenario|would)\b", re.IGNORECASE)
METRICS = ("macro_f1", "balanced_accuracy", "kappa", "accuracy", "spearman", "recall_easy", "recall_medium", "recall_hard")

Spread = tuple[float, float]  # (mean, sd) over the repeats


def split(cards: Sequence[Card]) -> tuple[list[Card], list[Card]]:
    """(rated, unrated): the cards with a difficulty, and the flattened quizzes (stack 2) without one. Cards
    are in neither, the lecture notebooks' (stack 1) and the quiz file's (stack 4) alike: they are not trained on
    and not predicted in the first run."""
    rated = [card for card in cards if card.difficulty in LEVEL_OF]
    unrated = [card for card in cards if card.difficulty not in LEVEL_OF and card.stack == STACK_QUIZ]
    return rated, unrated


def _words(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


def text_statistics(card: Card) -> list[float]:
    """The card's text statistics (TEXT_FEATURES): word counts (log), code, digits and scenario words in the question,
    and its commas. Longer, scenario-like questions are the harder ones (docs/draft/helper/concept.md)."""
    question = card.question
    return [
        float(np.log1p(_words(question))),
        float(np.log1p(_words(card.answer))),
        float(np.log1p(_words(card.explain))),
        float("`" in question),
        float(bool(re.search(r"\d", question))),
        float(bool(SCENARIO.search(question))),
        float(question.count(",")),
    ]


def course_features(index: similar.SimilarityIndex, ids: Sequence[str]) -> np.ndarray:
    """COURSE_FEATURES of the cards: the cosine to their deck vector (how central a card is to its deck) and the
    cosine of their first and fifth neighbour (0 when there is none above the minimum)."""
    names, deck_matrix = similar.deck_vectors(index.decks, index.vectors)
    deck_row = {name: row for row, name in enumerate(names)}
    row_of = {card_id: row for row, card_id in enumerate(index.ids)}
    rows = [row_of[card_id] for card_id in ids]
    vectors = index.vectors[rows]
    centrality = np.asarray(
        [float(vectors[position].dot(deck_matrix[deck_row[index.decks[row]]])[0]) for position, row in enumerate(rows)]
    )
    first, fifth = [], []
    for card_id in ids:
        scores = [score for _, score in index.related(card_id)]
        first.append(scores[0] if scores else 0.0)
        fifth.append(scores[4] if len(scores) > 4 else 0.0)
    return np.column_stack([centrality, first, fifth]) if ids else np.zeros((0, len(COURSE_FEATURES)))


@dataclass
class Features:
    dense: np.ndarray  # text statistics (and course features), not yet scaled
    rows: Optional[sparse.csr_matrix]  # the TF-IDF rows, or None

    def take(self, positions) -> "Features":
        return Features(self.dense[positions], self.rows[positions] if self.rows is not None else None)


def features(cards: Sequence[Card], index: similar.SimilarityIndex, feature_set: str = DEFAULT_FEATURES) -> Features:
    """The feature matrix of the cards for one of FEATURE_SETS; nothing in it depends on where a card is in the course."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {FEATURE_SETS}")
    ids = [card.id for card in cards]
    dense = np.asarray([text_statistics(card) for card in cards], dtype=float).reshape(len(cards), len(TEXT_FEATURES))
    if feature_set != "text":
        dense = np.column_stack([dense, course_features(index, ids)])
    rows = None
    if feature_set.endswith("tfidf"):
        row_of = {card_id: row for row, card_id in enumerate(index.ids)}
        rows = index.vectors[[row_of[card_id] for card_id in ids]]
    return Features(dense, rows)


class Classifier:
    """Logistic regression with balanced class weights on the scaled statistics (and the TF-IDF rows); its score is
    the expected level, P(medium) + 2 · P(hard)."""

    def fit(self, data: Features, levels: np.ndarray) -> "Classifier":
        self.scaler = StandardScaler().fit(data.dense)
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(self._matrix(data), levels)
        return self

    def _matrix(self, data: Features):
        scaled = self.scaler.transform(data.dense)
        return scaled if data.rows is None else sparse.hstack([data.rows, sparse.csr_matrix(scaled)]).tocsr()

    def predict_proba(self, data: Features) -> np.ndarray:
        """Probabilities in the order of LEVELS (a level missing from the training data gets 0)."""
        found = self.model.predict_proba(self._matrix(data))
        proba = np.zeros((found.shape[0], len(LEVELS)))
        proba[:, self.model.classes_] = found
        return proba

    def score(self, data: Features) -> np.ndarray:
        return self.predict_proba(data) @ np.arange(len(LEVELS))


def _order(cards: Sequence[Card], scores: np.ndarray) -> list[int]:
    """Positions by score, highest first; a tie goes by the card's text, never by its place in the course."""
    return sorted(range(len(cards)), key=lambda position: (-scores[position], cards[position].question, cards[position].answer))


def share_labels(cards: Sequence[Card], scores: np.ndarray, counts: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Each deck's cards ranked by score and cut by the deck's counts (easy, medium, hard): the top `hard` cards
    hard, the next `medium` medium, the rest easy (the cross-validation of the rated cards)."""
    labels = np.zeros(len(cards), dtype=int)
    for deck in {card.deck for card in cards}:
        members = [position for position, card in enumerate(cards) if card.deck == deck]
        ranked = [members[position] for position in _order([cards[m] for m in members], scores[members])]
        _, medium, hard = counts[deck]
        for rank, position in enumerate(ranked):
            labels[position] = LEVEL_OF["hard"] if rank < hard else LEVEL_OF["medium"] if rank < hard + medium else 0
    return labels


def band(rank: int, n: int, shares: Sequence[float] = COURSE_SHARES) -> int:
    """The level of the card at `rank` (1 = the highest score) of `n`: the band its midpoint (rank - 1/2)/n falls
    into, hard below the hard share, medium below hard plus medium, easy above."""
    midpoint = (rank - 0.5) / n
    _, medium, hard = shares
    return LEVEL_OF["hard"] if midpoint < hard else LEVEL_OF["medium"] if midpoint < hard + medium else 0


def band_labels(cards: Sequence[Card], scores: np.ndarray, shares: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Each deck's cards ranked by score among themselves, each taking its band of the deck's shares (the unrated
    quizzes: a deck's single extra quiz comes out easy, a deck of 12 comes out 8 / 3 / 1)."""
    labels = np.zeros(len(cards), dtype=int)
    for deck in {card.deck for card in cards}:
        members = [position for position, card in enumerate(cards) if card.deck == deck]
        ranked = [members[position] for position in _order([cards[m] for m in members], scores[members])]
        for rank, position in enumerate(ranked, start=1):
            labels[position] = band(rank, len(ranked), shares.get(deck, COURSE_SHARES))
    return labels


def cut_point_labels(
    cards: Sequence[Card], scores: np.ndarray, rated: Sequence[Card], rated_scores: np.ndarray,
    counts: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """The comparison rule: a card is hard when it scores at least as high as its deck's hard rated cards, medium
    when at least as high as the deck's medium ones; a deck without rated cards takes its bands (band_labels)."""
    labels = band_labels(cards, scores, {})
    for position, card in enumerate(cards):
        deck_scores = np.sort([rated_scores[i] for i, other in enumerate(rated) if other.deck == card.deck])[::-1]
        if not len(deck_scores):
            continue
        _, medium, hard = counts[card.deck]
        if hard and scores[position] >= deck_scores[hard - 1]:
            labels[position] = LEVEL_OF["hard"]
        elif medium and scores[position] >= deck_scores[hard + medium - 1]:
            labels[position] = LEVEL_OF["medium"]
        else:
            labels[position] = 0
    return labels


def measure(levels: np.ndarray, labels: np.ndarray, scores: Optional[np.ndarray] = None) -> dict[str, float]:
    """The METRICS of one set of labels against the true levels."""
    recalls = recall_score(levels, labels, labels=list(range(len(LEVELS))), average=None, zero_division=0)
    spearman = float("nan")
    if scores is not None and np.ptp(scores) > 0:
        # Scores equal to 9 places are ties (cards with the same features score the same up to the summation order)
        spearman = float(spearmanr(np.round(scores, 9), levels).correlation)
    return {
        "macro_f1": float(f1_score(levels, labels, labels=list(range(len(LEVELS))), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(levels, labels)),
        "kappa": float(cohen_kappa_score(levels, labels, weights="quadratic")) if len(set(labels)) > 1 or len(set(levels)) > 1 else 0.0,
        "accuracy": float(accuracy_score(levels, labels)),
        "spearman": spearman,
        **{f"recall_{level}": float(recall) for level, recall in zip(LEVELS, recalls)},
    }


def _spread(runs: Sequence[Mapping[str, float]]) -> dict[str, Spread]:
    """Mean and sample sd of every metric over the repeats (sd 0 for a single run)."""
    result = {}
    for name in METRICS:
        values = np.array([run[name] for run in runs], dtype=float)
        if np.all(np.isnan(values)):
            result[name] = (float("nan"), float("nan"))
            continue
        values = values[~np.isnan(values)]
        result[name] = (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return result


def nearest_scores(similarities: np.ndarray, train: np.ndarray, test: np.ndarray, levels: np.ndarray, k: int = NEAREST) -> np.ndarray:
    """The similarity baseline: each test card's expected level from its k most similar training cards, weighted by
    their cosine (the plain mean of their levels when none is similar at all)."""
    scores = np.zeros(len(test))
    for position, row in enumerate(test):
        cosines = similarities[row, train]
        nearest = np.argsort(-cosines, kind="stable")[: min(k, len(train))]
        weights = np.clip(cosines[nearest], 0.0, None)
        values = levels[train][nearest]
        scores[position] = float(values @ weights / weights.sum()) if weights.sum() > 0 else float(values.mean())
    return scores


@dataclass
class DifficultyReport:
    """What D1 measured and predicted: the metrics of the cross-validation (mean ± sd over the repeats) with the
    deck shares, the same with the most likely class, the baselines, the confusion matrix (rows: true level,
    columns: predicted, each row's shares, the mean over the repeats) and the predicted level of each unrated quiz.

    The course's report holds one report of the same kind per deck (`decks`, `for_deck`): the deck's rated quizzes
    measured out of fold within the same cross-validation (the course-wide model, the deck's own cards only), and
    its predicted quizzes. A deck's part gives a topic's part (`for_topic`): the topic's rated levels and
    predictions, counted from its card ids, with the deck's metrics."""

    feature_set: str
    rated_counts: tuple[int, int, int]
    unrated: int
    folds: int
    repeats: int
    metrics: dict[str, Spread] = field(default_factory=dict)
    most_likely: dict[str, Spread] = field(default_factory=dict)
    baselines: dict[str, dict[str, Spread]] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    predictions: dict[str, str] = field(default_factory=dict)  # card id -> level, by the deck shares
    scores: dict[str, float] = field(default_factory=dict)  # card id -> score of the unrated quizzes
    cut_point_predictions: dict[str, str] = field(default_factory=dict)  # the comparison rule
    seconds: float = 0.0
    note: str = ""
    scope: Optional[str] = None  # None: the whole course; a deck name: that deck's part; a topic name: its part
    decks: dict[str, "DifficultyReport"] = field(default_factory=dict)  # deck name -> its part (the course's report)
    levels: dict[str, str] = field(default_factory=dict)  # card id -> the authored level of the rated quizzes
    unrated_ids: tuple[str, ...] = ()  # the unrated quizzes (predicted or not)
    parent: Optional["DifficultyReport"] = None  # a topic's part: the deck's part whose metrics it shows

    def for_deck(self, name: str) -> "DifficultyReport":
        """The deck's part; a deck without quizzes gets an empty one that says so."""
        part = self.decks.get(name)
        if part is None:
            part = DifficultyReport(self.feature_set, (0, 0, 0), 0, self.folds, self.repeats, scope=name,
                                    note=self.note or "This deck has no quizzes: nothing to measure or predict.")
        return part

    def for_topic(self, name: str, card_ids: Iterable[str]) -> "DifficultyReport":
        """The topic's part (the cards view): its rated levels and its unrated quizzes with their predictions,
        counted from the topic's card ids, with this report's metrics, baselines and matrix, which stay the deck's
        (a topic's few rated quizzes are too few to measure). `parent` is this report, for the back's subtitle."""
        ids = set(card_ids)
        levels = {card_id: level for card_id, level in self.levels.items() if card_id in ids}
        unrated_ids = tuple(card_id for card_id in self.unrated_ids if card_id in ids)
        return dataclasses.replace(
            self, scope=name, rated_counts=self.counts(levels), unrated=len(unrated_ids), levels=levels, unrated_ids=unrated_ids,
            predictions={card_id: level for card_id, level in self.predictions.items() if card_id in ids},
            scores={card_id: score for card_id, score in self.scores.items() if card_id in ids},
            cut_point_predictions={card_id: level for card_id, level in self.cut_point_predictions.items() if card_id in ids},
            decks={}, parent=self,
        )

    @property
    def rated(self) -> int:
        return sum(self.rated_counts)

    @property
    def measured(self) -> bool:
        return bool(self.metrics)

    @staticmethod
    def counts(levels: Mapping[str, str]) -> tuple[int, int, int]:
        values = list(levels.values())
        return tuple(values.count(level) for level in LEVELS)

    @property
    def predicted_counts(self) -> tuple[int, int, int]:
        return self.counts(self.predictions)


def fit_difficulty(
    cards: Sequence[Card],
    index: similar.SimilarityIndex,
    feature_set: str = DEFAULT_FEATURES,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> DifficultyReport:
    """Measure D1 out of fold on the rated cards, then fit it on all of them and label the unrated quizzes by their
    deck's shares. `index` is the similarity index fitted on the course (CourseModel.index). With too few rated cards
    for a cross-validation the report has no metrics; with fewer than two levels it predicts nothing either."""
    started = time.perf_counter()
    rated, unrated = split(cards)
    rated = sorted(rated, key=lambda card: (card.question, card.answer, card.deck))  # not the course order
    levels = np.array([LEVEL_OF[card.difficulty] for card in rated], dtype=int)
    rated_counts = tuple(int((levels == number).sum()) for number in range(len(LEVELS)))
    rated_decks = np.array([card.deck for card in rated])
    counts = {  # deck -> its rated quizzes per level (easy, medium, hard)
        deck: tuple(int(((levels == number) & (rated_decks == deck)).sum()) for number in range(len(LEVELS)))
        for deck in set(rated_decks)
    }
    present = [count for count in rated_counts if count]
    folds = min(n_splits, min(present)) if present else 0
    report = DifficultyReport(
        feature_set, rated_counts, len(unrated), folds if folds >= 2 else 0, n_repeats if folds >= 2 else 0,
        levels={card.id: card.difficulty for card in rated}, unrated_ids=tuple(card.id for card in unrated),
    )
    if len(present) < 2:
        report.note = "Too few rated quizzes: no model."
        report.seconds = time.perf_counter() - started
        return report

    data = features(rated, index, feature_set)
    mean_scores = np.zeros(len(rated))
    deck_rows = {deck: np.flatnonzero(rated_decks == deck) for deck in counts}
    deck_runs = {deck: {"D1": [], "always easy": [], "similarity": [], "confusion": []} for deck in counts}
    if folds >= 2:
        row_of = {card_id: row for row, card_id in enumerate(index.ids)}
        rated_rows = index.vectors[[row_of[card.id] for card in rated]]
        similarities = similar.cosine_similarities(rated_rows)
        np.fill_diagonal(similarities, -np.inf)
        runs, likely_runs, nearest_runs, easy_runs, confusions = [], [], [], [], []
        splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_repeats, random_state=seed)
        oof = np.zeros(len(rated))
        oof_proba = np.zeros((len(rated), len(LEVELS)))
        oof_nearest = np.zeros(len(rated))
        for number, (train, test) in enumerate(splitter.split(np.zeros(len(rated)), levels)):
            model = Classifier().fit(data.take(train), levels[train])
            oof_proba[test] = model.predict_proba(data.take(test))
            oof[test] = oof_proba[test] @ np.arange(len(LEVELS))
            oof_nearest[test] = nearest_scores(similarities, train, test, levels)
            if (number + 1) % folds == 0:  # a repeat is complete: every rated card has its out-of-fold score
                labels = share_labels(rated, oof, counts)
                nearest_labels = share_labels(rated, oof_nearest, counts)
                easy_labels = np.zeros(len(rated), dtype=int)
                runs.append(measure(levels, labels, oof))
                likely_runs.append(measure(levels, oof_proba.argmax(axis=1), oof))
                nearest_runs.append(measure(levels, nearest_labels, oof_nearest))
                easy_runs.append(measure(levels, easy_labels))
                confusions.append(confusion_matrix(levels, labels, labels=list(range(len(LEVELS))), normalize="true"))
                for deck, rows in deck_rows.items():  # the same out-of-fold labels, the deck's quizzes only
                    deck_runs[deck]["D1"].append(measure(levels[rows], labels[rows], oof[rows]))
                    deck_runs[deck]["similarity"].append(measure(levels[rows], nearest_labels[rows], oof_nearest[rows]))
                    deck_runs[deck]["always easy"].append(measure(levels[rows], easy_labels[rows]))
                    deck_runs[deck]["confusion"].append(
                        confusion_matrix(levels[rows], labels[rows], labels=list(range(len(LEVELS))), normalize="true")
                    )
                mean_scores += oof / n_repeats
        report.metrics = _spread(runs)
        report.most_likely = _spread(likely_runs)
        report.baselines = {"always easy": _spread(easy_runs), "similarity": _spread(nearest_runs)}
        report.confusion = np.mean(confusions, axis=0)
    else:
        report.note = "Too few rated quizzes per level for a cross-validation: no metrics."

    model = Classifier().fit(data, levels)
    if folds < 2:
        mean_scores = model.score(data)
    if unrated:
        scores = model.score(features(unrated, index, feature_set))
        shares = {deck: tuple(count / sum(values) for count in values) for deck, values in counts.items()}
        labels = band_labels(unrated, scores, shares)
        cut = cut_point_labels(unrated, scores, rated, mean_scores, counts)
        report.predictions = {card.id: LEVELS[label] for card, label in zip(unrated, labels)}
        report.scores = {card.id: float(score) for card, score in zip(unrated, scores)}
        report.cut_point_predictions = {card.id: LEVELS[label] for card, label in zip(unrated, cut)}
    report.decks = _deck_parts(report, rated, unrated, counts, deck_runs)
    report.seconds = time.perf_counter() - started
    return report


def _deck_parts(
    report: DifficultyReport, rated: Sequence[Card], unrated: Sequence[Card], counts, deck_runs
) -> dict[str, DifficultyReport]:
    """One report per deck with quizzes: its rated quizzes' share of the cross-validation, and its predictions."""
    parts = {}
    for deck in sorted(set(counts) | {card.deck for card in unrated}):
        ids = [card.id for card in unrated if card.deck == deck]
        part = DifficultyReport(
            report.feature_set, counts.get(deck, (0, 0, 0)), len(ids), report.folds, report.repeats, scope=deck,
            predictions={card_id: report.predictions[card_id] for card_id in ids if card_id in report.predictions},
            scores={card_id: report.scores[card_id] for card_id in ids if card_id in report.scores},
            cut_point_predictions={card_id: report.cut_point_predictions[card_id] for card_id in ids if card_id in report.cut_point_predictions},
            levels={card.id: card.difficulty for card in rated if card.deck == deck},
            unrated_ids=tuple(ids),
        )
        runs = deck_runs.get(deck)
        if runs and runs["D1"]:
            part.metrics = _spread(runs["D1"])
            part.baselines = {name: _spread(runs[name]) for name in ("always easy", "similarity")}
            part.confusion = np.mean(runs["confusion"], axis=0)
        elif deck not in counts:
            part.note = "No rated quizzes in this deck: nothing to measure; its unrated quizzes take the course's shares."
        else:
            part.note = report.note
        parts[deck] = part
    return parts
