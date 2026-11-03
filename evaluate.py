"""Offline evaluation of the helper's models (app.models) on the course table. It is developed with the package app/
in the flashcards workspace (modules/helper/evaluate.py) and exported next to it into the capstone's root, where only
the export changes it (tools/export/export.md). From the capstone's root:

    uv run evaluate.py [--part <part>] [--table data/course] [--results results]

and in the flashcards workspace, from the repository root:

    modules/helper/.venv/Scripts/python.exe modules/helper/evaluate.py [--part <part>]

It reads the course table (app.course_table: cards.csv, decks.csv and topics.csv, by default in data/ next to this
file or at the repository root) and nothing of the viewer; the random positions of B0 and the random layouts are the
models' own draw (app.models.evaluation.random_directions, uniform on the sphere).

What it measures (docs/draft/helper/models/, each model's file):
  * B0, the random floor: the same-deck precision@5 of the 5 nearest cards at random positions on the sphere, over
    several seeds;
  * S1, the vectoriser variants (each a factory of anything with fit_transform): same-deck precision@5 (empty slots
    count as misses), the precision over the kept neighbours, the coverage and the number of distinct recommended
    cards, without and with the variant's own minimum score, the 10th percentile of its fifth neighbour's cosine;
    S2, the sentence encoders and their hybrids with S1's chosen settings (app.models.embeddings), measured the same
    way where sentence-transformers is installed, their embeddings cached in `--cache`;
  * C1, the chosen model's map: trustworthiness and top-3 overlap of the deck positions; of the card positions per
    model topic the within-topic overlap at k = min(5, n - 1), at the fixed k = 2 and k = 3 (over the topics with
    k + 2 cards or more) with the lift over random positions, and the trustworthiness at the layout's k; each
    against the SVD start and random positions; and the time the whole model takes to build;
  * D1, the difficulty model (models/difficulty.md): out of fold on the rated quizzes, every feature set with the
    deck shares and with the most likely class, the two baselines, and the predicted levels of the unrated quizzes
    by the deck shares and by the cut points of the rated cards; and, on the quizzes labelled by hand, the
    predictions' accuracy, macro-F1 and weighted kappa, the score's Spearman against the hand levels and the share
    of a deck's pairs the score orders as the hand levels do, with every row's paired difference of that share to
    D1's chosen row and its bootstrap interval over the judged quizzes, next to the hand levels' agreement with the
    authored levels of the rated quizzes judged blind; the within-deck order of every row per deck as rows of
    their own (`scope` "deck <id>"; the course's rows "all"), paired against D1's chosen row over the deck's
    quizzes; D2, the zero-shot tags (app.models.zero_shot, where transformers and the NLI models are there, the
    tags cached in `--cache`): D2 alone per model and template, judged the same way without folds (model D2, the
    variant naming the template and the model's size, the model in `params`), built on D1's frame and supplying
    no feature to it (app roadmap step 15);
  * T1, the topic model (models/topic.md): the agreement with the scripted topics of the nearest centroid rule, of
    the neighbour rule (the nearest other card of the deck) and of a logistic regression per deck, leave-one-out,
    next to chance by the topic sizes; by stack and per deck; T1's margin table (the cards kept at a threshold and
    their agreement), the coherence per topic, the moved cards and the emptied topics.

Every run appends one row per model and variant to the results folder (`--results`, by default results/ next to
this file): results.csv (B0, S1, C1), difficulty.csv (D1, D2) and topics.csv (T1). A results file whose columns grew is
written again with its rows kept. `--part similarity`, `--part difficulty` or `--part topics` runs one part only.

The hand side is judged in the flashcards workspace, since its forms hold card text: the hand marks of the
similarity part in hand_marks.csv (card, candidate, relevant, judged, judge, note) with their judging form
hand_check.md (about 20 cards of decks 17 and 24 with the top-5 neighbours of every variant combined, each to be
marked relevant or not), and the hand labels of D1 in hand_labels.csv (card, level, judged, judge, note) with their
form hand_labels.md (every unrated quiz and, blind, the rated quizzes of two decks, each to be labelled easy, medium
or hard by the rule of models/difficulty.md, "The hand-labelled sample"). Each file is read before its form is
written again and written after it, so a changed form, a removed variant or a changed sample loses no judgement;
`--judge` names who marked. With the marks, every variant's row carries its hand-judged precision@1, @3 and @5 over
the judged cards and the paired difference of its precision@5 to the chosen variant, bootstrapped over the cards.
The marks and the labels travel to the capstone without their notes, the forms do not: where a results folder holds
a file without its form, the hand columns are computed from the file alone and nothing is written back; where it
holds neither, that side's columns stay empty, the form is not written and one line says so. The form lists a quiz's
options with their feedback from the course table (the record's options), the answer for a quiz without them.

Every row of a run carries the run's start time to the minute (`run`), next to its date, so the standalone page
reads the newest run of each file (app/standalone/results.py; what the columns mean: app/docs/columns.md).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

sys.dont_write_bytecode = True  # as in the launcher: no __pycache__ next to the module's sources

MODULE_DIR = Path(__file__).resolve().parent  # the folder of the package app/: modules/helper/ here, the capstone's root
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))  # the helper's package, app, as the launcher puts it on the path

from app import course_table  # noqa: E402
from app.models import embeddings, evaluation, layout, similar, zero_shot  # noqa: E402
from app.models.course_model import CARD_NEIGHBOURS, DECK_NEIGHBOURS, CourseModel  # noqa: E402
from app.models.difficulty_model import (  # noqa: E402
    DEFAULT_FEATURES, FEATURE_SETS, LEVEL_OF, LEVELS, DifficultyReport, fit_difficulty, judge_tags, measure, split,
)
from app.models.topic_model import FOCUS_MARGIN, TopicModel  # noqa: E402

RESULTS_DIR = MODULE_DIR / "results"
# The course table: the capstone's data/ next to the package, or the flashcards workspace's at the repository root
TABLE_DIRS = (MODULE_DIR / "data" / "course",) + tuple(
    parent / "data" / "course" for parent in MODULE_DIR.parents[1:2]
)
SEEDS = (42, 1, 2, 3, 4)
FIXED_K = (2, 3)  # the fixed small k of the card layouts' overlap
# The hand check's sample, deck -> every n-th card: every card of the two NLP decks and of the transfer deck 18
HAND_CHECK_DECKS = {"NLP Fundamentals": 1, "LLM Foundations & Embeddings": 1, "AI Agents & RAG Systems": 1}
TRANSFER_DECK = "AI Agents & RAG Systems"  # judged with the others, read once after the choice
# The tags of the marks' note column (similarity.md, "S2", "How it is judged"): three on an irrelevant candidate,
# two on a relevant one; the evaluation counts per variant how many tagged pairs it lists
TAGS = ("lexical trap", "shared dataset", "generic term", "paraphrase", "code name")
# The embeddings' cache: the launchers' folder, data/cache/ next to the package (the capstone) or temp/ (here)
CACHE_DIRS = (MODULE_DIR / "data" / "cache", MODULE_DIR / "temp")
COLUMNS = [
    "date", "run", "model", "variant", "params", "seed", "precision_at_5", "kept_precision", "coverage", "distinct", "min_score",
    "hand_precision", "hand_p1", "hand_p3", "hand_diff", "hand_diff_low", "hand_diff_high", "judged_cards",
    "trust_decks", "overlap_decks", "overlap_cards", "overlap_k2", "lift_k2", "overlap_k3", "lift_k3", "trust_cards",
    "overlap_17", "overlap_24", "seconds", "notes",
]
MARK_COLUMNS = ["card", "candidate", "relevant", "judged", "judge", "note"]
RELEVANT = "x"
DIFFICULTY_COLUMNS = [
    "date", "run", "model", "variant", "scope", "params", "features", "label_rule", "folds", "repeats", "seed",
    "macro_f1", "macro_f1_sd", "balanced_accuracy", "balanced_accuracy_sd", "kappa", "kappa_sd", "accuracy", "accuracy_sd",
    "spearman", "recall_easy", "recall_medium", "recall_hard", "rated", "unrated",
    "predicted_easy", "predicted_medium", "predicted_hard",
    "hand_judged", "hand_macro_f1", "hand_kappa", "hand_accuracy", "hand_spearman", "hand_within_deck", "hand_within_pairs",
    "hand_diff", "hand_diff_low", "hand_diff_high",
    "seconds", "notes",
]
LABEL_COLUMNS = ["card", "level", "judged", "judge", "note"]
D2_SKIPPED = "D2 skipped: the zero-shot rows are not measured"
COURSE_SCOPE = "all"  # the scope of a row over the course; a deck's rows read "deck <id>"
# The rated quizzes judged blind next to the unrated ones: the two rated decks with the most unrated quizzes
HAND_LABEL_REFERENCE_DECKS = ("Logistic Regression & Evaluation Metrics", "Decision Trees & Ensemble Methods")
LABEL_HEADING = re.compile(r"^### (\S+) \(")
LEVEL_MARK = re.compile(r"^- \[[xX]\] (easy|medium|hard)\b")
NOTE_LINE = re.compile(r"^- note:\s?(.*)$")
TOPICS_COLUMNS = [
    "date", "run", "model", "scope", "cards", "topics", "agreement", "kept_share", "coherence_median", "coherence_below_half",
    "coherence_at_one", "moved", "emptied", "seconds", "notes",
]
RUN_FORMAT = "%Y-%m-%dT%H:%M"  # the run column: the run's start time to the minute
MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15)  # the margin table of models/topic.md
CANDIDATE = re.compile(r"^- \[([xX ])\] (\S+) \(")
NONE_RELEVANT = re.compile(r"^- \[[xX]\] none\b")  # judged: no candidate of this card is relevant
HEADING = re.compile(r"^## (\S+) \(")
NOT_JUDGED = "not judged yet"  # the tag of a candidate without a row in the marks file under a judged card


@dataclass(frozen=True)
class ResultFiles:
    """The files of a results folder."""

    results: Path  # B0, S1, C1
    difficulty: Path  # D1
    topics: Path  # T1
    hand_check: Path  # the judging form of the similarity part
    hand_marks: Path  # its marks
    hand_labels_form: Path  # the judging form of D1
    hand_labels: Path  # its labels


def result_files(folder: Path = RESULTS_DIR) -> ResultFiles:
    folder = Path(folder)
    return ResultFiles(folder / "results.csv", folder / "difficulty.csv", folder / "topics.csv", folder / "hand_check.md",
                       folder / "hand_marks.csv", folder / "hand_labels.md", folder / "hand_labels.csv")


def default_table_dir() -> Path:
    """The first course table folder of TABLE_DIRS that exists (the first one when none does)."""
    return next((folder for folder in TABLE_DIRS if folder.is_dir()), TABLE_DIRS[0])


def default_cache_dir() -> Path:
    """The embeddings' cache folder: the first of CACHE_DIRS whose parent (data/, the module folder) exists."""
    return next((folder for folder in CACHE_DIRS if folder.parent.is_dir()), CACHE_DIRS[0])


# --- the variants -----------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """A vectoriser variant of the similarity part: its name, a factory of its vectoriser (anything with
    fit_transform: TF-IDF, a scikit-learn Pipeline, a sentence-embedding wrapper), whether the card text takes the
    explanation, and its settings for the params column."""

    name: str
    make: Callable[[], object]
    with_explain: bool
    params: str
    model: str = "S1"  # the model column: S1 for the TF-IDF variants, S2 for the encoders and the hybrids


def tfidf_variant(name: str, with_explain: bool = False, **options) -> Variant:
    """A TF-IDF variant: the baseline's settings without sublinear tf, changed by `options`."""
    settings = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=False)
    settings.update(options)
    return Variant(name, lambda: similar.tfidf_vectoriser(**settings), with_explain, f"{settings}, explain={with_explain}")


CHOSEN = "chosen: uni+bigrams, min_df=2, sublinear, explain"  # the reference every paired difference is against


def variants() -> list[Variant]:
    """The vectoriser variants of the roadmap (step 3) plus the chosen settings, last. Every variant stays: the
    chosen one is the reference every interval is paired against."""
    return [
        tfidf_variant("unigrams, min_df=2", ngram_range=(1, 1)),
        tfidf_variant("uni+bigrams, min_df=1", min_df=1),
        tfidf_variant("uni+bigrams, min_df=2"),
        tfidf_variant("uni+bigrams, min_df=3", min_df=3),
        tfidf_variant("uni+bigrams, min_df=2, explain", with_explain=True),
        tfidf_variant("uni+bigrams, min_df=2, sublinear", sublinear_tf=True),
        tfidf_variant(CHOSEN, with_explain=similar.WITH_EXPLAIN, ngram_range=similar.NGRAM_RANGE, min_df=similar.MIN_DF,
                      sublinear_tf=similar.SUBLINEAR_TF),
    ]


S2_SKIPPED = "S2 skipped: sentence-transformers is not installed (the embeddings group), so the encoder variants are not measured"


def s2_variants(cache_dir: Optional[Path] = None, loader: Optional[Callable] = None) -> list[Variant]:
    """S2's variants (similarity.md, "S2"): each encoder alone and the hybrid of S1's chosen settings with it, every
    one on S1's document (with the explanation); the embeddings cached in `cache_dir`. Empty without the library
    (`loader` stands in for the models in the tests)."""
    if loader is None and not embeddings.encoder_available():
        return []
    cache = embeddings.EmbeddingCache(Path(cache_dir) / embeddings.CACHE_FOLDER) if cache_dir is not None else None

    def encoder(model_name: str):
        return lambda: embeddings.SentenceEncoder(model_name, cache=cache, loader=loader)

    def hybrid(model_name: str):
        return lambda: embeddings.HybridVectoriser(similar.tfidf_vectoriser(), embeddings.SentenceEncoder(model_name, cache=cache, loader=loader))

    chosen = f"{{'ngram_range': {similar.NGRAM_RANGE}, 'min_df': {similar.MIN_DF}, 'sublinear_tf': {similar.SUBLINEAR_TF}}}"
    rows = []
    for short, model_name in (("MiniLM", embeddings.MINILM), ("mpnet", embeddings.MPNET)):
        rows.append(Variant(short, encoder(model_name), True, f"model={model_name}, explain=True", "S2"))
        rows.append(Variant(f"hybrid: chosen + {short}", hybrid(model_name), True,
                            f"tfidf {chosen} + model={model_name}, each half at its own minimum, explain=True", "S2"))
    return rows


# --- the results files ------------------------------------------------------------------


def write_rows(rows: list[dict], path: Path, columns: list[str] = COLUMNS) -> None:
    """Append the rows to the results file. A file whose header is not these columns (the columns grew) is written
    again with the new header, its rows kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stored: list[dict] = []
    rewrite = not path.exists()
    if not rewrite:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != list(columns):
                stored, rewrite = list(reader), True
    with path.open("w" if rewrite else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if rewrite:
            writer.writeheader()
        for row in stored + rows:
            writer.writerow({column: row.get(column) or "" for column in columns})


def fmt(value) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def scope_note(deck_ids: dict[str, str]) -> str:
    """The decks of the course, for the notes column: "decks 01 to 24"."""
    ids = sorted(deck_ids.values())
    return f"decks {ids[0]} to {ids[-1]}" if ids else "no decks"


def stamped(rows: list[dict], run_key: str) -> list[dict]:
    """The rows with the run column set to the run's key (the start time, RUN_FORMAT)."""
    for row in rows:
        row["run"] = run_key
    return rows


# --- D1 -------------------------------------------------------------------------------


def difficulty_rows(report, today: str, model: str, variant: str, measures: dict, label_rule: str, note: str, predicted=None,
                    params: str = "", scope: str = COURSE_SCOPE) -> dict:
    """One row of difficulty.csv: the measures (name -> (mean, sd)) and, if given, the predicted levels; `params`
    names D2's model, `scope` the course or a deck."""
    row = {"date": today, "model": model, "variant": variant, "scope": scope, "params": params, "features": report.feature_set,
           "label_rule": label_rule, "folds": report.folds, "repeats": report.repeats, "seed": 42 if report.repeats else "",
           "rated": report.rated, "unrated": report.unrated, "seconds": f"{report.seconds:.2f}", "notes": note}
    for name, (mean, sd) in measures.items():
        row[name] = fmt(mean)
        if f"{name}_sd" in DIFFICULTY_COLUMNS:
            row[f"{name}_sd"] = fmt(sd)
    if predicted is not None:
        row["predicted_easy"], row["predicted_medium"], row["predicted_hard"] = report.counts(predicted)
    return row


def hand_measures(labels: dict[str, dict], predictions: dict[str, str], scores: dict[str, float] | None = None) -> tuple[dict, int]:
    """The METRICS of the predictions against the hand labels (the truth), over the judged cards among the predicted
    ones, and how many there are; ({}, 0) without one. `scores` gives the Spearman of the score with the hand level."""
    ids = [card_id for card_id in predictions if labels.get(card_id, {}).get("level") in LEVEL_OF]
    if not ids:
        return {}, 0
    truth = np.array([LEVEL_OF[labels[card_id]["level"]] for card_id in ids])
    predicted = np.array([LEVEL_OF[predictions[card_id]] for card_id in ids])
    values = np.array([scores[card_id] for card_id in ids]) if scores else None
    return measure(truth, predicted, values), len(ids)


def within_deck_by_deck(labels: dict[str, dict], scores: dict[str, float], deck_of: dict[str, str]) -> dict[str, tuple[float, int]]:
    """Per deck with a pair: of the pairs of its judged quizzes with different hand levels, the share whose scores
    are in the hand levels' order (a tie counts half; chance 0.5), and the number of pairs."""
    by_deck: dict[str, list] = {}
    for card_id, score in scores.items():
        level = labels.get(card_id, {}).get("level")
        if level in LEVEL_OF:
            by_deck.setdefault(deck_of[card_id], []).append((LEVEL_OF[level], score))
    result = {}
    for deck, members in by_deck.items():
        right = pairs = 0.0
        for (level_a, score_a), (level_b, score_b) in combinations(members, 2):
            if level_a == level_b:
                continue
            pairs += 1
            right += 0.5 if score_a == score_b else float((score_a > score_b) == (level_a > level_b))
        if pairs:
            result[deck] = (right / pairs, int(pairs))
    return result


def within_deck_concordance(labels: dict[str, dict], scores: dict[str, float], deck_of: dict[str, str]) -> tuple[float, int]:
    """The order a deck's extra quizzes get, over every deck: the share of the pairs in the hand order and their
    number (within_deck_by_deck); (NaN, 0) without a pair."""
    per_deck = within_deck_by_deck(labels, scores, deck_of)
    pairs = sum(count for _, count in per_deck.values())
    right = sum(share * count for share, count in per_deck.values())
    return (right / pairs if pairs else float("nan")), int(pairs)


def paired_order_bootstrap(labels: dict[str, dict], reference: Mapping[str, float], scores: Mapping[str, float], deck_of: Mapping[str, str],
                           draws: int = evaluation.BOOTSTRAP_DRAWS, seed: int = 42, level: float = evaluation.INTERVAL) -> tuple[float, float, float]:
    """The within-deck order of `scores` minus the reference's, over the judged quizzes both score, and its
    percentile interval over `draws` resamples of the quizzes with replacement, the pairs counted anew in each
    draw (the paired bootstrap of the order measure): (difference, low, high); NaN without a pair."""
    ids = [card_id for card_id in scores if card_id in reference and labels.get(card_id, {}).get("level") in LEVEL_OF]
    if not ids:
        return math.nan, math.nan, math.nan
    levels = np.array([LEVEL_OF[labels[card_id]["level"]] for card_id in ids])
    decks = np.array([deck_of[card_id] for card_id in ids])
    mine = np.array([scores[card_id] for card_id in ids], dtype=float)
    theirs = np.array([reference[card_id] for card_id in ids], dtype=float)
    first, second = np.triu_indices(len(ids), 1)

    def difference(chosen: np.ndarray) -> float:
        level_a, level_b = levels[chosen][first], levels[chosen][second]
        counted = (decks[chosen][first] == decks[chosen][second]) & (level_a != level_b)
        if not counted.any():
            return math.nan
        harder = (level_a > level_b)[counted]

        def share(values: np.ndarray) -> float:
            value_a, value_b = values[chosen][first][counted], values[chosen][second][counted]
            right = np.where(value_a == value_b, 0.5, ((value_a > value_b) == harder).astype(float))
            return float(right.mean())

        return share(mine) - share(theirs)

    whole = difference(np.arange(len(ids)))
    if math.isnan(whole):
        return math.nan, math.nan, math.nan
    generator = np.random.default_rng(seed)
    resampled = np.array([difference(generator.integers(0, len(ids), size=len(ids))) for _ in range(draws)])
    resampled = resampled[~np.isnan(resampled)]
    tail = 100 * (1.0 - level) / 2
    return whole, float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100 - tail))


def with_hand_columns(row: dict, measures: dict, judged: int, within: tuple[float, int] = (float("nan"), 0),
                      diff: Optional[tuple[float, float, float]] = None) -> dict:
    """The row with the hand columns, when any of its cards is judged; the within-deck order when there is a pair,
    and its paired difference to D1's chosen row with the interval (`diff`) where given."""
    if judged:
        row.update(hand_judged=judged, hand_macro_f1=fmt(measures["macro_f1"]), hand_kappa=fmt(measures["kappa"]),
                   hand_accuracy=fmt(measures["accuracy"]), hand_spearman=fmt(measures["spearman"]))
    if within[1]:
        row.update(hand_within_deck=fmt(within[0]), hand_within_pairs=within[1])
        if diff is not None and not math.isnan(diff[0]):
            row.update(hand_diff=fmt(diff[0]), hand_diff_low=fmt(diff[1]), hand_diff_high=fmt(diff[2]))
    return row


def tags_variant(template: zero_shot.Template, model_name: str) -> str:
    """The name of D2's rows of a template and a model: "criterion, base", "level names, large"."""
    return f"{template.name}, {zero_shot.size_of(model_name)}"


def difficulty_tags(cards, cache_dir: Optional[Path] = None, loader: Optional[Callable] = None) -> dict[str, zero_shot.Tags]:
    """D2's tags by variant (tags_variant: every template of zero_shot.TEMPLATES on every model of NLI_MODELS, the
    base model's first), from the cache in `cache_dir` or the models; empty, with one line, without the library,
    and a model that does not load loses its rows with one line (`loader` stands in for the pipeline in the
    tests)."""
    found: dict[str, zero_shot.Tags] = {}
    if loader is None and not zero_shot.tagger_available():
        print(f"{D2_SKIPPED}: D2's models need {zero_shot.LIBRARY} (the embeddings group)")
        return found
    for model_name in zero_shot.NLI_MODELS:
        for template in zero_shot.TEMPLATES:
            reasons: list[str] = []
            tags = zero_shot.tags_of(cards, cache_dir, template, model_name, loader=loader, output=reasons.append)
            if tags is None:
                print(f"{D2_SKIPPED} for the {zero_shot.size_of(model_name)} model: {reasons[0] if reasons else ''}".rstrip(": "))
                break
            variant = tags_variant(template, model_name)
            found[variant] = tags
            print(f"D2 {variant}: {len(tags.scores)} quizzes tagged by {tags.model_name} "
                  + ("from the cache" if tags.cached else f"on {tags.device} in {tags.seconds:.1f}s"))
    return found


def level_counts(labels: dict[str, dict], ids) -> tuple[int, int, int]:
    """The hand levels among the ids: (easy, medium, hard)."""
    found = [labels[card_id]["level"] for card_id in ids if card_id in labels]
    return tuple(found.count(level) for level in LEVELS)


def evaluate_difficulty(cards, today: str, note: str, labels: dict[str, dict] | None = None, vectoriser=None,
                        tags: Optional[Mapping[str, zero_shot.Tags]] = None, deck_ids: Optional[Mapping[str, str]] = None) -> list[dict]:
    """D1: every feature set, out of fold, with the deck shares and with the most likely class; the baselines once;
    the predicted levels of the unrated quizzes by the deck shares and by the cut points. With hand labels, the
    predictions of both rules and the always-easy baseline are scored on the judged quizzes (the hand columns),
    every row's within-deck order is paired against D1's chosen row (the hand_diff columns, bootstrapped over the
    judged quizzes) and printed per deck, and a `hand` row gives the hand levels' agreement with the authored levels
    of the rated quizzes judged blind; every deck-shares row with judged quizzes gets a row per deck with a pair
    (`scope` "deck <id>", by `deck_ids`; the course's rows "all"): the deck's within-deck order and its paired
    difference to D1's chosen row over the deck's quizzes, a readout and no verdict. With D2's tags (`tags`, by
    variant; difficulty_tags), D2 alone gets its rows per variant (model D2, the model in `params`), judged the
    same way without folds; without them they are skipped with one line. D2 supplies no feature to D1."""
    labels = labels or {}
    tags = tags or {}
    deck_ids = deck_ids or {}
    deck_of = {card.id: card.deck for card in cards}
    index = similar.SimilarityIndex.fit(cards, vectoriser, min_score=None)  # the app's vectoriser, as the launchers fit it
    reports: list[tuple[str, str, str, DifficultyReport]] = []  # (model, variant, params, report)
    for feature_set in FEATURE_SETS:
        report = fit_difficulty(cards, index, feature_set)
        if not report.measured:
            print(f"D1 {feature_set}: {report.note}")
            continue
        reports.append(("D1", f"{feature_set}{' (chosen)' if feature_set == DEFAULT_FEATURES else ''}", "", report))
    for name, found in tags.items():
        report = judge_tags(cards, found.scores)
        if not report.measured:
            print(f"D2 {name}: {report.note}")
            continue
        reports.append(("D2", name, f"model={found.model_name}" + (f", device={found.device}" if found.device else ""), report))
    chosen = next((report for model, _variant, _params, report in reports if model == "D1" and report.feature_set == DEFAULT_FEATURES), None)
    reference = chosen.scores if chosen is not None else {}
    rows = []
    for model, variant, params, report in reports:
        shares_hand, judged = hand_measures(labels, report.predictions, report.scores)
        cut_hand, _ = hand_measures(labels, report.cut_point_predictions, report.scores)
        within = within_deck_concordance(labels, report.scores, deck_of)
        diff = paired_order_bootstrap(labels, reference, report.scores, deck_of) if judged else None
        rows.append(with_hand_columns(
            difficulty_rows(report, today, model, variant, report.metrics, "deck shares", note, report.predictions, params),
            shares_hand, judged, within, diff))
        rows.append(difficulty_rows(report, today, model, variant, report.most_likely, "most likely class", note, params=params))
        if model == "D1":
            rows.append(with_hand_columns(
                difficulty_rows(report, today, model, variant, {}, "cut points of the rated cards", note, report.cut_point_predictions),
                cut_hand, judged, within, diff))
        if judged:  # the deck rows: the order within each deck with a pair, paired over the deck's quizzes alone
            for deck, pair_share in sorted(within_deck_by_deck(labels, report.scores, deck_of).items()):
                deck_labels = {card_id: labels[card_id] for card_id in report.scores if deck_of[card_id] == deck and card_id in labels}
                part = report.decks.get(deck)
                row = difficulty_rows(report, today, model, variant, {}, "deck shares", note, params=params,
                                      scope=f"deck {deck_ids.get(deck, deck)}")
                row.update(rated=part.rated if part else 0, unrated=part.unrated if part else 0,
                           hand_judged=sum(entry.get("level") in LEVEL_OF for entry in deck_labels.values()))
                rows.append(with_hand_columns(row, {}, 0, pair_share, paired_order_bootstrap(deck_labels, reference, report.scores, deck_of)))
        f1, kappa = report.metrics["macro_f1"], report.metrics["kappa"]
        print(f"{model} {variant:20s} deck shares: macro-F1 {f1[0]:.3f} ± {f1[1]:.3f}, weighted kappa {kappa[0]:.3f} ± {kappa[1]:.3f}, "
              f"accuracy {report.metrics['accuracy'][0]:.3f}, recall {report.metrics['recall_easy'][0]:.2f} / "
              f"{report.metrics['recall_medium'][0]:.2f} / {report.metrics['recall_hard'][0]:.2f} | most likely class: macro-F1 "
              f"{report.most_likely['macro_f1'][0]:.3f} | {report.unrated} unrated quizzes: {report.predicted_counts} "
              f"(cut points {report.counts(report.cut_point_predictions)}) ({report.seconds:.2f}s)")
        if model == "D1" and report.feature_set == DEFAULT_FEATURES:
            print("   confusion (rows true, columns predicted, shares): "
                  + " | ".join(" ".join(f"{value:.2f}" for value in row) for row in report.confusion))
            parts = [part for part in report.decks.values() if part.measured]
            kappas = [part.metrics["kappa"][0] for part in parts]
            print(f"   per deck ({len(parts)} measured): kappa mean {np.mean(kappas):.2f}, lowest {min(kappas):.2f}, highest {max(kappas):.2f}; "
                  f"accuracy mean {np.mean([part.metrics['accuracy'][0] for part in parts]):.2f}, "
                  f"macro-F1 mean {np.mean([part.metrics['macro_f1'][0] for part in parts]):.2f}")
        if judged:
            print(f"   on the {judged} hand-labelled quizzes: deck shares macro-F1 {shares_hand['macro_f1']:.3f}, weighted kappa "
                  f"{shares_hand['kappa']:.3f}, accuracy {shares_hand['accuracy']:.3f}, Spearman of the score {shares_hand['spearman']:.3f} | "
                  f"cut points macro-F1 {cut_hand['macro_f1']:.3f}, weighted kappa {cut_hand['kappa']:.3f}, accuracy {cut_hand['accuracy']:.3f} | "
                  f"within a deck {within[0]:.3f} of {within[1]} pairs in the hand order"
                  + (f" | paired difference to {DEFAULT_FEATURES} {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]"
                     if diff is not None and not math.isnan(diff[0]) else ""))
            per_deck = within_deck_by_deck(labels, report.scores, deck_of)
            print("   within a deck, per deck: " + ", ".join(f"{deck} {share:.2f} ({pairs})" for deck, (share, pairs) in sorted(per_deck.items())))
    if chosen is not None:
        for name, measures in chosen.baselines.items():
            row = difficulty_rows(chosen, today, "baseline", name, measures, "deck shares" if name == "similarity" else "constant", note)
            if name == "always easy":
                row = with_hand_columns(row, *hand_measures(labels, {card_id: "easy" for card_id in chosen.predictions}))
            rows.append(row)
            print(f"baseline {name:12s} macro-F1 {measures['macro_f1'][0]:.3f} ± {measures['macro_f1'][1]:.3f}, "
                  f"weighted kappa {measures['kappa'][0]:.3f} ± {measures['kappa'][1]:.3f}, accuracy {measures['accuracy'][0]:.3f}"
                  + (f" | on the hand-labelled quizzes: accuracy {row['hand_accuracy']}" if row.get("hand_judged") else ""))
        # The hand levels against the authored levels of the rated quizzes judged blind: how far the criterion
        # agrees with the quota the levels were written under (the truth is the authored level)
        authored = {card_id: level for card_id, level in chosen.levels.items() if labels.get(card_id, {}).get("level") in LEVEL_OF}
        if authored:
            hand = {card_id: labels[card_id]["level"] for card_id in authored}
            values, judged_rated = hand_measures({card_id: {"level": level} for card_id, level in authored.items()}, hand)
            unrated_counts = level_counts(labels, report.predictions)
            rated_counts = level_counts(labels, authored)
            rows.append({"date": today, "model": "hand", "variant": "the criterion against the authored levels", "scope": COURSE_SCOPE, "label_rule": "authored",
                         "rated": judged_rated, "unrated": sum(unrated_counts), "hand_judged": judged_rated,
                         "hand_macro_f1": fmt(values["macro_f1"]), "hand_kappa": fmt(values["kappa"]), "hand_accuracy": fmt(values["accuracy"]),
                         "recall_easy": fmt(values["recall_easy"]), "recall_medium": fmt(values["recall_medium"]), "recall_hard": fmt(values["recall_hard"]),
                         "notes": f"hand levels easy/medium/hard: {'/'.join(map(str, rated_counts))} of the rated quizzes judged blind "
                                  f"(authored {'/'.join(map(str, authored_counts(authored)))}), {'/'.join(map(str, unrated_counts))} of the unrated; {note}"})
            print(f"hand labels: the criterion against the authored levels of {judged_rated} rated quizzes judged blind: accuracy "
                  f"{values['accuracy']:.3f}, weighted kappa {values['kappa']:.3f}, macro-F1 {values['macro_f1']:.3f}; the share of each authored "
                  f"level called the same {values['recall_easy']:.2f} / {values['recall_medium']:.2f} / {values['recall_hard']:.2f}; "
                  f"hand levels {rated_counts} there, {unrated_counts} on the unrated quizzes")
    return rows


def authored_counts(levels: dict[str, str]) -> tuple[int, int, int]:
    """(easy, medium, hard) among the authored levels."""
    return tuple(list(levels.values()).count(level) for level in LEVELS)


# --- T1 -------------------------------------------------------------------------------


def topic_row(today: str, model: str, scope: str, cards: int, agreement: float, note: str, **extra) -> dict:
    row = {"date": today, "model": model, "scope": scope, "cards": cards, "agreement": fmt(agreement), "notes": note}
    row.update(extra)
    return row


def evaluate_topics(cards, seeds, deck_ids: dict[str, str], today: str, note: str, vectoriser=None) -> list[dict]:
    """T1 against the scripted topics, with the neighbour rule, the per-deck logistic regression and chance: by
    course, by stack and per deck; T1's margin table, coherence, moved cards and emptied topics (models/topic.md)."""
    started = time.perf_counter()
    index = similar.SimilarityIndex.fit(cards, vectoriser, min_score=None)  # the app's vectoriser, as the launchers fit it
    model = TopicModel(index, seeds)
    t1_seconds = time.perf_counter() - started
    scripted = model.scripted_of
    by_id = {card.id: card for card in cards}
    deck_order = sorted(dict.fromkeys(seed.deck for seed in seeds), key=lambda name: deck_ids.get(name, name))

    started = time.perf_counter()
    nearest: dict[str, str] = {}
    logistic: dict[str, str] = {}
    chance: dict[str, float] = {}
    for deck in deck_order:
        rows = index.rows_of_deck(deck)
        ids = [index.ids[row] for row in rows]
        labels = [scripted[card_id] for card_id in ids]
        vectors = index.vectors[rows]
        nearest.update(zip(ids, evaluation.nearest_labels(vectors, labels)))
        logistic.update(zip(ids, evaluation.leave_one_out_logistic(vectors, labels)))
        chance[deck] = evaluation.chance_by_sizes(labels)
    judges_seconds = time.perf_counter() - started

    def agree(predicted: dict, ids) -> float:
        ids = list(ids)
        return evaluation.agreement([predicted[card_id] for card_id in ids], [scripted[card_id] for card_id in ids])

    def chance_of(ids) -> float:
        ids = list(ids)
        return sum(chance[by_id[card_id].deck] for card_id in ids) / len(ids) if ids else math.nan

    all_ids = list(model.topic_of)
    coherences = [value for value in model.coherence.values() if not math.isnan(value)]
    rows = [
        topic_row(today, "T1", "all", len(all_ids), model.agreement, note, topics=len(seeds), kept_share=fmt(1.0),
                  coherence_median=fmt(float(np.median(coherences))), coherence_below_half=sum(value < 0.5 for value in coherences),
                  coherence_at_one=sum(value >= 1.0 for value in coherences), moved=model.moved, emptied=len(model.emptied),
                  seconds=f"{t1_seconds:.2f}"),
        topic_row(today, "1-NN", "all", len(all_ids), agree(nearest, all_ids), note, topics=len(seeds)),
        topic_row(today, "LR", "all", len(all_ids), agree(logistic, all_ids), note, topics=len(seeds), seconds=f"{judges_seconds:.2f}"),
        topic_row(today, "chance", "all", len(all_ids), chance_of(all_ids), note, topics=len(seeds)),
    ]
    print(f"T1 nearest centroid: agreement {model.agreement:.3f} over {len(all_ids)} cards in {len(seeds)} topics | "
          f"1-NN {agree(nearest, all_ids):.3f} | LR per deck {agree(logistic, all_ids):.3f} | chance {chance_of(all_ids):.3f} "
          f"(T1 {t1_seconds:.2f}s, the judges {judges_seconds:.2f}s)")
    print(f"   {model.moved} cards elsewhere ({model.moved / len(all_ids):.0%}), {len(model.emptied)} topics emptied; coherence median "
          f"{np.median(coherences):.2f}, {sum(value < 0.5 for value in coherences)} topics below 0.5, {sum(value >= 1.0 for value in coherences)} at 1.0; "
          f"FOCUS_MARGIN {FOCUS_MARGIN} hides {len(model.hidden)} cards")
    for stack in sorted({card.stack for card in cards}):
        ids = [card_id for card_id in all_ids if by_id[card_id].stack == stack]
        rows.append(topic_row(today, "T1", f"stack {stack}", len(ids), model.agreement_of(ids), note))
        print(f"   stack {stack}: T1 {model.agreement_of(ids):.3f} over {len(ids)} cards")
    print("   | Deck | Cards | Topics | T1 | 1-NN | LR | Chance |")
    print("   |---|---|---|---|---|---|---|")
    for deck in deck_order:
        ids = [card_id for card_id in all_ids if by_id[card_id].deck == deck]
        topics = len([seed for seed in seeds if seed.deck == deck])
        scope = f"deck {deck_ids.get(deck, deck)}"  # by id: the rows travel and the standalone page shows them
        values = (model.agreement_of(ids), agree(nearest, ids), agree(logistic, ids), chance_of(ids))
        for name, value in zip(("T1", "1-NN", "LR", "chance"), values):
            rows.append(topic_row(today, name, scope, len(ids), value, note, topics=topics))
        print(f"   | {deck_ids.get(deck, '')} {deck} | {len(ids)} | {topics} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    print("   | Margin at least | Share of the course | Agreement among them |")
    for threshold in MARGINS:
        share, agreement = model.kept(threshold)
        rows.append(topic_row(today, "T1", f"margin >= {threshold:.2f}", round(share * len(all_ids)), agreement, note, kept_share=fmt(share)))
        print(f"   | {threshold:.2f} | {share:.0%} | {agreement:.3f} |")
    return rows


# --- the hand check and its marks -------------------------------------------------------


def hand_check_cards(cards, decks: dict[str, int] = HAND_CHECK_DECKS):
    """The sample of the hand check: every n-th card of each sampled deck, in deck order (about 20 cards of the two
    NLP decks by default)."""
    chosen = []
    for deck, step in decks.items():
        members = [card for card in cards if card.deck == deck]
        chosen.extend(members[::step])
    return chosen


def short(text: str, length: int = 90) -> str:
    text = " ".join(text.replace("`", "").replace("**", "").split())
    return text if len(text) <= length else text[: length - 1] + "…"


def read_marks(path: Path) -> dict[str, dict[str, dict]]:
    """The marks file: card id -> candidate id -> its row (card, candidate, relevant, judged, judge, note). A card
    with any row is judged; its relevant candidates are the rows marked `x`."""
    if not path.exists():
        return {}
    marks: dict[str, dict[str, dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("card") and row.get("candidate"):
                marks.setdefault(row["card"], {})[row["candidate"]] = {column: row.get(column) or "" for column in MARK_COLUMNS}
    return marks


def write_marks(marks: dict[str, dict[str, dict]], path: Path) -> None:
    """The marks file, one row per judged candidate, in card and candidate id order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARK_COLUMNS)
        writer.writeheader()
        for card_id in sorted(marks):
            for candidate in sorted(marks[card_id]):
                writer.writerow({column: marks[card_id][candidate].get(column, "") for column in MARK_COLUMNS})


def relevant_of(marks: dict[str, dict[str, dict]]) -> dict[str, set[str]]:
    """The judged cards' relevant candidates: card id -> the candidate ids marked relevant (empty for a judged card
    with none)."""
    return {card_id: {candidate for candidate, row in rows.items() if row.get("relevant")} for card_id, rows in marks.items()}


def read_hand_check(path: Path) -> dict[str, dict]:
    """The judging form: card id -> {"candidates": its candidate ids in the form's order, "relevant": those marked
    `[x]`}, for the judged cards only: cards with at least one mark, or with the `- [x] none` line marked (no
    candidate is relevant); a card without any mark counts as not judged yet."""
    if not path.exists():
        return {}
    listed: dict[str, dict] = {}
    judged: set[str] = set()
    card_id = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING.match(line)
        if heading:
            card_id = heading.group(1)
            listed[card_id] = {"candidates": [], "relevant": set()}
            continue
        if card_id is None:
            continue
        if NONE_RELEVANT.match(line):
            judged.add(card_id)
            continue
        candidate = CANDIDATE.match(line)
        if candidate:
            listed[card_id]["candidates"].append(candidate.group(2))
            if candidate.group(1) != " ":
                listed[card_id]["relevant"].add(candidate.group(2))
                judged.add(card_id)
    return {card_id: entry for card_id, entry in listed.items() if card_id in judged}


def merge_marks(stored: dict[str, dict[str, dict]], form: dict[str, dict], today: str, judge: str) -> dict[str, dict[str, dict]]:
    """The form's judgements over the stored marks: under a judged card, every listed candidate gets a row, relevant
    (`x`) or not; a candidate whose stored row says the same keeps it, any other gets a row of today by the judge
    (a stored note stays); a stored row the form does not list (a removed variant's candidate, a card out of the
    sample) stays; a card the form does not judge keeps its rows."""
    merged = {card_id: {candidate: dict(row) for candidate, row in rows.items()} for card_id, rows in stored.items()}
    for card_id, entry in form.items():
        rows = merged.setdefault(card_id, {})
        for candidate in entry["candidates"]:
            relevant = RELEVANT if candidate in entry["relevant"] else ""
            row = rows.get(candidate)
            if row and row["relevant"] == relevant:
                continue
            rows[candidate] = {"card": card_id, "candidate": candidate, "relevant": relevant, "judged": today, "judge": judge,
                               "note": row["note"] if row else ""}
    return merged


def hand_precision(neighbours: dict[str, list], marks: dict[str, set[str]], k: int = similar.TOP_K) -> float:
    """Hand-judged precision@k over the judged cards: the mean share of a variant's top-k neighbours that are
    marked relevant; NaN without judged cards."""
    shares = evaluation.judged_precision(neighbours, marks, k)
    return float(np.mean(list(shares.values()))) if shares else float("nan")


def write_hand_check(cards, by_id, indexes: dict[str, object], seen_today: str, marks: dict[str, dict[str, dict]], note: str,
                     path: Path, decks: Optional[dict[str, int]] = None) -> int:
    """The combined candidates of all variants for the hand check, as a markdown checklist; the stored marks are
    ticked, and a candidate without a row under a judged card (a new variant's) is tagged `not judged yet`.
    `decks` is the sample (HAND_CHECK_DECKS by default)."""
    decks = HAND_CHECK_DECKS if decks is None else decks
    relevant = relevant_of(marks)
    lines = [
        f"# Hand check: related cards of {', '.join(decks)}",
        "",
        f"Written by evaluate.py on {seen_today} ({note}). For every card below, mark every candidate that is a "
        "genuinely related card for a learner (the same concept, a prerequisite, or the natural next card): put `x` "
        "in its `[ ]`, as many or as few as apply; leave the others empty. If no candidate fits, mark the `none` "
        "line. A card without any mark counts as not judged yet; under a judged card, an unmarked candidate counts "
        f"as not relevant, and one tagged `{NOT_JUDGED}` is new since the card was judged. The columns say which "
        "variants listed the candidate, with the cosine. Later runs keep every judgement in hand_marks.csv and "
        "compute every variant's hand-judged precision from them.",
        "",
    ]
    names = list(indexes)
    sample = hand_check_cards(cards, decks)
    for card in sample:
        lines.append(f"## {card.id} ({card.deck}): {short(card.question)}")
        lines.append("")
        candidates: dict[str, dict[str, float]] = {}
        for name, index in indexes.items():
            for other, score in index.related(card.id):
                candidates.setdefault(other, {})[name] = score
        ordered = sorted(candidates, key=lambda other: (-len(candidates[other]), -max(candidates[other].values())))
        judged = card.id in marks
        for other in ordered:
            columns = ", ".join(f"{name}: {candidates[other][name]:.2f}" for name in names if name in candidates[other])
            box = "[x]" if other in relevant.get(card.id, set()) else "[ ]"
            tag = f"  —  {NOT_JUDGED}" if judged and other not in marks[card.id] else ""
            lines.append(f"- {box} {other} ({by_id[other].deck}): {short(by_id[other].question)}{tag}  —  {columns}")
        judged_none = judged and not relevant[card.id]
        lines.append(f"- {'[x]' if judged_none else '[ ]'} none of these is relevant")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(sample)


# --- the hand labels of D1 ------------------------------------------------------------


def read_labels(path: Path) -> dict[str, dict]:
    """The labels file: card id -> its row (card, level, judged, judge, note). A row whose level is none of the
    three (a note by hand, say) is kept but never scored."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["card"]: {column: row.get(column) or "" for column in LABEL_COLUMNS}
                for row in csv.DictReader(handle) if row.get("card")}


def write_labels(labels: dict[str, dict], path: Path) -> None:
    """The labels file, one row per judged card in id order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for card_id in sorted(labels):
            writer.writerow({column: labels[card_id].get(column, "") for column in LABEL_COLUMNS})


def read_label_form(path: Path) -> dict[str, dict]:
    """The marks of the form: card id -> {"level", "note"} for every card with exactly one level marked. A card
    without a mark is not judged yet; a card with two levels marked is reported and skipped."""
    if not path.exists():
        return {}
    found: dict[str, dict] = {}
    card_id = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = LABEL_HEADING.match(line)
        if heading:
            card_id = heading.group(1)
            found[card_id] = {"levels": [], "note": ""}
            continue
        if card_id is None:
            continue
        marked = LEVEL_MARK.match(line)
        if marked:
            found[card_id]["levels"].append(marked.group(1))
            continue
        note = NOTE_LINE.match(line)
        if note:
            found[card_id]["note"] = note.group(1).strip()
    marks = {}
    for card_id, entry in found.items():
        if len(entry["levels"]) == 1:
            marks[card_id] = {"level": entry["levels"][0], "note": entry["note"]}
        elif entry["levels"]:
            print(f"hand labels: {card_id} has {len(entry['levels'])} levels marked in {path.name}, skipped")
    return marks


def merge_labels(stored: dict[str, dict], marks: dict[str, dict], today: str, judge: str) -> dict[str, dict]:
    """The form's marks over the stored labels: a card marked with another level than stored, or not stored, gets
    a row of today by the judge; a card marked with its stored level keeps its row (and takes the form's note); a
    stored card the form does not mark keeps its row."""
    merged = {card_id: dict(row) for card_id, row in stored.items()}
    for card_id, mark in marks.items():
        row = merged.get(card_id)
        if row and row["level"] == mark["level"]:
            row["note"] = mark["note"] or row.get("note", "")
            continue
        merged[card_id] = {"card": card_id, "level": mark["level"], "judged": today, "judge": judge, "note": mark["note"]}
    return merged


def label_sample(cards, reference_decks=HAND_LABEL_REFERENCE_DECKS) -> tuple[list, list]:
    """(the unrated quizzes, the rated quizzes of the reference decks), each in course order."""
    rated, unrated = split(cards)
    return unrated, [card for card in rated if card.deck in reference_decks]


def one_line(text: str) -> str:
    return " ".join((text or "").split())


def quoted(text: str) -> list[str]:
    """The text as a markdown block quote, line by line."""
    return [f"> {line}".rstrip() for line in (text or "").strip().splitlines()] or [">"]


def write_hand_labels_form(cards, labels: dict[str, dict], today: str, note: str, path: Path,
                           reference_decks=HAND_LABEL_REFERENCE_DECKS) -> tuple[int, int]:
    """The judging form of the hand labels: every unrated quiz and, blind, the rated quizzes of the reference decks,
    each with its question in full, its options with their feedback (the record's; the answer for a quiz without
    options), the three level boxes and a note line; the stored labels are marked. The authored levels are never
    written. Returns the two counts."""
    unrated, reference = label_sample(cards, reference_decks)
    decks = ", ".join(sorted({card.deck for card in reference}))
    lines = [
        "# Hand labels: the difficulty of the quizzes",
        "",
        f"Written by evaluate.py on {today} ({note}). Label every quiz below easy, medium or hard by the demand on a "
        "learner who has worked through the lecture, judged blind: answer the quiz first, then mark the level by what "
        "the answer took. Put `x` in exactly one of the three `[ ]`; a quiz without a mark counts as not judged yet, "
        "and the `note:` line may say why. Easy is recall: one fact, definition or rule the lecture states, or one "
        "line of code read off directly. Medium is apply: one step, a rule applied to a new case, a short expression "
        "traced with one twist, two close concepts told apart, or a many-choice whose correct options each follow "
        "from one rule. Hard is analyse: several steps or concepts held together, nested or chained code, a "
        "calculation over a given table, a scenario whose condition flips the answer, or a distractor that is right "
        "in a nearby case. The count and the length of the options are not the criterion, the demand per option is. "
        "The rule in full: docs/draft/helper/models/difficulty.md, \"The hand-labelled sample\". Later runs keep the "
        "marks in hand_labels.csv and score D1's predictions on the judged quizzes.",
        "",
        f"## The unrated quizzes ({len(unrated)})",
        "",
    ]

    def entry(card) -> None:
        lines.append(f"### {card.id} ({card.deck}): {short(card.question)}")
        lines.append("")
        lines.extend(quoted(card.question))
        lines.append("")
        if not card.options:
            lines.append(f"- answer: {one_line(card.answer)}")
        else:
            for position, (option, feedback) in enumerate(zip(card.options, card.answers)):
                mark = "correct" if position in card.correct else "wrong"
                lines.append(f"- ({mark}) {one_line(option)}" + (f"  —  {one_line(feedback)}" if feedback.strip() else ""))
        row = labels.get(card.id, {})
        lines.extend(f"- [{'x' if row.get('level') == level else ' '}] {level}" for level in LEVELS)
        lines.append(f"- note: {row.get('note', '')}".rstrip())
        lines.append("")

    for card in unrated:
        entry(card)
    lines += [
        f"## The rated quizzes of {decks}, judged blind ({len(reference)})",
        "",
        "Their authored levels are not shown. The evaluation compares the hand levels with them, so the criterion is "
        "measured against the quota the levels were written under.",
        "",
    ]
    for card in reference:
        entry(card)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(unrated), len(reference)


# --- the similarity models and the map ------------------------------------------------


neighbour_scores = similar.neighbour_scores  # moved into the pipeline (S2's own minimum needs it there)


def tags_of(marks: dict[str, dict[str, dict]]) -> dict[str, set[tuple[str, str]]]:
    """The tagged pairs of the marks file by tag: a note that holds one of TAGS (whole words, any case) tags its
    (card, candidate) pair; a note may hold several."""
    tagged: dict[str, set[tuple[str, str]]] = {tag: set() for tag in TAGS}
    for card_id, rows in marks.items():
        for candidate, row in rows.items():
            note = (row.get("note") or "").lower()
            for tag in TAGS:
                if re.search(rf"\b{re.escape(tag)}\b", note):
                    tagged[tag].add((card_id, candidate))
    return tagged


def tag_report(indexes: dict[str, object], tagged: dict[str, set[tuple[str, str]]], k: int = similar.TOP_K) -> dict[str, dict[str, int]]:
    """Per variant and tag, how many of the tagged pairs the variant lists among the card's first k neighbours: a
    trap, a shared dataset or a generic term still listed counts against it, a paraphrase or a code name listed
    for it. Printed as a table; the counts returned by variant and tag."""
    counts: dict[str, dict[str, int]] = {}
    tags = [tag for tag in TAGS if tagged.get(tag)]
    if not tags:
        return counts
    print("   tagged pairs listed in the top 5 (" + ", ".join(f"{tag}: {len(tagged[tag])}" for tag in tags) + ")")
    print("   | Variant | " + " | ".join(tags) + " |")
    print("   |---|" + "---|" * len(tags))
    for name, index in indexes.items():
        counts[name] = {}
        for tag in tags:
            listed = sum(
                candidate in [other for other, _score in list(index.related(card_id))[:k]] for card_id, candidate in tagged[tag]
            )
            counts[name][tag] = listed
        print(f"   | {name} | " + " | ".join(f"{counts[name][tag]} of {len(tagged[tag])}" for tag in tags) + " |")
    return counts


def deck_report(per_card: dict[str, dict[str, float]], per_card_p1: dict[str, dict[str, float]], deck_of: dict[str, str],
                deck_ids: dict[str, str]) -> None:
    """The hand columns per sampled deck (hand p@5, p@1, the judged cards, the paired difference to the chosen
    variant), printed and not written: the transfer deck's numbers are read apart from the others'."""
    judged = sorted(per_card.get(CHOSEN, {}))
    decks = sorted({deck_of[card_id] for card_id in judged if card_id in deck_of}, key=lambda deck: deck_ids.get(deck, deck))
    if not decks:
        return
    print("   hand columns per deck (p@5, p@1 over the deck's judged cards; the difference to the chosen variant with its interval)")
    for deck in decks:
        cards = [card_id for card_id in judged if deck_of.get(card_id) == deck]
        print(f"   deck {deck_ids.get(deck, deck)} ({len(cards)} judged card{'s' if len(cards) != 1 else ''}"
              + (", the transfer deck" if deck == TRANSFER_DECK else "") + ")")
        for name in per_card:
            p5 = float(np.mean([per_card[name][card_id] for card_id in cards]))
            p1 = float(np.mean([per_card_p1[name][card_id] for card_id in cards]))
            mean, low, high = evaluation.paired_bootstrap([per_card[name][card_id] - per_card[CHOSEN][card_id] for card_id in cards])
            print(f"      {name:52s} p@5 {p5:.3f}, p@1 {p1:.2f}, difference {mean:+.3f} [{low:+.3f}, {high:+.3f}]")


def evaluate_similarity(cards, seeds, deck_ids: dict[str, str], today: str, note: str, marks: dict[str, set[str]],
                        cache_dir: Optional[Path] = None, vectoriser=None, tagged: Optional[dict[str, set[tuple[str, str]]]] = None,
                        loader: Optional[Callable] = None) -> tuple[list[dict], dict]:
    """B0, the S1 variants, the S2 variants (the encoders and the hybrids, with the library) and C1's map (the deck
    layout and the per-topic card layouts of the app's vectoriser, `vectoriser`, TF-IDF's chosen settings by
    default). `marks`: the judged cards' relevant candidates (card id -> ids), empty without the hand side;
    `tagged`: the tagged pairs by tag (tags_of), for the tag report; `cache_dir`: the embeddings' cache."""
    deck_of = {card.id: card.deck for card in cards}
    ids = [card.id for card in cards]
    rows: list[dict] = []

    # B0: the random floor, the models' own draw
    floors = [evaluation.random_floor(ids, deck_of, evaluation.random_directions(len(ids), seed)) for seed in SEEDS]
    for seed, floor in zip(SEEDS, floors):
        rows.append({"date": today, "model": "B0", "variant": "random positions, 5 nearest", "params": "uniform on the sphere",
                     "seed": seed, "precision_at_5": fmt(floor), "notes": note})
    print(f"B0 random floor: same-deck precision@5 {np.mean(floors):.3f} ± {np.std(floors):.3f} over seeds {SEEDS}")

    # S1 and S2: the vectoriser variants, without and with the minimum score, each at its own 10th percentile of
    # the fifth neighbour's cosine
    indexes = {}
    per_card: dict[str, dict[str, float]] = {}
    per_card_p1: dict[str, dict[str, float]] = {}
    variant_rows: dict[str, dict] = {}
    encoder_variants = s2_variants(cache_dir, loader)
    if not encoder_variants:
        print(S2_SKIPPED)
    for variant in variants() + encoder_variants:
        started = time.perf_counter()
        index = similar.SimilarityIndex.fit(cards, variant.make(), min_score=0.0, with_explain=variant.with_explain)
        seconds = time.perf_counter() - started
        cosines = similar.cosine_similarities(index.vectors)
        fifth = neighbour_scores(cosines, similar.TOP_K)
        percentiles = [float(np.percentile(fifth, share)) for share in (10, 50, 90)] if fifth.size else [math.nan] * 3
        min_score = percentiles[0]
        precision = evaluation.same_deck_precision(index.neighbours, deck_of)
        kept = similar.top_neighbours(ids, cosines, similar.TOP_K, min_score)
        kept_precision = evaluation.same_deck_precision(kept, deck_of)
        kept_only = evaluation.kept_precision(kept, deck_of)
        covered, distinct = evaluation.coverage(kept)
        vocabulary = getattr(index.vectoriser, "vocabulary_", None)
        params = variant.params + (f", vocab={len(vocabulary)}" if vocabulary is not None else "")
        if isinstance(index.vectoriser, embeddings.HybridVectoriser):
            params += ", minima=({:.3f}, {:.3f}), weights=({:.2f}, {:.2f})".format(*index.vectoriser.minima, *index.vectoriser.weights)
        encoder = index.vectoriser.second if isinstance(index.vectoriser, embeddings.HybridVectoriser) else index.vectoriser
        if isinstance(encoder, embeddings.SentenceEncoder):
            params += f", device={encoder.device or 'none (cache)'}"  # what the seconds were measured on; a cache hit loads no model
        per_card[variant.name] = evaluation.judged_precision(kept, marks, similar.TOP_K)
        per_card_p1[variant.name] = evaluation.judged_precision(kept, marks, 1)
        p1, p3, p5 = (hand_precision(kept, marks, k) for k in (1, 3, similar.TOP_K))
        row = {"date": today, "model": variant.model, "variant": variant.name, "params": params, "seed": "",
               "precision_at_5": fmt(precision), "kept_precision": fmt(kept_only), "coverage": fmt(covered),
               "distinct": distinct, "min_score": fmt(min_score), "hand_precision": fmt(p5), "hand_p1": fmt(p1), "hand_p3": fmt(p3),
               "judged_cards": len(marks) if marks else "", "seconds": f"{seconds:.2f}",
               "notes": f"precision with min_score {min_score:.3f}: {kept_precision:.3f}; {note}"}
        rows.append(row)
        variant_rows[variant.name] = row
        print(f"{variant.model} {variant.name:52s} precision@5 {precision:.3f} | 5th neighbour p10 {percentiles[0]:.3f}, median {percentiles[1]:.3f}, "
              f"p90 {percentiles[2]:.3f} | min_score {min_score:.3f}: precision {kept_precision:.3f}, kept-slot precision {kept_only:.3f}, "
              f"coverage {covered:.3f}, distinct {distinct}" + (f", vocab {len(vocabulary)}" if vocabulary is not None else "")
              + (f" | hand p@1 {p1:.2f}, p@3 {p3:.2f}, p@5 {p5:.3f} over {len(marks)} cards" if marks else "") + f" ({seconds:.2f}s)")
        indexes[variant.name] = similar.SimilarityIndex(index.ids, index.decks, index.vectors, kept, index.vectoriser)

    # The paired difference of hand p@5 to the chosen variant, bootstrapped over the judged cards
    if marks:
        judged = sorted(marks)
        for name, row in variant_rows.items():
            differences = [per_card[name][card_id] - per_card[CHOSEN][card_id] for card_id in judged]
            mean, low, high = evaluation.paired_bootstrap(differences)
            row.update(hand_diff=fmt(mean), hand_diff_low=fmt(low), hand_diff_high=fmt(high))
            print(f"   {name:52s} hand p@5 difference to the chosen variant {mean:+.3f} [{low:+.3f}, {high:+.3f}]"
                  + (" (itself)" if name == CHOSEN else ""))
        deck_report(per_card, per_card_p1, deck_of, deck_ids)
        if tagged:
            tag_report(indexes, tagged)

    # The score distribution of the chosen settings (the basis of MIN_SCORE)
    chosen = indexes[CHOSEN]
    cosines = similar.cosine_similarities(chosen.vectors)
    for rank in range(1, similar.TOP_K + 1):
        column = neighbour_scores(cosines, rank)
        if column.size:
            print(f"  chosen, score of neighbour {rank}: p10 {np.percentile(column, 10):.3f}, median {np.median(column):.3f}, "
                  f"p90 {np.percentile(column, 90):.3f}")

    # C1: the app's model's positions (the deck layout and the card layouts per model topic), against the SVD
    # start and random directions; the app's vectoriser at its own minimum, as the launchers fit it
    started = time.perf_counter()
    model = CourseModel(cards, deck_ids, seeds, vectoriser, min_score=None)
    seconds = time.perf_counter() - started
    start = layout.svd_start(model.deck_vectors)
    random_decks = evaluation.random_directions(len(model.deck_points), 42)
    deck_measures = {
        "model": (model.deck_trustworthiness, model.deck_overlap),
        "SVD start": (layout.trustworthiness_score(model.deck_vectors, start, DECK_NEIGHBOURS), layout.neighbour_overlap(model.deck_vectors, start, DECK_NEIGHBOURS)),
        "random": (layout.trustworthiness_score(model.deck_vectors, random_decks, DECK_NEIGHBOURS), layout.neighbour_overlap(model.deck_vectors, random_decks, DECK_NEIGHBOURS)),
    }
    kinds = ("model", "SVD start", "random")
    layouts: dict[str, list[tuple]] = {kind: [] for kind in kinds}  # (deck, vectors, points) per topic with two cards or more
    row_of = {card_id: row for row, card_id in enumerate(model.index.ids)}
    for placed in model.topics.values():
        if len(placed.ids) < 2:
            continue
        vectors = layout.as_dense(model.index.vectors[[row_of[card_id] for card_id in placed.ids]])
        points = (
            np.array([placed.positions[card_id] for card_id in placed.ids]),
            layout.svd_start(vectors),
            evaluation.random_directions(len(placed.ids), 42),
        )
        for kind, laid in zip(kinds, points):
            layouts[kind].append((placed.deck, vectors, laid))
    laid_out = len(layouts["model"])
    fixed = {kind: {k: evaluation.mean_overlap([(vectors, points) for _, vectors, points in layouts[kind]], k) for k in FIXED_K} for kind in kinds}
    params = (f"k_decks={DECK_NEIGHBOURS}, k_cards=min({CARD_NEIGHBOURS}, n-1), fixed k over the topics with k+2 cards or more, "
              f"iterations={layout.ITERATIONS}, step={layout.STEP}, repulsion={layout.REPULSION}")
    for kind in kinds:
        trust, overlap = deck_measures[kind]
        own_k = [(deck, layout.neighbour_overlap(vectors, points, min(CARD_NEIGHBOURS, len(vectors) - 1)))
                 for deck, vectors, points in layouts[kind]]
        trusts = [layout.trustworthiness_score(vectors, points, min(CARD_NEIGHBOURS, len(vectors) - 1)) for _, vectors, points in layouts[kind]]
        trusts = [value for value in trusts if not math.isnan(value)]
        mean_cards = float(np.nanmean([value for _, value in own_k])) if own_k else math.nan
        overlap_17 = float(np.nanmean([value for deck, value in own_k if deck == "NLP Fundamentals"] or [math.nan]))
        overlap_24 = float(np.nanmean([value for deck, value in own_k if deck == "LLM Foundations & Embeddings"] or [math.nan]))
        lifts = {k: evaluation.lift(fixed[kind][k][0], fixed["random"][k][0]) for k in FIXED_K}
        rows.append({"date": today, "model": "C1", "variant": kind, "params": params, "seed": 42 if kind == "random" else "",
                     "trust_decks": fmt(trust), "overlap_decks": fmt(overlap), "overlap_cards": fmt(mean_cards),
                     "overlap_k2": fmt(fixed[kind][2][0]), "lift_k2": fmt(lifts[2]), "overlap_k3": fmt(fixed[kind][3][0]), "lift_k3": fmt(lifts[3]),
                     "trust_cards": fmt(float(np.mean(trusts))) if trusts else "",
                     "overlap_17": fmt(overlap_17), "overlap_24": fmt(overlap_24),
                     "seconds": f"{seconds:.2f}" if kind == "model" else "",
                     "notes": f"deck layout and card layouts per model topic ({laid_out} topics with two cards or more, "
                              f"{fixed[kind][2][1]} at k = 2, {fixed[kind][3][1]} at k = 3, trust_cards over {len(trusts)} topics); {note}"})
        print(f"C1 {kind:10s} decks: trustworthiness {trust:.3f}, top-3 overlap {overlap:.3f} | cards: within-topic overlap "
              f"{mean_cards:.3f} at k = min(5, n-1), deck 17 {overlap_17:.3f}, deck 24 {overlap_24:.3f} | k = 2: {fixed[kind][2][0]:.3f} "
              f"(lift {lifts[2]:.3f}, {fixed[kind][2][1]} topics), k = 3: {fixed[kind][3][0]:.3f} (lift {lifts[3]:.3f}, {fixed[kind][3][1]} topics) "
              f"| trustworthiness {np.mean(trusts) if trusts else math.nan:.3f} over {len(trusts)} topics")
    print(f"the whole model (index, deck layout, T1, {laid_out} topic layouts, clustering) built in {seconds:.2f}s; "
          f"{sum(len(placed.links) for placed in model.topics.values())} card connections, {len(model.deck_links)} deck connections, "
          f"{len(model.hidden)} cards hidden")
    return rows, indexes


# --- the run --------------------------------------------------------------------------


def run(table_dir: Path, results_dir: Path, part: str = "all", judge: str = "", today: Optional[str] = None,
        run_key: Optional[str] = None, cache_dir: Optional[Path] = None, vectoriser=None, tagger_loader: Optional[Callable] = None) -> None:
    """The evaluation on the course table in `table_dir`, its rows into `results_dir`: one part or all three, every
    row stamped with the run's key (`run_key`, by default the start time to the minute). The hand side of a part
    runs where its file or its form is in `results_dir`: with the form, the file is kept and the form written again;
    with the file alone (the capstone), the hand columns are computed and nothing is written back; with neither,
    the hand columns stay empty and no form is written. The parts measure the app's vectoriser (`vectoriser`, by
    default embeddings.app_vectoriser with the embeddings cached in `cache_dir`), next to the similarity part's
    variants; the difficulty part takes D2's tags from the same cache or the NLI models (`tagger_loader` stands in
    for the pipeline in the tests)."""
    files = result_files(results_dir)
    if vectoriser is None:
        vectoriser = embeddings.app_vectoriser(cache_dir)
    table = course_table.load_table(table_dir)
    cards, deck_ids, seeds = table.cards, table.deck_ids, table.seeds
    by_id = {card.id: card for card in cards}
    note = scope_note(deck_ids)
    today = today or date.today().isoformat()
    run_key = run_key or datetime.now().strftime(RUN_FORMAT)
    marks_form, labels_form = files.hand_check.exists(), files.hand_labels_form.exists()
    with_marks = files.hand_marks.exists() or marks_form
    with_labels = files.hand_labels.exists() or labels_form
    marks = merge_marks(read_marks(files.hand_marks), read_hand_check(files.hand_check), today, judge) if with_marks else {}
    labels = merge_labels(read_labels(files.hand_labels), read_label_form(files.hand_labels_form), today, judge) if with_labels else {}
    print(f"{len(cards)} cards in {len(deck_ids)} decks ({table_dir}, {note}); hand-check marks for {len(marks)} cards, "
          f"hand labels for {len(labels)} quizzes; the rows go to {results_dir} (run {run_key}); the app's vectoriser: "
          f"{embeddings.describe(vectoriser)}" + (f", the embeddings cached in {cache_dir}" if cache_dir is not None else ""))

    # T1: the topic model, into its own file
    if part in ("all", "topics"):
        topics = stamped(evaluate_topics(cards, seeds, deck_ids, today, note, vectoriser), run_key)
        write_rows(topics, files.topics, TOPICS_COLUMNS)
        print(f"{len(topics)} rows appended to {files.topics}")
    # D1: the difficulty model, into its own file, with the hand labels kept and their form written again
    if part in ("all", "difficulty"):
        tags = difficulty_tags(cards, cache_dir, tagger_loader)
        difficulty = stamped(evaluate_difficulty(cards, today, note, labels, vectoriser, tags, deck_ids), run_key)
        write_rows(difficulty, files.difficulty, DIFFICULTY_COLUMNS)
        print(f"{len(difficulty)} rows appended to {files.difficulty}")
        if labels_form:
            write_labels(labels, files.hand_labels)
            unrated, reference = write_hand_labels_form(cards, labels, today, note, files.hand_labels_form)
            print(f"hand labels: {unrated} unrated quizzes and {reference} rated ones judged blind in {files.hand_labels_form}; "
                  f"{len(labels)} judged, kept in {files.hand_labels}")
        elif with_labels:
            print(f"hand labels: {len(labels)} judged quizzes read from {files.hand_labels}; no {files.hand_labels_form.name} "
                  f"in {results_dir}: nothing written back")
        else:
            print(f"hand labels: no {files.hand_labels.name} in {results_dir}: the hand columns stay empty and "
                  f"{files.hand_labels_form.name} is not written")
    if part not in ("all", "similarity"):
        return

    rows, indexes = evaluate_similarity(cards, seeds, deck_ids, today, note, relevant_of(marks), cache_dir, vectoriser, tags_of(marks))
    write_rows(stamped(rows, run_key), files.results)
    if marks_form:
        write_marks(marks, files.hand_marks)
        listed = write_hand_check(cards, by_id, indexes, today, marks, note, files.hand_check)
        print(f"{len(rows)} rows appended to {files.results}; hand check for {listed} cards in {files.hand_check}"
              + (f" ({len(marks)} judged, the marks kept in {files.hand_marks})" if marks else " (no marks yet)"))
    elif with_marks:
        print(f"{len(rows)} rows appended to {files.results}; hand check: the marks of {len(marks)} cards read from "
              f"{files.hand_marks}; no {files.hand_check.name} in {results_dir}: nothing written back")
    else:
        print(f"{len(rows)} rows appended to {files.results}; hand check: no {files.hand_marks.name} in {results_dir}: "
              f"the hand columns stay empty and {files.hand_check.name} is not written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=None, help=f"the course table's folder (default: {default_table_dir()})")
    parser.add_argument("--results", default=None, help=f"the folder of the results files (default: {RESULTS_DIR})")
    parser.add_argument("--part", choices=("all", "similarity", "difficulty", "topics"), default="all",
                        help="the similarity models (B0, S1, C1), the difficulty model D1, the topic model T1, or all (default)")
    parser.add_argument("--judge", default="", help="who marked hand_check.md and hand_labels.md: written into hand_marks.csv "
                        "and hand_labels.csv with every new or changed mark")
    parser.add_argument("--cache", default=None, help=f"the folder of the cache of S2's embeddings and D2's tags (default: {default_cache_dir()})")
    arguments = parser.parse_args()
    table_dir = Path(arguments.table) if arguments.table else default_table_dir()
    missing = course_table.missing_files(table_dir)
    if missing:
        raise SystemExit(f"No course table in {table_dir} ({', '.join(path.name for path in missing)} missing).")
    run(table_dir, Path(arguments.results) if arguments.results else RESULTS_DIR, arguments.part, arguments.judge,
        cache_dir=Path(arguments.cache) if arguments.cache else default_cache_dir())


if __name__ == "__main__":
    main()
