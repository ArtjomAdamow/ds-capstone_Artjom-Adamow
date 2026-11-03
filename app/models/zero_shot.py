"""D2: zero-shot difficulty tags (app roadmap step 12; docs/draft/helper/models/difficulty.md, "D2"): the demand on a
learner read off the quiz with its resolution by natural language inference, nothing trained.

    the question, its options, the correct feedback, the explanation ─→ NLI model, three hypotheses, one softmax
        ─→ P(recall), P(apply), P(analyse); the score P(apply) + 2 · P(analyse), 0..2 as D1's
    (the scores cached by model, template and texts, as the embeddings are)

* `premise(card)`: the quiz as the hand judge read it: the question, each answer option on a line in file order,
  the feedback of the correct options, then the explanation without its closing `Source:` line (the origin's
  template line of a repo question); the wrong options' feedback stays out (an explanation in every rated quiz,
  "try again" in the unrated ones). A card without options gives its question, its answer and its explanation.
* `Template`: the hypothesis template and its three labels, one per level. CRITERION, the primary: the hand
  criterion's three demands (recall, apply, analyse), written before D2 existed, so nothing is tuned; LEVEL_NAMES,
  the control: the lesson's own form with the level names. NLI_MODELS: the base model and the large one of the same
  family (app roadmap step 15), each read by the evaluation with both templates; NLI_MODEL the one the app reads.
* `ZeroShotTagger`: transformers' zero-shot-classification pipeline on S2's device (embeddings.resolve_device), the library imported when the
  model is loaded, so the package imports without torch; `scores(cards)` gives one row of three probabilities per
  card (one softmax over the three hypotheses), from the cache where it holds them (the embeddings' cache class,
  the file named by the model and the template, keyed by the premises in order).
* `tags_of`: the tags of a course's quizzes (the rated and the unrated ones; flashcards are not tagged) for the
  evaluation and the app, as `Tags`; None with one printed line without the library or the model. `cached_tags`:
  the same from the cache alone, for the standalone page, which loads no model.
"""

from __future__ import annotations

import importlib.util
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .difficulty_model import LEVELS, split
from .embeddings import CACHE_FOLDER, EmbeddingCache, resolve_device

# The models: trained for zero-shot classification with a hypothesis template, an 8,192-token window, so no premise
# is cut (difficulty.md, "D2", "Decisions"): the base model (299 MB, step 12) and the large one of the same family
# (792 MB, app roadmap step 15), both measured by the evaluation. NLI_MODEL is the app's D2: the large one, with
# which alone the criterion template reads above chance on the hand labels (53 against 41 of 77 pairs; a choice
# between two on the labels, declared in difficulty.md, "D2", "Decisions"); the page reads its tags from the cache
BASE_MODEL = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
LARGE_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
NLI_MODELS = (BASE_MODEL, LARGE_MODEL)
MODEL_SIZES = {BASE_MODEL: "base", LARGE_MODEL: "large"}  # how the evaluation's rows name them
NLI_MODEL = LARGE_MODEL
LIBRARY = "transformers"
BATCH_SIZE = 16
FETCH_COMMAND = "python -c \"from transformers import pipeline; pipeline('zero-shot-classification', model={name!r})\""
SOURCE_LINE = re.compile(r"\s*\bSource:.*\Z", re.DOTALL)  # the closing line of a repo question's explanation


@dataclass(frozen=True)
class Template:
    """A hypothesis template of the pipeline ({} takes the label) and its three labels, one per level of LEVELS."""

    name: str
    hypothesis: str
    labels: tuple[str, str, str]


CRITERION = Template(
    "criterion", "To answer this question, a learner must {}.",
    ("recall one fact or rule the lesson states", "apply one rule to a new case", "reason through several steps or concepts at once"),
)
LEVEL_NAMES = Template("level names", "This question is {}.", LEVELS)
TEMPLATES = (CRITERION, LEVEL_NAMES)  # the primary first, the control second


def premise(card) -> str:
    """The text D2 classifies: the question, the options one per line, the correct options' feedback, the
    explanation without its `Source:` line; without options the question, the answer and the explanation."""
    parts = [(card.question or "").strip()]
    options = tuple(getattr(card, "options", ()) or ())
    if options:
        answers = tuple(getattr(card, "answers", ()) or ())
        parts.extend(option.strip() for option in options)
        parts.extend(
            answers[index].strip() for index in getattr(card, "correct", ()) if index < len(answers) and answers[index].strip()
        )
    elif (card.answer or "").strip():
        parts.append(card.answer.strip())
    explain = SOURCE_LINE.sub("", card.explain or "").strip()
    if explain:
        parts.append(explain)
    return "\n".join(part for part in parts if part)


def size_of(model_name: str) -> str:
    """The model's short name in the evaluation's rows: base, large, or the name's last part."""
    return MODEL_SIZES.get(model_name, model_name.rsplit("/", 1)[-1])


def expected_level(probabilities: Sequence[float]) -> float:
    """D2's score on D1's scale: P(apply) + 2 · P(analyse), 0 to 2."""
    return float(probabilities[1] + 2.0 * probabilities[2])


def tagger_available(find_spec: Callable = importlib.util.find_spec) -> bool:
    """Whether transformers imports here (nothing is imported)."""
    try:
        return find_spec(LIBRARY) is not None
    except (ImportError, ValueError):
        return False


class ZeroShotTagger:
    """The zero-shot-classification pipeline of one model and one template on `device` (embeddings.resolve_device's
    when None, as S2's encoder); `loader(model_name, device)` stands in for the pipeline in the tests. `scores(cards)` reads the course's tags from the cache where it holds them, else
    classifies the premises and caches them; `seconds` and `cached` say what the last call did."""

    def __init__(self, model_name: str = NLI_MODEL, template: Template = CRITERION, cache: Optional[EmbeddingCache] = None,
                 batch_size: int = BATCH_SIZE, loader: Optional[Callable] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.template = template
        self.cache = cache
        self.batch_size = batch_size
        self._loader = loader
        self._classifier = None
        self.device = device  # resolved at the load: "cpu" or "cuda"
        self.seconds = 0.0
        self.cached = False

    @property
    def cache_name(self) -> str:
        """The cache's model name: the model and the template, so each template has a file of its own."""
        return f"{self.model_name} {self.template.name}"

    def load(self):
        """The pipeline, loaded once (the library imported here, so the package imports without torch)."""
        if self._classifier is None:
            self.device = resolve_device(self.device)
            if self._loader is not None:
                self._classifier = self._loader(self.model_name, self.device)
            else:
                from transformers import pipeline

                self._classifier = pipeline("zero-shot-classification", model=self.model_name, device=self.device)
        return self._classifier

    def classify(self, premises: Sequence[str]) -> np.ndarray:
        """One row of three probabilities per premise, in the order of the template's labels (one softmax over the
        three hypotheses: the pipeline without multi_label)."""
        classifier = self.load()
        premises = list(premises)
        if not premises:
            return np.zeros((0, len(LEVELS)), dtype=np.float32)
        started = time.perf_counter()
        results = classifier(premises, candidate_labels=list(self.template.labels), hypothesis_template=self.template.hypothesis,
                             multi_label=False, batch_size=self.batch_size)
        self.seconds = time.perf_counter() - started
        if isinstance(results, dict):
            results = [results]
        rows = []
        for result in results:
            by_label = dict(zip(result["labels"], result["scores"]))
            rows.append([float(by_label[label]) for label in self.template.labels])
        return np.asarray(rows, dtype=np.float32).reshape(len(premises), len(LEVELS))

    def scores(self, cards: Sequence) -> np.ndarray:
        """The cards' tags: from the cache where it holds them, else classified and cached."""
        premises = [premise(card) for card in cards]
        scores = self.cache.read(self.cache_name, premises) if self.cache is not None else None
        self.cached = scores is not None
        if scores is None:
            scores = self.classify(premises)
            if self.cache is not None and premises:
                self.cache.write(self.cache_name, premises, scores)
        else:
            self.seconds = 0.0
        return scores


@dataclass(frozen=True)
class Tags:
    """The tags of a course's quizzes under one template: card id -> the three probabilities (LEVELS' order)."""

    template: Template
    model_name: str
    scores: dict[str, np.ndarray]
    seconds: float  # the classification's time, 0 for a cache hit
    cached: bool
    device: str = ""  # the device that classified them, "" for a cache hit

    def expected(self) -> dict[str, float]:
        """Every quiz's score, the expected level."""
        return {card_id: expected_level(probabilities) for card_id, probabilities in self.scores.items()}


def tags_of(cards: Sequence, cache_dir: Optional[Path] = None, template: Template = CRITERION, model_name: str = NLI_MODEL,
            output: Callable = print, loader: Optional[Callable] = None) -> Optional[Tags]:
    """The tags of the course's quizzes (the rated and the unrated ones) under `template`, cached in `cache_dir`
    (the launchers' folder, with the embeddings). None, with one line that says why and how to fetch the model,
    without the library or the model."""
    if loader is None and not tagger_available():
        output(f"D2's model {model_name} needs transformers (the embeddings group); the zero-shot tags are skipped.")
        return None
    rated, unrated = split(cards)
    quizzes = rated + unrated
    cache = EmbeddingCache(Path(cache_dir) / CACHE_FOLDER) if cache_dir is not None else None
    tagger = ZeroShotTagger(model_name, template, cache=cache, loader=loader)
    try:
        scores = tagger.scores(quizzes)
    except Exception as error:  # noqa: BLE001  (a missing model, an offline hub: the evaluation and the app go on)
        output(f"D2's model {model_name} could not be loaded ({error.__class__.__name__}); the zero-shot tags are skipped. "
               f"Fetch it once with: {FETCH_COMMAND.format(name=model_name)}")
        return None
    return Tags(template, model_name, {card.id: scores[position] for position, card in enumerate(quizzes)}, tagger.seconds, tagger.cached,
                tagger.device or "")


def cached_tags(cards: Sequence, cache_dir: Optional[Path], template: Template = CRITERION,
                model_name: str = NLI_MODEL) -> Optional[Tags]:
    """The tags of the course's quizzes from the cache alone, as `tags_of` wrote them: no model is loaded and no
    library is needed (the standalone page's D2). None without the folder or where the cache does not hold them."""
    if cache_dir is None:
        return None
    rated, unrated = split(cards)
    quizzes = rated + unrated
    tagger = ZeroShotTagger(model_name, template, cache=EmbeddingCache(Path(cache_dir) / CACHE_FOLDER))
    scores = tagger.cache.read(tagger.cache_name, [premise(card) for card in quizzes]) if quizzes else None
    if scores is None:
        return None
    return Tags(template, model_name, {card.id: scores[position] for position, card in enumerate(quizzes)}, 0.0, True)
