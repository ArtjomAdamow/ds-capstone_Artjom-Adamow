"""The standalone page's models and its one callback (Dash; nothing of the viewer).

Page fits the course model, D1 and P1 on the course table once, as the viewer's serving process does, S1's TF-IDF
index next to it where the app runs S2's encoder, and D2's report from the tags the launcher's cache holds; it reads
the results files of the offline evaluation, and answers what the three views, the map's layers and the right
column's regions show; everything shown is the models', nothing is stored.

The views: the decks on the sphere; a deck's topics (an entered deck) at random positions, since the model places
no topic yet, with the deck's tree of its topics in the figure slot (topic_tree: the clustering of the topic
vectors of the membership in force, the full tree, no line on the map); a topic's cards (an entered topic) on the
surface of its sphere.

The map by layers: the check boxes inside the map (the `layers` key of the `view` store, the checked model
letters, none at load) say which models draw, each on a channel of its own, and the models' own dependencies are
kept (Page.layer_states: a layer is on, on its fallback, or not functioning, and its check box says which):

* S draws the neighbour pairs inside the entered topic as lines along the surface, the selected card's bright, from
  the generation open in its tab, else the app's (S2; S1 without the encoder); it needs nothing;
* C draws the positions (the decks', a topic's cards') and the connections below the preset, and a deck's centroid;
  it uses S's vectors and T1's topics: with S off it does not function, with T off it shows its fallback, the
  scripted topics laid out and clustered (CourseModel.layout_of, fitted on first use and kept);
* T draws the membership (T1 keeps the scripted topic ids), the fill by margin on a teal scale, a cross below the
  threshold, a violet diamond for a card T1 moved in from another scripted topic, an open pink diamond for a card
  scripted here that T1 moved elsewhere, placed at the normalised mean direction of its most similar cards among the
  topic's members, an emptied topic as an open circle, a topic's dot grown with its share of cards below the
  threshold; with S off it does not function;
* D draws the ring colours: a rated quiz's by its level, a predicted quiz's amber, its level on the hover (and the
  other generation's where they disagree, with a ring behind), from the generation open in its tab, else D1; it
  needs nothing (D1 on its text statistics, D2 on its tags; D2 without cached tags says so);
* P draws P1's answer as the highlighted target; it uses S's index and T1's membership: with S off it does not
  function and the request box is disabled, with T off it shows its fallback, P1 over the scripted topics (a
  PromptModel on their membership, fitted on first use and kept).

The bare channels: without C the positions are random (evaluation.random_directions over the view's sorted ids
with a fixed seed, the same in every state); without T in force the membership is the scripted topics, the fill
grey; the topics view stays at random in every state. Nothing checked, the bare course.

The right column: the tab bar (the model letters S, C, T, D and P; one open at a time, the `panel` key of the
store, a second click closes it), the open model's generations as tabs (layout.GENERATIONS; the newest open when
the tab opens, the `level` key, None for the newest; the "app" mark on the one the app runs, Page.app_levels), the
section (Page.section: the headline, the generation's two numbers with their contrast, sentence, verdict and
explanation from the results rows, app.headlines; the chosen variant in plain words; with no tab open one
line on the five models; on top for the live demo T's line of the view next to the course and P's request box with
its answer: the decision line and the three nearest by label, each a click away), the selected dot's tag strip
(Page.tag_strip: its title and tags; hidden with nothing selected) and the figure slot (Page.figure_slot: one
figure by the open tab, or Details, Page.details, in its place: the open generation's description, its live
measures, the tables of its newest run whose rows open their description on a click, one at a time, the `measure`
key, and what moved in).

The breadcrumb (Page.where), the tag strip and the map's hover name the dots and the views by id: no learning
material, no card's text, no deck's description and no deck's name. A topic's name, the one name that is no id,
shows nowhere either: the hover gives its label instead (Page.topic_label: its topic id, then a hash of its name,
made at runtime and stored nowhere), the breadcrumb and the dendrogram's leaves its id alone, and a results cell
that names a deck or a topic shows the id or the label (results.Results.by_ids).

The callback (register_callback) keeps the view in the `view` store: a click on a dot selects it, Open or the tag
strip's Open enters a selected deck or topic, Back goes up a level with the dot it came from selected (in the decks
view it unselects); a check box's change sets the layers; a tab opens its model or closes it; the Details button
opens Details in the figure slot's place or closes it; a generation's tab opens that generation; a results row's
click opens its description or closes the open one (both through the hidden input `row-click`, a JSON of the
generation or the row's key and a sequence number, or of P1's target); Go or Enter in P's box sends the request
(Page.find: P1's answer, the request, its time and its target kept in the store's `prompt`, the view navigated
into the target where P1 navigates; "no match" moves nothing). A tab or generation change closes Details and its
open row. No keys, groups, editing, confidence, quiz or image window.
"""

import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from dash import Dash, Input, Output, State, html
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..course_table import CourseTable
from ..figures import (
    ELSEWHERE, HERE, PREDICTED_COLOR, THIS_DECK, confusion_figure, dendrogram_figure, margin_color, margin_histogram, neighbours_figure,
    number,
)
from ..headlines import chosen_label, headlines, shown
from ..models import clustering, embeddings, similar
from ..models.course_model import CARD_NEIGHBOURS, DECK_NEIGHBOURS, CourseModel, Point, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty, judge_tags
from ..models.evaluation import random_directions
from ..models.layout import as_dense, fibonacci_sphere
from ..models.prompt_model import CARD, CENTROID, DEFAULT_ROUTES, TEXT, PromptModel, variant_label
from ..panels import CARDS_VIEW, CUT_LABELS, DECK_PART, DECKS_VIEW, PANEL_LETTERS, TOPICS_VIEW, d2_label, empty_slot, graph, s2_label
from ..results import LEVEL_FILES, Results, course_note, read_glossary
from .layout import (
    CROSS_SYMBOL, GENERATIONS, LEVEL_COLORS, MOVED_IN_COLOR, MOVED_IN_SYMBOL, MOVED_OUT_COLOR, MOVED_OUT_SYMBOL, NO_CARDS, NO_LINKS,
    OVERVIEW_HINT, PREDICTED, PROMPT_HINT, PROMPT_OFF, STACK_NAMES, TOPICS_NOTE, card_class, details_class, layer_options,
    level_section, level_tabs, live_line, map_figure, metrics_panel, model_readout, no_tab_line, page_layout, prompt_answer, prompt_class,
    section_head, tab_class, tag_class, tag_strip_body, topic_counts_of,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"  # next to evaluate.py: the capstone's root
TITLE = "Guided learning: the models"
CARD_PART = 4  # a topic's preset: parts of at most four cards
TOPIC_HASH_BYTES = 3  # a topic label's hash: six hex digits
RANDOM_SEED = 42  # the random positions of the scripted topics: one draw per view, the same in every state
MARGIN_PERCENTILE = 95  # the top of the teal scale: the course's 95th percentile of the margins
SIMILAR_MEMBERS = 5  # a moved-out card sits at the mean direction of this many most similar members of its topic
HOME = {"view": DECKS_VIEW, "deck": None, "topic": None, "selected": None, "panel": None, "level": None, "measure": None,
        "prompt": None, "details": False, "layers": []}
ENTER_TRIGGERS = ("open", "tag-open")
PROMPT_TRIGGERS = ("prompt-go", "prompt-text")  # Go, and Enter in the request box (its n_submit)
ROW_INPUT = "row-click"  # the hidden input a generation's tab, a results row's or a target's click sets (assets/panel_rows.js)
LAYERS_INPUT = "layers"  # the check boxes inside the map
DETAILS_TRIGGER = "details-toggle"
DECK_TABLE_LEVELS = ("T1", "D1", "D2")  # the generations with a table of the entered deck's rows
# A layer's state (Page.layer_states) and the positions' channel (Page.positions_of)
ON, FALLBACK, OFF = "on", "fallback", "off"
MODEL, SCRIPTED, RANDOM = "model", "scripted", "random"
Key = tuple[str, str]  # (deck id, topic id)
NO_ENCODER = (
    "The app runs on TF-IDF here: S2's encoder needs sentence-transformers (the embeddings group) and its model, so S2 "
    "has no live measures and the map shows nothing of it."
)
NO_TAGS = (
    "No D2 tags of this course in the cache: evaluate.py --part difficulty tags the quizzes once with the embeddings "
    "group (transformers and the model) into the launcher's cache, where the page reads them at its start (the "
    "criterion template on the large model, the app's D2)."
)
TOPIC_TREE_NOTE = "The clustering of the deck's topics by their vectors, the mean of their cards', the full tree: the topic connections the viewer's Helper writes."
# The readout's hints, a row's tooltip
TRUSTWORTHINESS = "scikit-learn's trustworthiness of the map for the text vectors: 1 when none of a dot's k nearest on the map is a stranger by text."
OVERLAP = "The share of a dot's k most similar by text that are also among its k nearest on the map, averaged over the dots."
MEAN_OVERLAP = (
    f"The mean over the topics of more than {CARD_NEIGHBOURS + 1} cards: in a smaller one every other card is a neighbour "
    "(k = n − 1), so the overlap is 1 by construction."
)
MEAN_TRUSTWORTHINESS = "The mean over the topics of more than 2k cards, where scikit-learn defines it."
BETWEEN = "The cards whose topic T1 chose by a margin below {margin} (the best centroid's cosine minus the second best's)."
MAP_S = "On the map S's layer draws the neighbour pairs inside the entered topic as lines, the selected card's bright."
DESCRIPTIONS = {
    "S1": "One TF-IDF over the course's card texts, unit rows, the cosine as the similarity: every card's five nearest "
          "cards above its own minimum score; the app's vectors until S2 took them over. " + MAP_S,
    "S2": "A sentence encoder over the course's card texts, alone or hybrid with S1's TF-IDF (the two halves' unit rows "
          "side by side, each at its own minimum), unit rows, the cosine as the similarity: every card's five nearest "
          "cards above the index's own minimum score; the vectors the app and every other model read. " + MAP_S,
    "C1": "The course model: the layout step on the app's vectors places the decks and, per topic, the cards on the surface "
          "of the topic's sphere; the clustering gives the connections and the cut-off preset. It builds on T1's topics; "
          "on the map C's layer draws the positions and the connections.",
    "T1": "The topic model: the nearest of the deck's scripted topic centroids by cosine, leave-one-out, on the app's "
          "vectors; the scripted topics are its input, not the truth. On the map T's layer draws its membership.",
    "D1": "The difficulty model the app runs: logistic regression on the question's statistics, labelled by the deck "
          "shares, predicts the level of the unrated quizzes; it reads neither C1 nor T1. On the map D's layer draws the "
          "levels as rings.",
    "P1": "The navigation prompt: a request in the learner's words is scored against T1's topics by three routes on the "
          "app's vectors (the nearest card, the topic's centroid, its headings), each at its floor from the cards' own "
          "questions; the first topic is entered where the card route clears its floor there, its nearest card "
          "selected with it, else no match. No language model. On the map P's layer marks the target.",
    "D2": "Zero-shot tags: an NLI model (the large ModernBERT) reads each quiz with its resolution against "
          "the hand criterion's three demands (the criterion template), nothing fitted, on D1's frame: the expected level "
          "is cut by the deck shares. Within noise of D1's order on the hand labels and below it, so a judge of the "
          "evaluation and D1 stays the app's. The page reads the tags from the launcher's cache; D's layer draws its "
          "levels as rings, a ring behind a quiz where it disagrees with D1.",
}


def merges_of(vectors) -> np.ndarray:
    """scipy's linkage of the rows, as models.clustering.merge_links makes it (the same distances and method), so
    its merges are the connections' merge order with their real distances."""
    return linkage(squareform(clustering.cosine_distances(vectors), checks=False), method=clustering.METHOD)


def _share(part: int, whole: int) -> str:
    return f"{part:,} ({part / whole:.0%})" if whole else "–"


def _mean(values: list[float]) -> str:
    """The mean and how many topics it covers; a dash without any."""
    if not values:
        return "–"
    return f"{number(sum(values) / len(values))} ({len(values)} topic{'' if len(values) == 1 else 's'})"


class Page:
    """The models fitted on the course table, the results files, and what the views, the layers and the regions
    show of them."""

    ON, FALLBACK, OFF = ON, FALLBACK, OFF

    def __init__(
        self, table: CourseTable, model: Optional[CourseModel] = None, difficulty: Optional[DifficultyReport] = None,
        results_dir: Optional[Path] = None, vectoriser=None, min_score: Optional[float] = similar.MIN_SCORE,
        tags: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> None:
        """`vectoriser` and `min_score` are the index's when the model is fitted here: the app's vectoriser at its
        own minimum from the launchers (standalone.main), TF-IDF at the fixed minimum by default. `tags` are D2's,
        card id -> the three probabilities (zero_shot.cached_tags), for D2's report; None: D2 says how to get them."""
        self.table = table
        self.model = model if model is not None else CourseModel(table.cards, table.deck_ids, table.seeds, vectoriser, min_score=min_score)
        self.difficulty = difficulty if difficulty is not None else fit_difficulty(table.cards, self.model.index)
        # D2 judged on the cached tags, nothing fitted; the generations' reports by name
        self.reports: dict[str, Optional[DifficultyReport]] = {"D1": self.difficulty, "D2": judge_tags(table.cards, tags) if tags else None}
        # S1 and S2, each its own index: S2's is the app's where it runs an encoder, and S1's TF-IDF is then fitted
        # here at its own minimum; without the encoder the app's index is S1's and S2 has none
        index = self.model.index
        encoded = isinstance(index.vectoriser, (embeddings.HybridVectoriser, embeddings.SentenceEncoder))
        self.indexes: dict[str, Optional[similar.SimilarityIndex]] = {
            "S1": similar.SimilarityIndex.fit(table.cards, min_score=None) if encoded else index, "S2": index if encoded else None,
        }
        # The generation the app runs per model, the "app" mark: S1 where the page runs on TF-IDF
        self.app_levels = {"S": "S2" if encoded else "S1", "C": "C1", "T": "T1", "D": "D1", "P": "P1"}
        self.topic_model = self.model.topic_model
        self.card_by_id = {card.id: card for card in table.cards}
        self.deck_by_id = {deck.deck_id: deck for deck in table.decks}
        self._row_of = {card_id: row for row, card_id in enumerate(self.model.index.ids)}
        self._deck_of_card = {card.id: self.model.deck_ids.get(card.deck, card.deck) for card in table.cards}
        self._deck_cards: dict[str, list[str]] = {}
        for card in table.cards:
            self._deck_cards.setdefault(self._deck_of_card[card.id], []).append(card.id)
        self.deck_rank = clustering.preset_rank(self.model.deck_links, DECK_PART)
        self.between = self.model.hidden  # the cards T1 placed by a margin below its threshold
        # The scripted topics as layouts without positions: the membership of the map without T
        self.scripted: dict[Key, TopicLayout] = {}
        self.scripted_by_deck: dict[str, list[TopicLayout]] = {}
        for seed in self.topic_model.seeds:
            deck_id = self.model.deck_ids.get(seed.deck, seed.deck)
            ids = [card_id for card_id in seed.card_ids if card_id in self._row_of]
            placed = TopicLayout(seed.deck, deck_id, seed.topic_id, seed.name, ids, {}, [], math.nan, math.nan)
            self.scripted[(deck_id, seed.topic_id)] = placed
            self.scripted_by_deck.setdefault(deck_id, []).append(placed)
        self._scripted_links: dict[Key, list[tuple[str, str]]] = {}
        margins = list(self.topic_model.margin_of.values())
        self.margin_scale = float(np.percentile(margins, MARGIN_PERCENTILE)) if margins else 1.0
        # P1 on the app's index and T1, as the serving process fits it; its fallback on the scripted topics on first use
        self.prompt = PromptModel(self.model.index, self.topic_model, table.cards)
        self._scripted_prompt: Optional[PromptModel] = None
        chosen = {"P1": variant_label(self.model.index.vectoriser), "D2": d2_label()}
        if s2_label() is not None:
            chosen["S2"] = s2_label()
        self.results = Results(results_dir, course_note(table.deck_ids), self.name_ids(), chosen=chosen)
        self.glossary = read_glossary()

    def name_ids(self) -> dict[str, str]:
        """Every deck's and topic's name of the table with what shows instead: the deck's id, the topic's label."""
        names = {deck.name: deck.deck_id for deck in self.table.decks if deck.name}
        for seed in self.topic_model.seeds:
            if seed.name:
                names.setdefault(seed.name, self.label_of(seed.topic_id, seed.name))
        return names

    # --- the layers --------------------------------------------------------------------------------------------

    @staticmethod
    def layer_states(layers: Optional[Iterable[str]]) -> dict[str, str]:
        """Every checked layer's state from the checked set alone: on, on its fallback, or not functioning (ON,
        FALLBACK, OFF): S and D need nothing; T needs S; C and P need S, and show their fallback without T."""
        checked = set(layers or ())
        states: dict[str, str] = {}
        if "S" in checked:
            states["S"] = ON
        if "D" in checked:
            states["D"] = ON
        if "T" in checked:
            states["T"] = ON if "S" in checked else OFF
        for letter in ("C", "P"):
            if letter in checked:
                states[letter] = OFF if "S" not in checked else (ON if states.get("T") == ON else FALLBACK)
        return states

    def in_force(self, state: dict, letter: str) -> Optional[str]:
        """The layer's state under the view's checked set; None unchecked."""
        return self.layer_states(state.get("layers")).get(letter)

    def uses_model(self, state: dict) -> bool:
        """Whether the membership in force is T1's (T's layer on)."""
        return self.in_force(state, "T") == ON

    def positions_of(self, state: dict) -> str:
        """The positions' channel: the model's with C on (MODEL), the scripted topics laid out with C on its
        fallback (SCRIPTED), random otherwise (RANDOM)."""
        return {ON: MODEL, FALLBACK: SCRIPTED}.get(self.in_force(state, "C"), RANDOM)

    @staticmethod
    def level_of(state: dict) -> Optional[str]:
        """The open generation: the state's, or the newest of the open tab's; None with no tab open."""
        generations = GENERATIONS.get(state.get("panel"), ())
        if not generations:
            return None
        return state.get("level") if state.get("level") in generations else generations[-1]

    def layer_level(self, state: dict, letter: str) -> str:
        """The generation a layer draws: the one open in its tab, else the app's (S2, D1; S1 without the encoder)."""
        return self.level_of(state) if state.get("panel") == letter else self.app_levels[letter]

    def similarity_of(self, state: dict) -> Optional[similar.SimilarityIndex]:
        """S's index: the generation's open in S's tab, else the app's; S2 without the encoder falls back on S1's,
        the app's index."""
        return self.indexes.get(self.layer_level(state, "S")) or self.indexes["S1"]

    def difficulty_of(self, state: dict) -> Optional[DifficultyReport]:
        """D's report: the generation's open in D's tab, else D1's; None for D2 without cached tags."""
        return self.reports.get(self.layer_level(state, "D"))

    def prompt_of(self, state: dict) -> Optional[PromptModel]:
        """P's model: P1 on T1's membership, or on the scripted topics with P's layer on its fallback (fitted on
        first use and kept); None with P's layer not functioning (S off)."""
        in_force = self.in_force(state, "P")
        if in_force == OFF:
            return None
        if in_force == FALLBACK:
            if self._scripted_prompt is None:
                members = {seed.key: [card_id for card_id in seed.card_ids if card_id in self._row_of] for seed in self.topic_model.seeds}
                stand_in = SimpleNamespace(seeds=self.topic_model.seeds, members=members)
                self._scripted_prompt = PromptModel(self.model.index, stand_in, self.table.cards)
            return self._scripted_prompt
        return self.prompt

    def other_report(self, report: DifficultyReport) -> Optional[DifficultyReport]:
        """The other D generation's report (D2's next to D1's and the reverse); None without it."""
        return self.reports.get("D2" if report.model == "D1" else "D1")

    def disagreements(self, ids: Iterable[str], report: DifficultyReport) -> dict[str, str]:
        """The quizzes among `ids` the generation predicted at another level than the other generation did:
        card id -> the other generation's level; empty without the other generation's report."""
        other = self.other_report(report)
        if other is None:
            return {}
        return {
            card_id: other.predictions[card_id] for card_id in ids
            if card_id in report.predictions and card_id in other.predictions and other.predictions[card_id] != report.predictions[card_id]
        }

    # --- the membership and the positions ----------------------------------------------------------------------

    def topics(self, deck_id: Optional[str], state: Optional[dict] = None) -> list[TopicLayout]:
        """The deck's topics in file order: the model's, or the scripted ones without T in force."""
        if not deck_id:
            return []
        if state is None or self.uses_model(state):
            return self.model.topics_of_deck(deck_id)
        return list(self.scripted_by_deck.get(deck_id, []))

    def topic(self, deck_id: Optional[str], topic_id: Optional[str]) -> Optional[TopicLayout]:
        """The model's layout of a topic."""
        return self.model.topics.get((deck_id, topic_id))

    def entered(self, state: dict) -> Optional[TopicLayout]:
        """The entered topic under the membership in force."""
        key = (state.get("deck"), state.get("topic"))
        return (self.model.topics if self.uses_model(state) else self.scripted).get(key)

    def topic_of_dot(self, deck_id: Optional[str], dot: Optional[str], state: Optional[dict] = None) -> Optional[TopicLayout]:
        return next((topic for topic in self.topics(deck_id, state) if self.topic_dot(topic) == dot), None)

    @staticmethod
    def topic_dot(topic: TopicLayout) -> str:
        """A topic's dot id on the map: the deck id and the topic id."""
        return f"{topic.deck_id}:{topic.topic_id}"

    @staticmethod
    def label_of(topic_id: str, name: str) -> str:
        """A topic's label in place of its name: the topic id, then a hash of the name (blake2s, so the same on
        every run), made at runtime and stored nowhere."""
        digest = hashlib.blake2s(name.encode("utf-8"), digest_size=TOPIC_HASH_BYTES).hexdigest()
        return f"{topic_id}-{digest}"

    @classmethod
    def topic_label(cls, topic: TopicLayout) -> str:
        return cls.label_of(topic.topic_id, topic.name)

    def label_in_deck(self, deck_id: Optional[str], topic_id: str) -> str:
        """A topic of the deck by its label; a dash for none."""
        placed = self.model.topics.get((deck_id, topic_id))
        return self.topic_label(placed) if placed is not None else "–"

    @staticmethod
    def random_points(ids, seed: int = RANDOM_SEED) -> dict[str, Point]:
        """The ids at random on the sphere: the evaluation's draw over the sorted ids, so the same ids get the same
        points whatever layers are checked."""
        ordered = sorted(ids)
        return {dot: tuple(float(value) for value in point) for dot, point in zip(ordered, random_directions(len(ordered), seed))}

    def deck_points(self, state: Optional[dict] = None) -> dict[str, Point]:
        """The decks of the table the model placed, in deck id order: at the model's positions with C drawing (on
        or on its fallback: the decks' positions read no topic), or at random."""
        ids = [deck.deck_id for deck in self.table.decks if deck.deck_id in self.model.deck_points]
        if state is None or self.positions_of(state) != RANDOM:
            return {deck_id: self.model.deck_points[deck_id] for deck_id in ids}
        return self.random_points(ids)

    def topic_points(self, deck_id: Optional[str], state: Optional[dict] = None) -> dict[str, Point]:
        """The deck's topics at random on the sphere: the model has no topic positions yet."""
        return self.random_points([self.topic_dot(topic) for topic in self.topics(deck_id, state)])

    def scripted_layout(self, topic: TopicLayout) -> TopicLayout:
        """C's fallback: the scripted topic laid out and clustered on the app's vectors (CourseModel.layout_of,
        kept after the first call)."""
        return self.model.layout_of(topic.deck_id, topic.topic_id, topic.ids, topic.name)

    def card_points(self, topic: TopicLayout, state: Optional[dict] = None) -> dict[str, Point]:
        """The topic's cards: at the model's directions with C on (a card without one, in a topic of a single card,
        at an even spread; with T on the moved-out cards too, at the mean direction of their most similar members),
        at the scripted topic's layout with C on its fallback, or at random."""
        positions = MODEL if state is None else self.positions_of(state)
        if positions == RANDOM:
            return self.random_points(topic.ids)
        laid_out = topic.positions if positions == MODEL else self.scripted_layout(topic).positions
        points = {card_id: laid_out[card_id] for card_id in topic.ids if card_id in laid_out}
        missing = [card_id for card_id in topic.ids if card_id not in points]
        outs = self.moved_out(topic) if state is not None and self.uses_model(state) else []
        for card_id in outs:
            point = self._mean_direction(card_id, points)
            if point is None:
                missing.append(card_id)
            else:
                points[card_id] = point
        for card_id, point in zip(missing, fibonacci_sphere(len(missing))):
            points[card_id] = tuple(float(value) for value in point)
        return {card_id: points[card_id] for card_id in list(topic.ids) + outs}

    def _mean_direction(self, card_id: str, points: dict[str, Point]) -> Optional[Point]:
        """The normalised mean of the points of the card's SIMILAR_MEMBERS most similar cards among `points` (the
        app's cosine); None without a placed member or with a zero mean."""
        members = list(points)
        if not members or card_id not in self._row_of:
            return None
        vectors = self._vectors(members)
        own = self._vectors([card_id])[0]
        order = np.argsort(-(vectors @ own))[:SIMILAR_MEMBERS]
        mean = np.mean([points[members[index]] for index in order], axis=0)
        length = float(np.linalg.norm(mean))
        if length < 1e-9:
            return None
        return tuple(float(value) for value in mean / length)

    def _vectors(self, ids) -> np.ndarray:
        return as_dense(self.model.index.vectors[[self._row_of[card_id] for card_id in ids]])

    # --- T1's moves and markers --------------------------------------------------------------------------------

    def moved_in(self, topic: TopicLayout) -> list[str]:
        """The topic's members under T1 that the script put into another topic of the deck."""
        scripted = self.topic_model.scripted_of
        return [card_id for card_id in topic.ids if scripted.get(card_id, topic.topic_id) != topic.topic_id]

    def moved_out(self, topic: TopicLayout) -> list[str]:
        """The cards scripted into the topic that T1 placed in another topic of the deck."""
        placed = self.scripted.get((topic.deck_id, topic.topic_id))
        if placed is None:
            return []
        return [card_id for card_id in placed.ids if self.topic_model.topic_of.get(card_id) != topic.topic_id]

    def between_share(self, topic: TopicLayout) -> float:
        """The share of the topic's cards below the threshold."""
        return len(self.between.intersection(topic.ids)) / len(topic.ids) if topic.ids else 0.0

    def t1_markers(self, topic: TopicLayout, points: dict[str, Point]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """With T: every drawn card's colour (its margin on the teal scale, the diamonds' colours), symbol (a cross
        below the threshold, the diamonds of the moved cards; a diamond before the cross) and hover (its id and
        margin)."""
        ins, outs = set(self.moved_in(topic)), set(self.moved_out(topic))
        colors: dict[str, str] = {}
        symbols: dict[str, str] = {}
        labels: dict[str, str] = {}
        for card_id in points:
            margin = self.topic_model.margin_of.get(card_id, math.nan)
            if card_id in outs:
                colors[card_id], symbols[card_id] = MOVED_OUT_COLOR, MOVED_OUT_SYMBOL
            elif card_id in ins:
                colors[card_id], symbols[card_id] = MOVED_IN_COLOR, MOVED_IN_SYMBOL
            else:
                colors[card_id] = margin_color(0.0 if math.isnan(margin) else margin, self.margin_scale)
                symbols[card_id] = CROSS_SYMBOL if card_id in self.between else "circle"
            labels[card_id] = f"{card_id} · margin {margin:.3f}" if not math.isnan(margin) else card_id
        return colors, symbols, labels

    def t1_tags(self, card_id: str, topic: TopicLayout) -> list[str]:
        """A selected card's tags with T: where it fits and by what margin with its runner-up ("fits 03 by 0.12 ·
        runner-up 05"), and for a moved card the topic it came from or went to, by id."""
        margin = self.topic_model.margin_of.get(card_id)
        second = self.topic_model.second_of.get(card_id, "")
        fits = f"fits {self.topic_model.topic_of.get(card_id, topic.topic_id)}"
        if margin is not None:
            fits += f" by {margin:.2f}"
        if second:
            fits += f" · runner-up {second}"
        tags = [fits]
        if card_id in self.moved_out(topic):
            tags.append(f"T1's topic {self.topic_model.topic_of.get(card_id, '–')}")
        elif card_id in self.moved_in(topic):
            tags.append(f"scripted into {self.topic_model.scripted_of.get(card_id, '–')}")
        return tags

    # --- S's lines and neighbours ---------------------------------------------------------------------------

    def neighbour_pairs(self, points: dict[str, Point], state: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """With S, the neighbour pairs among the drawn cards under S's index, each once: (the others, the selected
        card's)."""
        index = self.similarity_of(state)
        if index is None:
            return [], []
        selected = state.get("selected")
        pairs: list[tuple[str, str]] = []
        own: list[tuple[str, str]] = []
        seen: set[frozenset] = set()
        for card_id in points:
            for other, _score in index.neighbours.get(card_id, []):
                key = frozenset((card_id, other))
                if other in points and key not in seen:
                    seen.add(key)
                    (own if selected in key else pairs).append((card_id, other))
        return pairs, own

    def neighbours_of(self, card_id: str, state: dict) -> list[tuple[str, float, str]]:
        """The card's neighbours under S's index, best first: (its id, the score, where it lies: this topic, this
        deck or another deck)."""
        index = self.similarity_of(state)
        topic = self.entered(state)
        here = set(topic.ids) if topic is not None else set()
        rows = []
        for other, score in index.neighbours.get(card_id, []) if index is not None else []:
            if other in here:
                where = HERE
            elif self._deck_of_card.get(other) == state.get("deck"):
                where = THIS_DECK
            else:
                where = ELSEWHERE
            rows.append((other, score, where))
        return rows

    def neighbour_counts(self, card_id: str, state: dict) -> str:
        """The tag strip's tag with S: how many neighbours the card has by place ("3 here · 1 this deck · 1
        elsewhere"), or none."""
        places = [where for _other, _score, where in self.neighbours_of(card_id, state)]
        if not places:
            return "no neighbour above the minimum"
        return f"{places.count(HERE)} here · {places.count(THIS_DECK)} this deck · {places.count(ELSEWHERE)} elsewhere"

    # --- D's rings -------------------------------------------------------------------------------------------

    def d_rings(self, points: dict[str, Point], report: DifficultyReport) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """With D: every drawn card's ring colour (its authored level's, the predicted amber; none for a card not
        rated), class (layout.card_class) and hover part (a predicted quiz's level, and the other generation's
        where it differs; the map draws a ring behind those, layout.disagreement_trace)."""
        rings: dict[str, str] = {}
        classes: dict[str, str] = {}
        levels: dict[str, str] = {}
        other = self.other_report(report)
        differing = self.disagreements(points, report)
        for card_id in points:
            predicted = report.predictions.get(card_id)
            classes[card_id] = card_class(self.card_by_id[card_id], predicted is not None)
            if classes[card_id] in LEVEL_COLORS:
                rings[card_id] = LEVEL_COLORS[classes[card_id]]
            elif classes[card_id] == PREDICTED:
                rings[card_id] = PREDICTED_COLOR
                levels[card_id] = f"{report.model}: {predicted}"
                if card_id in differing:
                    levels[card_id] += f" ({other.model}: {differing[card_id]})"
        return rings, classes, levels

    # --- the connections and the trees -------------------------------------------------------------------------

    def deck_links(self) -> list[tuple[str, str]]:
        """The deck connections below the preset: the first merges, until a part would hold more than DECK_PART."""
        return list(self.model.deck_links[: self.deck_rank])

    def deck_tree(self):
        """(the merges, the deck ids of their rows) of the decks' clustering; None below two decks."""
        ids = [self.model.deck_ids[name] for name in self.model.deck_names]
        return (merges_of(self.model.deck_vectors), ids) if len(ids) > 1 else None

    def links_of(self, topic: TopicLayout, state: dict) -> list[tuple[str, str]]:
        """The topic's connections in merge order under the membership in force: the model's, or the clustering of
        the scripted topic's cards, made once."""
        if self.uses_model(state):
            return list(topic.links)
        key = (topic.deck_id, topic.topic_id)
        if key not in self._scripted_links:
            self._scripted_links[key] = clustering.merge_links(topic.ids, self._vectors(topic.ids)) if len(topic.ids) > 1 else []
        return list(self._scripted_links[key])

    def card_links(self, topic: Optional[TopicLayout], state: dict) -> list[tuple[str, str]]:
        """The topic's connections below the preset: the first merges, until a part would hold more than CARD_PART."""
        if topic is None:
            return []
        links = self.links_of(topic, state)
        return links[: clustering.preset_rank(links, CARD_PART)]

    def card_tree(self, topic: Optional[TopicLayout]):
        """(the merges, the card ids of their rows) of the topic's clustering under its membership; None below two
        cards."""
        if topic is None or len(topic.ids) < 2:
            return None
        return merges_of(self._vectors(topic.ids)), list(topic.ids)

    def topic_tree(self, deck_id: Optional[str], state: dict):
        """(the merges, the dot ids of their rows, the labels by dot id) of the deck's topics under the membership
        in force: the clustering of the topic vectors (CourseModel.topic_vectors, the mean of a topic's cards'
        unit rows), the tree the viewer's topic connections replay to; None below two topics with cards."""
        topics = self.topics(deck_id, state)
        names, vectors = self.model.topic_vectors(deck_id, [(topic.topic_id, topic.ids) for topic in topics])
        if len(names) < 2:
            return None
        by_id = {topic.topic_id: topic for topic in topics}
        ids = [self.topic_dot(by_id[name]) for name in names]
        return merges_of(vectors), ids, {self.topic_dot(by_id[name]): self.topic_label(by_id[name]) for name in names}

    # --- what a view shows -------------------------------------------------------------------------------------

    def dots(self, state: dict) -> list[str]:
        """The dot ids of the view in `state`."""
        if state["view"] == DECKS_VIEW:
            return list(self.deck_points(state))
        if state["view"] == TOPICS_VIEW:
            return list(self.topic_points(state["deck"], state))
        topic = self.entered(state)
        return list(self.card_points(topic, state)) if topic is not None else []

    def where(self, state: dict) -> str:
        """The breadcrumb under the top bar, by ids and without a topic's hash: the number of decks, "Deck 17",
        or "Deck 17 › Topic 02"."""
        if state["view"] == DECKS_VIEW:
            return f"{len(self.deck_points())} decks"
        if state["view"] == TOPICS_VIEW:
            return f"Deck {state['deck']}"
        return f"Deck {state['deck']} › Topic {state['topic']}"

    def scope(self, state: dict) -> str:
        """The view's part of the course by ids, as the section's headline names it: all decks, a deck, or a
        deck's topic."""
        if state["view"] == DECKS_VIEW:
            return "all decks"
        if state["view"] == TOPICS_VIEW:
            return f"deck {state['deck']}"
        return f"deck {state['deck']} › topic {state['topic']}"

    def target_dot(self, state: dict) -> Optional[str]:
        """P1's target of the last request as a dot of the view (the card in its topic's cards view, the topic in
        its deck's topics view, the deck in the decks view); None without one in the view."""
        asked = state.get("prompt") if isinstance(state.get("prompt"), dict) else {}
        target = asked.get("target")
        if not target:
            return None
        deck_id, topic_id, card = (list(target) + [None])[:3]
        if state["view"] == DECKS_VIEW:
            return deck_id
        if state.get("deck") != deck_id:
            return None
        if state["view"] == TOPICS_VIEW:
            return f"{deck_id}:{topic_id}"
        return card if state.get("topic") == topic_id else None

    # --- the tag strip ---------------------------------------------------------------------------------------

    def tag_strip(self, state: dict) -> tuple[list, str]:
        """The selected dot's title and tags, and the strip's class: a deck's or a topic's counts with Open; a
        card's stack, its level as a coloured tag (authored, or predicted by D's generation, with the other
        generation's where they differ), "between two topics" only when true, with S's layer in force its
        neighbours by place, with T's where it fits by what margin; hidden with nothing selected."""
        view, selected = state["view"], state.get("selected")
        if view == DECKS_VIEW and selected in self.deck_by_id:
            topics = self.topics(selected, state)
            tags = [(f"{len(topics)} topics", ""), (f"{sum(len(topic.ids) for topic in topics)} cards", "")]
            return tag_strip_body(f"Deck {selected}", tags), tag_class(True, True)
        if view == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected, state)
            if topic is not None:
                tags = [(f"{count} {label}", "") for label, count in topic_counts_of(topic, self.card_by_id)]
                if self._rest(topic):
                    tags.insert(0, ("rest topic", ""))
                if not topic.ids:
                    tags.append(("no cards: the model moved every card elsewhere", ""))
                return tag_strip_body(f"Topic {topic.topic_id}", tags), tag_class(True, True)
        topic = self.entered(state)
        if view == CARDS_VIEW and selected in self.card_by_id and topic is not None:
            card = self.card_by_id[selected]
            tags: list[tuple[str, str]] = [(STACK_NAMES[card.stack], "")]
            report = self.difficulty_of(state) if self.in_force(state, "D") else self.difficulty
            if card.difficulty:
                tags.append((card.difficulty, f"level {card.difficulty}"))
            elif report is not None and selected in report.predictions:
                text = f"{report.predictions[selected]} · predicted by {report.model}"
                if selected in self.disagreements([selected], report):
                    text += f" · {self.other_report(report).model}: {self.disagreements([selected], report)[selected]}"
                tags.append((text, "level predicted"))
            if selected in self.between:
                tags.append(("between two topics", "t"))
            if self.in_force(state, "S") == ON:
                tags.append((self.neighbour_counts(selected, state), "s"))
            if self.uses_model(state) and selected in self.topic_model.topic_of:
                tags += [(text, "t") for text in self.t1_tags(selected, topic)]
            return tag_strip_body(f"Card {selected}", tags), tag_class(True, False)
        return [], tag_class(False, False)

    # --- the figure slot and Details -------------------------------------------------------------------------

    def dendrogram(self, state: dict) -> list:
        """The dendrogram of the view's membership with the preset's line labelled on the plot, its leaves by id
        (a topic by its id, the hover with its label), filling the slot; or why there is none."""
        view = state["view"]
        labels: dict = {}
        hovers: dict = {}
        if view == DECKS_VIEW:
            tree = self.deck_tree()
            kept = self.deck_rank
            cut = CUT_LABELS[view].format(size=DECK_PART)
        elif view == TOPICS_VIEW:
            topic_tree = self.topic_tree(state["deck"], state)
            tree = topic_tree[:2] if topic_tree is not None else None
            hovers = topic_tree[2] if topic_tree is not None else {}
            labels = {dot: dot.split(":")[1] for dot in hovers}  # the topic id without the hash
            kept = len(tree[1]) - 1 if tree is not None else None  # no preset: the full tree
            cut = CUT_LABELS[view]
        else:
            topic = self.entered(state)
            tree = self.card_tree(topic)
            kept = len(self.card_links(topic, state))
            cut = CUT_LABELS[view].format(size=CARD_PART)
        if tree is None:
            return [empty_slot(NO_LINKS[view])]
        merges, ids = tree
        return [graph(dendrogram_figure(merges, ids, labels, kept, hovers or None, cut), "dendrogram-graph")]

    def figure_slot(self, state: dict) -> list:
        """The slot's one figure by the open tab: none, C or P the dendrogram; S the selected card's neighbours,
        the dendrogram until a card is selected; T the histogram of the view's margins; D the generation's
        confusion matrix of the view's part (D2 without cached tags: why not; a part without metrics: its note)."""
        panel = state.get("panel")
        selected = state.get("selected")
        if panel == "S" and state["view"] == CARDS_VIEW and selected in self.card_by_id:
            rows = self.neighbours_of(selected, state)
            if self.indexes.get(self.level_of(state)) is None:
                return [empty_slot(NO_ENCODER)]
            return [graph(neighbours_figure(rows), "neighbours-graph")] if rows else [empty_slot("No neighbour above the minimum score.")]
        if panel == "T":
            margins = self.margins(self.view_ids(state))
            return [graph(margin_histogram(margins, self.topic_model.margin), "margin-histogram")] if margins else [empty_slot(NO_CARDS)]
        if panel == "D":
            report = self.reports.get(self.level_of(state))
            if report is None:
                return [empty_slot(NO_TAGS)]
            part, _scope = self.metrics_report(state, report)
            if not part.measured:
                return [empty_slot(part.note or "No difficulty model.")]
            return [graph(confusion_figure(part), "difficulty-confusion")]
        return self.dendrogram(state)

    def slot_shows_dendrogram(self, state: dict) -> bool:
        """Whether the slot's figure is the dendrogram (with no tab, C or P open, or S without a card selected)."""
        panel = state.get("panel")
        return panel in (None, "C", "P") or (panel == "S" and not (state["view"] == CARDS_VIEW and state.get("selected") in self.card_by_id))

    def details(self, state: dict) -> list:
        """Details' content in the slot's place: the open generation's description, its live measures, its tables of
        its newest run with the one open row's description (the glossary), or why there is none (no file: how to
        make it; no rows of it); then what moved in: the dendrogram while the slot shows another figure, the view's
        hint lines."""
        level = self.level_of(state)
        if level is None:
            return []
        live = {"S": self._live_similarity, "C": self._live_course, "T": self._live_topics, "D": self._live_difficulty,
                "P": self._live_prompt}[level[0]](level, state)
        deck_id = state.get("deck") if level in DECK_TABLE_LEVELS and state["view"] != DECKS_VIEW else None
        tables = self.results.tables(level, deck_id)
        hint = self.results.hint(LEVEL_FILES[level]) if tables is None else self.results.no_rows(level)
        moved: list = []
        if not self.slot_shows_dendrogram(state):
            moved += [html.Div("The view's tree", className="card-label metrics-heading")] + self.dendrogram(state)
        moved.append(html.Div(OVERVIEW_HINT[state["view"]], className="card-hint"))
        return [level_section(DESCRIPTIONS[level], live, tables, hint, level, state.get("measure"), self.glossary, moved)]

    # --- the section -----------------------------------------------------------------------------------------

    def section(self, state: dict) -> tuple[list, list]:
        """The section's head and its live part: with no tab open one line on the five models; else the headline,
        the generation's two numbers (app.headlines) with the chosen variant in plain words, or the hint
        without the results; the live part T's line of the view next to the course, P's answer or its hint."""
        level = self.level_of(state)
        if level is None:
            return no_tab_line(), []
        numbers = headlines(self.results, level)
        hint = self.results.hint(LEVEL_FILES[level]) if self.results.run(level) is None else self.results.no_rows(level)
        head = section_head(level, numbers, chosen_label(self.results, level), hint, self.scope(state))
        live: list = []
        if level == "T1":
            live = live_line(self.topics_line(state))
        elif level == "P1":
            live = self.prompt_live(state)
        return head, live

    def topics_line(self, state: dict) -> str:
        """T's line of the view next to the course: "this deck: 70 % agree, course 64 %"."""
        ids = self.view_ids(state)
        course = shown(self.topic_model.agreement_of(list(self.topic_model.topic_of)), True)
        if state["view"] == DECKS_VIEW or not ids:
            return f"course: {course} agree with the script"
        here = {TOPICS_VIEW: "this deck", CARDS_VIEW: "this topic"}[state["view"]]
        return f"{here}: {shown(self.topic_model.agreement_of(ids), True)} agree, course {course}"

    def prompt_entries(self, answer) -> list[tuple[str, str, str]]:
        """P1's answer as entries: (the target's ids, its label by ids, every route's score)."""
        entries = []
        for target in answer.targets:
            deck_id = self.model.deck_ids.get(target.key[0], target.key[0])
            card = target.card
            label = f"deck {deck_id} › {self.label_in_deck(deck_id, target.key[1])}" + (f" · card {card}" if card else "")
            scores = " · ".join(f"{route} {target.scores[route]:.2f}" for route in (CARD, CENTROID, TEXT)) + f" · mean {target.mean:.2f}"
            entries.append((f"{deck_id}:{target.key[1]}" + (f":{card}" if card else ""), label, scores))
        return entries

    def prompt_live(self, state: dict, scores: bool = False) -> list:
        """P's answer to the last request on top: its decision line with the time and the three nearest topics by
        label, each a click away (in Details with every route's score); or how to ask, or that P does not
        function."""
        model = self.prompt_of(state)
        if model is None:
            return [html.P(PROMPT_OFF, className="empty-state")]
        asked = state.get("prompt") if isinstance(state.get("prompt"), dict) else {}
        if not asked.get("text"):
            return [html.P(PROMPT_HINT, className="empty-state")]
        answer = model.find(asked["text"])
        return prompt_answer(asked["text"], f"{answer.message} · {float(asked.get('ms') or 0):.1f} ms", self.prompt_entries(answer), scores)

    # --- the live measures, in Details -----------------------------------------------------------------------

    def metrics_report(self, state: dict, report: Optional[DifficultyReport] = None) -> tuple[DifficultyReport, str]:
        """The report's part of the view (D1's by default; the course, the entered deck, or the entered topic under
        its membership) with the scope it names."""
        report = report if report is not None else self.difficulty
        if state["view"] == DECKS_VIEW:
            return report, self.scope(state)
        deck = self.deck_by_id.get(state["deck"])
        part = report.for_deck(deck.name if deck else "")
        topic = self.entered(state)
        if state["view"] == TOPICS_VIEW or topic is None:
            return part, f"deck {state['deck']}"
        return part.for_topic(topic.name, topic.ids), self.scope(state)

    def metrics(self, state: dict, report: Optional[DifficultyReport] = None):
        """The report's scores of the view in Details, D1's by default (the matrix is the slot's)."""
        return metrics_panel(*self.metrics_report(state, report))

    def measures(self, state: dict) -> tuple[str, list]:
        """C1's readout: C1's own measures of the view, as (what they cover, [(heading, [(label, value, hint)])]),
        over the course, the entered deck or the entered topic (the model's layouts, whatever is checked)."""
        if state["view"] == DECKS_VIEW:
            topics = list(self.model.topics.values())
            decks = [
                ("trustworthiness", number(self.model.deck_trustworthiness), TRUSTWORTHINESS),
                (f"top-{DECK_NEIGHBOURS} overlap", number(self.model.deck_overlap), OVERLAP),
                ("connections kept", f"{self.deck_rank} of {len(self.model.deck_links)}", f"Below the cut-off: parts of at most {DECK_PART} decks."),
            ]
            scope = f"All {len(self.deck_points())} decks, {sum(len(topic.ids) for topic in topics):,} cards"
            return scope, [
                (f"Decks (k = {DECK_NEIGHBOURS})", decks),
                ("Topics (T1)", self._topic_measures(topics)),
                ("Cards on their topic's sphere", self._layout_measures(topics)),
            ]
        if state["view"] == TOPICS_VIEW:
            topics = self.topics(state["deck"])
            random = ("dot positions", "none yet", "The model places no topic: the dots stand at random on the sphere.")
            return f"This deck: {len(topics)} topics, {sum(len(topic.ids) for topic in topics)} cards", [
                ("Topics (T1)", self._topic_measures(topics) + [random]),
                ("Cards on their topic's sphere", self._layout_measures(topics)),
            ]
        topic = self.topic(state["deck"], state["topic"])
        if topic is None:
            return "", []
        count = len(topic.ids)
        if count < 2:
            sphere = [("layout", "none", "A topic of fewer than two cards gets no layout and no connections.")]
        else:
            k = min(CARD_NEIGHBOURS, count - 1)
            sphere = [
                ("neighbours k", f"{k} of {count - 1}", f"k = min({CARD_NEIGHBOURS}, n − 1): the most similar cards each card is pulled toward and measured against."),
                ("top-k overlap", "trivial" if k == count - 1 else number(topic.overlap), OVERLAP + " With k = n − 1 every other card is a neighbour: 1 by construction."),
                ("trustworthiness", number(topic.trustworthiness), TRUSTWORTHINESS + " Defined for more than 2k cards."),
                ("connections kept", f"{len(topic.links[: clustering.preset_rank(topic.links, CARD_PART)])} of {len(topic.links)}", f"Below the cut-off: parts of at most {CARD_PART} cards."),
            ]
        between = ("between two topics", f"{len(self.between.intersection(topic.ids))} of {count}", BETWEEN.format(margin=self.topic_model.margin))
        return f"This topic: {count} cards", [("Cards on the sphere", sphere), ("Topic (T1)", [between])]

    def _topic_measures(self, topics: list[TopicLayout]) -> list[tuple[str, str, str]]:
        """T1 over these topics: how many keep cards, and how many of their cards it placed between two topics."""
        ids = [card_id for topic in topics for card_id in topic.ids]
        return [
            ("with cards", f"{sum(bool(topic.ids) for topic in topics)} of {len(topics)}", "A topic T1 left without cards keeps its dot."),
            ("between two topics", _share(len(self.between.intersection(ids)), len(ids)), BETWEEN.format(margin=self.topic_model.margin)),
        ]

    @staticmethod
    def _layout_measures(topics: list[TopicLayout]) -> list[tuple[str, str, str]]:
        """The card layouts of these topics: the mean overlap where k leaves cards out, the mean trustworthiness where
        it is defined."""
        overlaps = [topic.overlap for topic in topics if len(topic.ids) - 1 > CARD_NEIGHBOURS]
        trusted = [topic.trustworthiness for topic in topics if not math.isnan(topic.trustworthiness)]
        return [("top-k overlap", _mean(overlaps), MEAN_OVERLAP), ("trustworthiness", _mean(trusted), MEAN_TRUSTWORTHINESS)]

    def _rest(self, topic: TopicLayout) -> bool:
        return any(seed.rest for seed in self.topic_model.seeds if seed.topic_id == topic.topic_id and seed.deck == topic.deck)

    def margins(self, ids) -> list[float]:
        """T1's margins of these cards, where it has one."""
        return [self.topic_model.margin_of[card_id] for card_id in ids if card_id in self.topic_model.margin_of]

    def view_ids(self, state: dict) -> list[str]:
        """The cards of the view under T1: the course's, the entered deck's, or the entered topic's."""
        if state["view"] == DECKS_VIEW:
            return list(self.topic_model.topic_of)
        if state["view"] == TOPICS_VIEW:
            return [card_id for topic in self.model.topics_of_deck(state["deck"]) for card_id in topic.ids]
        topic = self.topic(state["deck"], state["topic"])
        return list(topic.ids) if topic is not None else []

    def _live_similarity(self, level: str, state: dict) -> list:
        """S1's or S2's live measures over the course: its vectoriser, whether the app runs it, its vocabulary, the
        encoder's dimensions and cache, the hybrid's minima and weights, its minimum score and coverage (the
        selected card's neighbours are the slot's)."""
        index = self.indexes.get(level)
        if index is None:
            return [html.P(NO_ENCODER, className="empty-state")]
        vectoriser = index.vectoriser
        tfidf = vectoriser.first if isinstance(vectoriser, embeddings.HybridVectoriser) else vectoriser
        encoder = vectoriser.second if isinstance(vectoriser, embeddings.HybridVectoriser) else vectoriser
        vocabulary = getattr(tfidf, "vocabulary_", None)
        with_neighbour = sum(bool(index.neighbours.get(card_id)) for card_id in index.ids)
        distinct = {other for items in index.neighbours.values() for other, _score in items}
        rows = [
            ("vectoriser", embeddings.describe(vectoriser), "S1's TF-IDF, or S2's sentence encoder or its hybrid with TF-IDF."),
            ("the app's index", "yes" if index is self.model.index else "no", "Whether the app's neighbours, positions and topics come from this generation's vectors."),
            ("vocabulary", f"{len(vocabulary):,}" if vocabulary is not None else "–", "The terms of the fitted TF-IDF: unigrams and bigrams in at least two cards."),
        ]
        if isinstance(encoder, embeddings.SentenceEncoder):
            rows.append(("embedding dimensions", f"{encoder.dimensions:,}" if encoder.dimensions else "–", "The length of the encoder's vectors."))
            rows.append(("embeddings", "from the cache" if encoder.cached else f"encoded in {encoder.seconds:.1f} s on {encoder.device}",
                         "Whether the start read the course's embeddings from the cache (a derived file keyed by the model and the texts) or encoded them, and on which device."))
            rows.append(("device", encoder.device or "–",
                         "Where the encoder runs, the requests' texts as well: cuda (the GPU) where torch sees one, else cpu; APP_DEVICE chooses (auto, cpu, cuda)."))
        if isinstance(vectoriser, embeddings.HybridVectoriser):
            rows.append(("hybrid minima", " / ".join(number(value) for value in vectoriser.minima),
                         "Each half's own minimum: the 10th percentile of its fifth neighbour's cosine (TF-IDF, then the encoder)."))
            rows.append(("hybrid weights", " / ".join(f"{value:.2f}" for value in vectoriser.weights),
                         "The weight of each half's score in the hybrid's cosine, 1 / (1 − minimum): the sum of the two scores each normalised at its own minimum."))
        rows += [
            ("minimum score", number(index.min_score), "A neighbour below this cosine is dropped: the 10th percentile of the fifth neighbour's score, the index's own."),
            ("cards with a neighbour", _share(with_neighbour, len(index.ids)), "The cards with at least one neighbour above the minimum score."),
            ("distinct neighbours", f"{len(distinct):,}", "How many different cards appear as some card's neighbour."),
        ]
        return model_readout(f"All {len(index.ids):,} cards", [(f"{level} ({embeddings.describe(vectoriser)}, cosine)", rows)])

    def _live_course(self, level: str, state: dict) -> list:
        live = model_readout(*self.measures(state))
        if state["view"] == TOPICS_VIEW:
            live = live + [html.Div(TOPICS_NOTE, className="card-hint")]
        return live

    def _live_topics(self, level: str, state: dict) -> list:
        """T1's live measures of the view: the agreement, the cards between two topics, the moved cards (the
        selected card's margin is the tag strip's, the histogram the slot's)."""
        ids = self.view_ids(state)
        rows = [
            ("agreement with the script", number(self.topic_model.agreement_of(ids)) if ids else "–", "The share of the view's cards whose model topic is their scripted one."),
            ("between two topics", _share(len(self.between.intersection(ids)), len(ids)), BETWEEN.format(margin=self.topic_model.margin)),
        ]
        topic = self.topic(state["deck"], state["topic"]) if state["view"] == CARDS_VIEW else None
        if topic is not None:
            rows += [
                ("moved in", str(len(self.moved_in(topic))), "The topic's members under T1 that the script put elsewhere."),
                ("moved out", str(len(self.moved_out(topic))), "The cards scripted here that T1 placed elsewhere."),
            ]
        else:
            moved = sum(self.topic_model.topic_of[card_id] != self.topic_model.scripted_of.get(card_id) for card_id in ids)
            rows.append(("moved", _share(moved, len(ids)), "The view's cards whose model topic is not their scripted one."))
        scope = {DECKS_VIEW: f"All {len(ids):,} cards", TOPICS_VIEW: f"This deck: {len(ids)} cards", CARDS_VIEW: f"This topic: {len(ids)} cards"}[state["view"]]
        return model_readout(scope, [("T1 (nearest centroid)", rows)])

    def _live_difficulty(self, level: str, state: dict) -> list:
        report = self.reports.get(level)
        if report is None:
            return [html.P(NO_TAGS, className="empty-state")]
        return [self.metrics(state, report)]

    def _live_prompt(self, level: str, state: dict) -> list:
        """P1's part of Details: the answer with every route's score, then the floors over the self set."""
        floors = self.prompt.floors
        live = self.prompt_live(state, scores=True)
        rows = [
            ("vectoriser", embeddings.describe(self.prompt.index.vectoriser), "The app's index: a request's vector comes from its vectoriser."),
            ("routes", "+".join(DEFAULT_ROUTES), "The variant the app runs: the card route, until the choice rule names another."),
            ("card route floor", f"{floors.card:.3f}", "The 10th percentile of a question's first neighbour's cosine among the other cards."),
            ("centroid route floor", f"{floors.centroid:.3f}", "The 10th percentile of a question's cosine to its own topic's leave-one-out centroid."),
            ("text route floor", f"{floors.text:.3f}", "The 10th percentile of a question's cosine to its own topic's headings."),
            ("self set", f"{len(self.prompt.self_ids):,} questions", "Every card's question alone, the card held out: the floors' sample."),
        ]
        return live + model_readout(f"All {len(self.prompt.keys):,} topics", [("P1 (floors from the self set)", rows)])

    # --- the moves ---------------------------------------------------------------------------------------------

    def go_to(self, state: dict, deck_id: Optional[str], topic_id: Optional[str], card: Optional[str] = None) -> dict:
        """The view entered into a target as the viewer navigates: the topic's cards, the card selected where one is
        given (step unselects a card the view does not hold); an unknown topic moves nothing."""
        if (deck_id, topic_id) not in self.model.topics:
            return state
        return {**state, "view": CARDS_VIEW, "deck": deck_id, "topic": topic_id, "selected": card}

    def find(self, state: dict, text: Optional[str]) -> dict:
        """A request from P's box: P's model's answer (P1, or its fallback on the scripted topics), the request, its
        time and its target kept in the store (the section and the map read them), P's tab open, and the view
        navigated into the target where P1 navigates; "no match" moves nothing. An empty request, or P not
        functioning, changes nothing."""
        text = (text or "").strip()
        model = self.prompt_of(state)
        if not text or model is None:
            return state
        started = time.perf_counter()
        answer = model.find(text)
        seconds = time.perf_counter() - started
        if state.get("panel") != "P":
            state = {**state, "panel": "P", "level": None, "details": False}
        asked = {"text": text, "ms": round(1000 * seconds, 1), "target": None}
        if answer.navigate:
            deck_id = self.model.deck_ids.get(answer.topic[0], answer.topic[0])
            asked["target"] = [deck_id, answer.topic[1], answer.card]
            state = self.go_to(state, deck_id, answer.topic[1], answer.card)
        return {**state, "prompt": asked, "measure": None}

    def enter(self, state: dict) -> dict:
        """Open: a selected deck's topics view, a selected topic's cards view; nothing else."""
        selected = state.get("selected")
        if state["view"] == DECKS_VIEW and selected in self.deck_by_id:
            return {**state, "view": TOPICS_VIEW, "deck": selected, "topic": None, "selected": None}
        if state["view"] == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected, state)
            if topic is not None:
                return {**state, "view": CARDS_VIEW, "topic": topic.topic_id, "selected": None}
        return state

    def back(self, state: dict) -> dict:
        """Back: up a level, with the dot it came from selected; in the decks view the selection goes."""
        if state["view"] == CARDS_VIEW:
            topic = self.topic(state["deck"], state["topic"])
            return {**state, "view": TOPICS_VIEW, "topic": None, "selected": self.topic_dot(topic) if topic else None}
        if state["view"] == TOPICS_VIEW:
            return {**state, "view": DECKS_VIEW, "deck": None, "topic": None, "selected": state["deck"]}
        return {**state, "selected": None}

    def can_enter(self, state: dict) -> bool:
        return self.enter(state) is not state

    @staticmethod
    def _clicked(value, key: str) -> Optional[str]:
        """The `key` of the hidden input's value (a JSON of `measure`, `level` or `target` and `seq`,
        assets/panel_rows.js); None without it or for anything else."""
        try:
            parsed = json.loads(value) if isinstance(value, str) else None
        except ValueError:
            return None
        found = parsed.get(key) if isinstance(parsed, dict) else None
        return found if isinstance(found, str) and found else None

    @classmethod
    def measure_of(cls, value) -> Optional[str]:
        """The results row's key in the hidden input's value; None for the close box or anything else."""
        return cls._clicked(value, "measure")

    @classmethod
    def generation_of(cls, value) -> Optional[str]:
        """The generation whose tab was clicked, in the hidden input's value; None for a results row's click."""
        return cls._clicked(value, "level")

    @classmethod
    def target_of(cls, value) -> Optional[list[str]]:
        """P1's target whose entry was clicked, in the hidden input's value: [deck id, topic id] and the card where
        the entry shows one; None for anything else."""
        found = cls._clicked(value, "target")
        parts = found.split(":") if found else []
        return parts if len(parts) in (2, 3) else None

    def step(self, state: Optional[dict], trigger: Optional[str], click_data: Optional[dict] = None, measure=None,
             request: Optional[str] = None, layers: Optional[Sequence[str]] = None) -> dict:
        """The view after a trigger: a click on the map selects a dot of the view, Open or the tag strip's Open
        enters, Back goes up, a check box's change sets the layers, a tab opens its model with its newest
        generation (or closes it when open; either closes Details and an open description), the Details button
        opens or closes Details; through the hidden input (`measure`), a generation's tab opens that generation of
        the open tab (and closes Details and an open description), a results row's click opens its description (and
        closes the open one; the same row's or the close box's click closes it), a click on one of P1's targets goes
        to it; Go or Enter in P's box runs the `request` (find). A selected dot the view no longer holds is
        unselected."""
        state = {**HOME, **(state if isinstance(state, dict) else {})}
        if state["view"] not in (DECKS_VIEW, TOPICS_VIEW, CARDS_VIEW) or state["panel"] not in (None, *PANEL_LETTERS):
            state = dict(HOME)
        if state["level"] not in GENERATIONS.get(state["panel"], ()):
            state["level"] = None
        state["layers"] = [letter for letter in PANEL_LETTERS if letter in (state["layers"] or [])]
        state["details"] = bool(state["details"]) and state["panel"] is not None
        if trigger == "map":
            points = (click_data or {}).get("points") or []
            dot = points[0].get("customdata") if points and isinstance(points[0], dict) else None
            if dot in self.dots(state):
                state["selected"] = dot
        elif trigger == LAYERS_INPUT:
            state["layers"] = [letter for letter in PANEL_LETTERS if letter in (layers or [])]
        elif trigger in ENTER_TRIGGERS:
            state = self.enter(state)
        elif trigger in PROMPT_TRIGGERS:
            state = self.find(state, request)
        elif trigger == "back":
            state = self.back(state)
        elif isinstance(trigger, str) and trigger.startswith("tab-") and trigger[4:] in PANEL_LETTERS:
            state["panel"] = None if state["panel"] == trigger[4:] else trigger[4:]
            state["level"] = None
            state["measure"] = None
            state["details"] = False
        elif trigger == DETAILS_TRIGGER:
            state["details"] = not state["details"] and state["panel"] is not None
            state["measure"] = None
        elif trigger == ROW_INPUT:
            level = self.generation_of(measure)
            target = self.target_of(measure)
            if target is not None:
                state = self.go_to(state, target[0], target[1], target[2] if len(target) > 2 else None)
            elif level is not None:
                if level in GENERATIONS.get(state["panel"], ()) and level != self.level_of(state):
                    state["level"] = level
                    state["measure"] = None
                    state["details"] = False
            else:
                key = self.measure_of(measure)
                state["measure"] = None if key is None or key == state["measure"] else key
        if state.get("selected") is not None and state["selected"] not in self.dots(state):
            state["selected"] = None
        return state


def register_callback(app: Dash, page: Page):
    """The page's one callback; returns the function (for tests). Its outputs: the map, the view, the breadcrumb,
    Open's and Back's disabled state, the check boxes' options (their labels say the layers' states) and their
    value (the store's layers), the five tabs' classes, the generation tabs, the section's head and live part, the request box's class and its
    input's disabled state, the Details button's class, the tag strip's body and class, the figure slot. Its
    inputs: the map's click, the check boxes, Open, Back, the tag strip's Open, the five tabs, the Details button,
    Go and Enter of P's box, and the hidden input of a generation's, a results row's or a target's click; its states
    are the box's text and the view."""

    @app.callback(
        Output("map", "figure"),
        Output("view", "data"),
        Output("where", "children"),
        Output("open", "disabled"),
        Output("back", "disabled"),
        Output(LAYERS_INPUT, "options"),
        Output(LAYERS_INPUT, "value"),
        *[Output(f"tab-{letter}", "className") for letter in PANEL_LETTERS],
        Output("levels", "children"),
        Output("section-head", "children"),
        Output("section-live", "children"),
        Output("prompt-box", "className"),
        Output("prompt-text", "disabled"),
        Output(DETAILS_TRIGGER, "className"),
        Output("tag-body", "children"),
        Output("tag-strip", "className"),
        Output("figure-slot", "children"),
        Input("map", "clickData"),
        Input(LAYERS_INPUT, "value"),
        Input("open", "n_clicks"),
        Input("back", "n_clicks"),
        Input("tag-open", "n_clicks"),
        *[Input(f"tab-{letter}", "n_clicks") for letter in PANEL_LETTERS],
        Input(DETAILS_TRIGGER, "n_clicks"),
        Input("prompt-go", "n_clicks"),
        Input("prompt-text", "n_submit"),
        Input(ROW_INPUT, "value"),
        State("prompt-text", "value"),
        State("view", "data"),
    )
    def update(click_data, layers, *rest):
        from dash import ctx

        state = page.step(rest[-1], ctx.triggered_id, click_data, measure=rest[-3], request=rest[-2], layers=layers)
        panel, level = state.get("panel"), page.level_of(state)
        head, live = page.section(state)
        tags, tag_class_name = page.tag_strip(state)
        return (
            map_figure(page, state),
            state,
            page.where(state),
            not page.can_enter(state),
            state["view"] == DECKS_VIEW and not state.get("selected"),
            layer_options(page.layer_states(state["layers"])),
            list(state["layers"]),
            *[tab_class(letter == panel) for letter in PANEL_LETTERS],
            level_tabs(panel, level, page.app_levels.get(panel) if panel else None),
            head,
            live,
            prompt_class(panel == "P"),
            page.prompt_of(state) is None,
            details_class(state["details"], panel is not None),
            tags,
            tag_class_name,
            page.details(state) if state["details"] else page.figure_slot(state),
        )

    return update


def create_app(page: Page, viewer_url: Optional[str] = None) -> Dash:
    """The Dash app of the page, in the viewer's theme (assets/style.css), with its callback; with `viewer_url` the
    Sputnik link in the top bar (the viewer served next to the page)."""
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.layout = page_layout(HOME, viewer_url)
    register_callback(app, page)
    return app
