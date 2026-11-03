"""The standalone page's models and its one callback (Dash; nothing of the viewer).

Page fits the course model and D1 on the course table once, as the viewer's serving process does, reads the results
files of the offline evaluation, and answers what the three views and the four model panels show; everything shown
is the model's, nothing is stored.

The views: the decks on the sphere; a deck's topics (an entered deck) at random positions, since the model places
no topic (draft roadmap step 19); a topic's cards (an entered topic) on the surface of its sphere.

The panels (one per model letter, S, C, T and D; one expanded at a time, in the `view` store) decide what the map
shows of the models:

* every panel collapsed, or S or D expanded: the scripted topics' membership, random positions on the sphere
  (evaluation.random_directions over the view's sorted ids with a fixed seed, the same for every panel), no
  connections, centroid or axes; D colours the cards as the viewer does after Helper (a rated quiz by its level, a
  quiz D1 predicted in the predicted colour as a cross, a flashcard grey; Page.d1_markers), with the legend inside the
  map, and opens the difficulty window under the dendrogram's window with D1's confusion matrix and the shares of the
  levels of the view's part (Page.difficulty_window);
* C: T1's membership at C1's positions (the decks', the cards' directions), the connections below the preset, the
  centroid and the axes;
* T: the same positions, the centroid and the axes, no connections, and T1's markers: a card's margin on a teal
  scale, a cross below the threshold, a violet diamond for a card T1 moved in from another scripted topic, an open
  pink diamond for a card scripted here that T1 moved elsewhere, placed at the normalised mean direction of its most
  similar cards among the topic's members; an emptied topic as an open circle, a topic's dot grown with its share of
  cards below the threshold.

A change of membership keeps the view (T1 keeps the scripted topic ids); a selected card the new membership does
not hold is unselected. The dendrogram's window shows the tree of the view's membership with the preset's line in
every panel state (a scripted topic's by clustering.merge_links on its cards).

A panel's sections, one per level (S1, C1, T1, D1): the level's live measures of the view (C1's own measures,
Page.measures: the layouts' neighbour measures, the connections the preset keeps and T1's margins; D1's scores;
S1's vocabulary, minimum score and coverage; T1's agreement with the scripted topics, the moved cards and the
margins of the view) and the newest run's table of its results file (standalone.results), whose rows open their
description on a click, one at a time (the `measure` key of the `view` store: the row's key, or None).

The panels, the map's hover and the line under the top bar (Page.where) name the dots and the views by id
(Page.scope): no learning material, no card's text, no deck's description and no deck's name. A topic's name, the
one name that is no id, shows nowhere either: the map's hover and the line under the top bar give its label instead
(Page.topic_label: its topic id, then a hash of its name, made at runtime and stored nowhere), and a results cell
that names a deck or a topic shows the id or the label (results.Results.by_ids).

The callback (register_callback) keeps the view in the `view` store: a click on a dot selects it, Open or a click on
a selected deck's or topic's panel enters it, Back goes up a level with the dot it came from selected (in the decks
view it unselects); a panel's row expands it and collapses any other, its close box collapses it; a results row's
click (the hidden input `measure-click`, a JSON of the row's key and a sequence number) opens its description or
closes the open one. No keys, groups, editing, progress, quiz or image window.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from dash import Dash, Input, Output, State, dcc, html
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..course_table import CourseTable
from ..figures import dendrogram_figure, number
from ..models import clustering, embeddings, similar
from ..models.course_model import CARD_NEIGHBOURS, DECK_NEIGHBOURS, CourseModel, Point, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.evaluation import random_directions
from ..models.layout import as_dense, fibonacci_sphere
from .layout import (
    CARDS_VIEW, CROSS_SYMBOL, DECKS_VIEW, MOVED_IN_COLOR, MOVED_IN_SYMBOL, MOVED_OUT_COLOR, MOVED_OUT_SYMBOL, PANEL_LETTERS,
    TOPICS_NOTE, TOPICS_VIEW, box_class, card_class, card_panel, class_style, deck_panel, level_section, map_figure, margin_color,
    margin_histogram, metrics_panel, metrics_window, model_readout, overview_panel, page_layout, t1_legend, topic_panel,
    window_class,
)
from .results import Results, course_note, read_glossary

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"  # next to evaluate.py: modules/helper/ here, the capstone's root
TITLE = "Guided learning: the models"
DECK_PART = 5  # the decks view's preset: parts of at most five decks (as the viewer's Helper presets them)
CARD_PART = 4  # a topic's preset: parts of at most four cards
TOPIC_HASH_BYTES = 3  # a topic label's hash: six hex digits
RANDOM_SEED = 42  # the random positions of the scripted topics: one draw per view, the same for every panel
MARGIN_PERCENTILE = 95  # the top of the teal scale: the course's 95th percentile of the margins
SIMILAR_MEMBERS = 5  # a moved-out card sits at the mean direction of this many most similar members of its topic
MODEL_PANELS = ("C", "T")  # the panels whose map shows T1's membership at C1's positions
HOME = {"view": DECKS_VIEW, "deck": None, "topic": None, "selected": None, "panel": None, "measure": None}
ENTER_TRIGGERS = ("open", "dot-panel")
MEASURE_INPUT = "measure-click"  # the hidden input a results row's click sets (assets/measure_rows.js)
RESULTS_FILES = {"S": "results.csv", "C": "results.csv", "T": "topics.csv", "D": "difficulty.csv"}
Key = tuple[str, str]  # (deck id, topic id)
# The readout's hints, a row's tooltip
TRUSTWORTHINESS = "scikit-learn's trustworthiness of the map for the text vectors: 1 when none of a dot's k nearest on the map is a stranger by text."
OVERLAP = "The share of a dot's k most similar by text that are also among its k nearest on the map, averaged over the dots."
MEAN_OVERLAP = (
    f"The mean over the topics of more than {CARD_NEIGHBOURS + 1} cards: in a smaller one every other card is a neighbour "
    "(k = n − 1), so the overlap is 1 by construction."
)
MEAN_TRUSTWORTHINESS = "The mean over the topics of more than 2k cards, where scikit-learn defines it."
BETWEEN = "The cards whose topic T1 chose by a margin below {margin} (the best centroid's cosine minus the second best's)."
DESCRIPTIONS = {
    "S1": "One TF-IDF over the course's card texts, unit rows, the cosine as the similarity: every card's five nearest "
          "cards above a minimum score, and the vectors every other model reads. The map shows nothing of it yet: the "
          "scripted topics at random positions.",
    "S2": "A sentence encoder over the course's card texts, alone or hybrid with S1's TF-IDF (the two halves' unit rows "
          "side by side, each at its own minimum), unit rows, the cosine as the similarity: every card's five nearest "
          "cards above the index's own minimum score, and the vectors every other model reads. The map shows nothing of "
          "it yet: the scripted topics at random positions.",
    "C1": "The course model: the layout step on S1's vectors places the decks and, per topic, the cards on the surface of "
          "the topic's sphere; the clustering gives the connections and the cut-off preset. It builds on T1's topics.",
    "T1": "The topic model: the nearest of the deck's scripted topic centroids by cosine, leave-one-out, on S1's vectors; "
          "the scripted topics are its input, not the truth. The map shows its membership at C1's positions.",
    "D1": "The difficulty model: logistic regression on the question's statistics, labelled by the deck shares, predicts the "
          "level of the unrated quizzes; it reads neither C1 nor T1, so the map shows the scripted topics.",
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
    """The models fitted on the course table, the results files, and what the views and the panels show of them."""

    def __init__(
        self, table: CourseTable, model: Optional[CourseModel] = None, difficulty: Optional[DifficultyReport] = None,
        results_dir: Optional[Path] = None, vectoriser=None, min_score: Optional[float] = similar.MIN_SCORE,
    ) -> None:
        """`vectoriser` and `min_score` are the index's when the model is fitted here: the app's vectoriser at its
        own minimum from the launchers (standalone.main), TF-IDF at the fixed minimum by default."""
        self.table = table
        self.model = model if model is not None else CourseModel(table.cards, table.deck_ids, table.seeds, vectoriser, min_score=min_score)
        self.difficulty = difficulty if difficulty is not None else fit_difficulty(table.cards, self.model.index)
        self.topic_model = self.model.topic_model
        self.card_by_id = {card.id: card for card in table.cards}
        self.deck_by_id = {deck.deck_id: deck for deck in table.decks}
        self._row_of = {card_id: row for row, card_id in enumerate(self.model.index.ids)}
        self.deck_rank = clustering.preset_rank(self.model.deck_links, DECK_PART)
        self.between = self.model.hidden  # the cards T1 placed by a margin below its threshold
        # The scripted topics as layouts without positions: the membership of the map without the model
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
        self.results = Results(results_dir, course_note(table.deck_ids), self.name_ids())
        self.glossary = read_glossary()

    def name_ids(self) -> dict[str, str]:
        """Every deck's and topic's name of the table with what shows instead: the deck's id, the topic's label."""
        names = {deck.name: deck.deck_id for deck in self.table.decks if deck.name}
        for seed in self.topic_model.seeds:
            if seed.name:
                names.setdefault(seed.name, self.label_of(seed.topic_id, seed.name))
        return names

    # --- the membership and the positions per panel ----------------------------------------------------------

    @staticmethod
    def uses_model(state: dict) -> bool:
        """Whether the map shows T1's membership at C1's positions (C or T expanded)."""
        return state.get("panel") in MODEL_PANELS

    def topics(self, deck_id: Optional[str], state: Optional[dict] = None) -> list[TopicLayout]:
        """The deck's topics in file order: the model's, or the scripted ones without the model."""
        if not deck_id:
            return []
        if state is None or self.uses_model(state):
            return self.model.topics_of_deck(deck_id)
        return list(self.scripted_by_deck.get(deck_id, []))

    def topic(self, deck_id: Optional[str], topic_id: Optional[str]) -> Optional[TopicLayout]:
        """The model's layout of a topic."""
        return self.model.topics.get((deck_id, topic_id))

    def entered(self, state: dict) -> Optional[TopicLayout]:
        """The entered topic under the state's membership."""
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
        points whatever panel is expanded."""
        ordered = sorted(ids)
        return {dot: tuple(float(value) for value in point) for dot, point in zip(ordered, random_directions(len(ordered), seed))}

    def deck_points(self, state: Optional[dict] = None) -> dict[str, Point]:
        """The decks of the table the model placed, in deck id order: at the model's positions, or at random."""
        ids = [deck.deck_id for deck in self.table.decks if deck.deck_id in self.model.deck_points]
        if state is None or self.uses_model(state):
            return {deck_id: self.model.deck_points[deck_id] for deck_id in ids}
        return self.random_points(ids)

    def topic_points(self, deck_id: Optional[str], state: Optional[dict] = None) -> dict[str, Point]:
        """The deck's topics at random on the sphere: the model has no topic positions (draft roadmap step 19)."""
        return self.random_points([self.topic_dot(topic) for topic in self.topics(deck_id, state)])

    def card_points(self, topic: TopicLayout, state: Optional[dict] = None) -> dict[str, Point]:
        """The topic's cards: at the model's directions (a card without one, in a topic of a single card, at an even
        spread; with T the moved-out cards too, at the mean direction of their most similar members), or at random
        without the model."""
        if state is not None and not self.uses_model(state):
            return self.random_points(topic.ids)
        points = {card_id: topic.positions[card_id] for card_id in topic.ids if card_id in topic.positions}
        missing = [card_id for card_id in topic.ids if card_id not in points]
        outs = self.moved_out(topic) if state is not None and state.get("panel") == "T" else []
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
        """The normalised mean of the points of the card's SIMILAR_MEMBERS most similar cards among `points` (S1's
        cosine); None without a placed member or with a zero mean."""
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

    def t1_rows(self, card_id: str, topic: TopicLayout) -> list[tuple[str, str]]:
        """A selected card's rows with T: its margin, its runner-up topic by label, and for a moved card the topic
        it came from or went to."""
        margin = self.topic_model.margin_of.get(card_id)
        rows = [("margin", number(margin) if margin is not None else "–")]
        second = self.topic_model.second_of.get(card_id, "")
        rows.append(("runner-up topic", self.label_in_deck(topic.deck_id, second) if second else "–"))
        if card_id in self.moved_out(topic):
            rows.append(("T1's topic", self.label_in_deck(topic.deck_id, self.topic_model.topic_of.get(card_id, ""))))
        elif card_id in self.moved_in(topic):
            rows.append(("scripted topic", self.label_in_deck(topic.deck_id, self.topic_model.scripted_of.get(card_id, ""))))
        return rows

    # --- D1's markers ------------------------------------------------------------------------------------------

    def d1_markers(self, points: dict[str, Point]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """With D: every drawn card's colour, symbol and class (layout.card_class: its authored level, "predicted"
        for a quiz D1 predicted, a cross in the predicted colour, "flashcard" or "unrated quiz" grey)."""
        colors: dict[str, str] = {}
        symbols: dict[str, str] = {}
        classes: dict[str, str] = {}
        for card_id in points:
            classes[card_id] = card_class(self.card_by_id[card_id], card_id in self.difficulty.predictions)
            colors[card_id], symbols[card_id] = class_style(classes[card_id])
        return colors, symbols, classes

    # --- the connections and the trees -------------------------------------------------------------------------

    def deck_links(self) -> list[tuple[str, str]]:
        """The deck connections below the preset: the first merges, until a part would hold more than DECK_PART."""
        return list(self.model.deck_links[: self.deck_rank])

    def deck_tree(self):
        """(the merges, the deck ids of their rows) of the decks' clustering; None below two decks."""
        ids = [self.model.deck_ids[name] for name in self.model.deck_names]
        return (merges_of(self.model.deck_vectors), ids) if len(ids) > 1 else None

    def links_of(self, topic: TopicLayout, state: dict) -> list[tuple[str, str]]:
        """The topic's connections in merge order under the state's membership: the model's, or the clustering of
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
        """The line under the top bar, by ids: the number of decks, the entered deck's id, or the deck's id and the
        topic's label."""
        if state["view"] == DECKS_VIEW:
            return f"{len(self.deck_points())} decks"
        if state["view"] == TOPICS_VIEW:
            return f"deck {state['deck']}"
        topic = self.topic(state["deck"], state["topic"])
        return f"deck {state['deck']} › topic {self.topic_label(topic) if topic else ''}"

    def scope(self, state: dict) -> str:
        """The view's part of the course by ids, as the panels name it: all decks, a deck, or a deck's topic."""
        if state["view"] == DECKS_VIEW:
            return "all decks"
        if state["view"] == TOPICS_VIEW:
            return f"deck {state['deck']}"
        return f"deck {state['deck']} › topic {state['topic']}"

    def dot_panel(self, state: dict):
        """Window A: the selected dot's ids and counts under the view's membership, or the dendrogram of the view's
        membership with the preset's line, its leaves by id."""
        view, selected = state["view"], state.get("selected")
        if view == DECKS_VIEW and selected in self.deck_by_id:
            topics = self.topics(selected, state)
            return deck_panel(selected, len(topics), sum(len(topic.ids) for topic in topics))
        if view == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected, state)
            if topic is not None:
                return topic_panel(topic, self.card_by_id, rest=self._rest(topic))
        topic = self.entered(state)
        if view == CARDS_VIEW and selected in self.card_by_id and topic is not None:
            extra = self.t1_rows(selected, topic) if state.get("panel") == "T" else []
            return card_panel(
                self.card_by_id[selected], state["deck"], state["topic"], self.difficulty.predictions.get(selected),
                between=selected in self.between, extra=extra,
            )
        if view == DECKS_VIEW:
            tree = self.deck_tree()
            kept = self.deck_rank
            title, subtitle = "Constellations of decks", "The model's deck connections, cut into parts of at most five decks."
        elif view == TOPICS_VIEW:
            tree, kept, title, subtitle = None, None, self.scope(state), None
        else:
            tree = self.card_tree(topic)
            kept = len(self.card_links(topic, state))
            title = self.scope(state)
            subtitle = "The clustering of the topic's cards, cut into parts of at most four cards."
        figure = None
        if tree is not None:
            merges, ids = tree
            figure = dendrogram_figure(merges, ids, {}, kept)  # no labels: the leaves by id
        return overview_panel(title, figure, view, subtitle)

    def metrics_report(self, state: dict) -> tuple[DifficultyReport, str]:
        """D1's report of the view (the course, the entered deck, or the entered topic under its membership) with
        the scope it names."""
        if state["view"] == DECKS_VIEW:
            return self.difficulty, self.scope(state)
        deck = self.deck_by_id.get(state["deck"])
        part = self.difficulty.for_deck(deck.name if deck else "")
        topic = self.entered(state)
        if state["view"] == TOPICS_VIEW or topic is None:
            return part, f"deck {state['deck']}"
        return part.for_topic(topic.name, topic.ids), self.scope(state)

    def metrics(self, state: dict):
        """D1's scores of the view in the D panel (the matrix and the shares are the difficulty window's)."""
        return metrics_panel(*self.metrics_report(state))

    def difficulty_window(self, state: dict) -> tuple[list, str]:
        """The difficulty window's content and class: with D expanded, D1's confusion matrix and the shares of the
        levels of the view's part; otherwise empty and collapsed."""
        if state.get("panel") != "D":
            return [], window_class(False)
        return [metrics_window(*self.metrics_report(state))], window_class(True)

    def measures(self, state: dict) -> tuple[str, list]:
        """C1's readout: C1's own measures of the view, as (what they cover, [(heading, [(label, value, hint)])]),
        over the course, the entered deck or the entered topic (the model's layouts, whatever panel is expanded)."""
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

    # --- the model panels --------------------------------------------------------------------------------------

    def view_ids(self, state: dict) -> list[str]:
        """The cards of the view under T1: the course's, the entered deck's, or the entered topic's."""
        if state["view"] == DECKS_VIEW:
            return list(self.topic_model.topic_of)
        if state["view"] == TOPICS_VIEW:
            return [card_id for topic in self.model.topics_of_deck(state["deck"]) for card_id in topic.ids]
        topic = self.topic(state["deck"], state["topic"])
        return list(topic.ids) if topic is not None else []

    def panels(self, state: dict) -> tuple[list, list[str]]:
        """The content of each panel (the expanded one's sections, the others empty) and each box's class."""
        expanded = state.get("panel")
        sections = {letter: getattr(self, f"_section_{letter}") for letter in PANEL_LETTERS}
        contents = [sections[letter](state) if letter == expanded else [] for letter in PANEL_LETTERS]
        return contents, [box_class(letter == expanded) for letter in PANEL_LETTERS]

    def _section(self, letter: str, title: str, live: list, state: dict):
        """A level's section: its title and description, the live measures, its tables of the newest run with the
        one open row's description (the glossary), or the hint how to make the results file."""
        deck_id = state.get("deck") if letter == "T" and state["view"] != DECKS_VIEW else None
        tables = self.results.tables(letter, deck_id)
        hint = self.results.hint(RESULTS_FILES[letter])
        return level_section(title, DESCRIPTIONS[title[:2]], live, tables, hint, letter, state.get("measure"), self.glossary)

    def _section_S(self, state: dict) -> list:
        index = self.model.index
        vectoriser = index.vectoriser
        tfidf = vectoriser.first if isinstance(vectoriser, embeddings.HybridVectoriser) else vectoriser
        encoder = vectoriser.second if isinstance(vectoriser, embeddings.HybridVectoriser) else vectoriser
        vocabulary = getattr(tfidf, "vocabulary_", None)
        with_neighbour = sum(bool(index.neighbours.get(card_id)) for card_id in index.ids)
        distinct = {other for items in index.neighbours.values() for other, _score in items}
        rows = [
            ("vectoriser", embeddings.describe(vectoriser), "The vectoriser the app runs: S1's TF-IDF, or S2's sentence encoder or its hybrid with TF-IDF (app roadmap step 23)."),
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
        level = "S2" if isinstance(encoder, embeddings.SentenceEncoder) else "S1"
        live = model_readout(f"All {len(index.ids):,} cards", [(f"{level} ({embeddings.describe(vectoriser)}, cosine)", rows)])
        return [self._section("S", f"{level} · similarity", live, state)]

    def _section_C(self, state: dict) -> list:
        live = model_readout(*self.measures(state))
        if state["view"] == TOPICS_VIEW:
            live = live + [html.Div(TOPICS_NOTE, className="card-hint")]
        return [self._section("C", "C1 · course model", live, state)]

    def _section_T(self, state: dict) -> list:
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
        selected = state.get("selected")
        if state["view"] == CARDS_VIEW and topic is not None and selected in self.card_by_id and selected in self.topic_model.topic_of:
            rows += [(f"card {selected}: {label}", value, "The selected card, with T.") for label, value in self.t1_rows(selected, topic)]
        scope = {DECKS_VIEW: f"All {len(ids):,} cards", TOPICS_VIEW: f"This deck: {len(ids)} cards", CARDS_VIEW: f"This topic: {len(ids)} cards"}[state["view"]]
        margins = [self.topic_model.margin_of[card_id] for card_id in ids if card_id in self.topic_model.margin_of]
        live = model_readout(scope, [("T1 (nearest centroid)", rows)]) + [
            dcc.Graph(figure=margin_histogram(margins, self.topic_model.margin), config={"displayModeBar": False}, className="margin-histogram"),
            t1_legend(self.margin_scale, self.topic_model.margin),
        ]
        return [self._section("T", "T1 · topics", live, state)]

    def _section_D(self, state: dict) -> list:
        return [self._section("D", "D1 · difficulty", [self.metrics(state)], state)]

    # --- the moves ---------------------------------------------------------------------------------------------

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
            return {**HOME, "panel": state.get("panel"), "selected": state["deck"]}
        return {**state, "selected": None}

    def can_enter(self, state: dict) -> bool:
        return self.enter(state) is not state

    @staticmethod
    def measure_of(value) -> Optional[str]:
        """The row's key in the hidden input's value (a JSON of `measure` and `seq`, assets/measure_rows.js); None
        for the close box or anything else."""
        try:
            parsed = json.loads(value) if isinstance(value, str) else None
        except ValueError:
            return None
        measure = parsed.get("measure") if isinstance(parsed, dict) else None
        return measure if isinstance(measure, str) and measure else None

    def step(self, state: Optional[dict], trigger: Optional[str], click_data: Optional[dict] = None, measure=None) -> dict:
        """The view after a trigger: a click on the map selects a dot of the view, Open or a click on the dot's
        panel enters, Back goes up, a panel's row expands it (and collapses any other, and closes an open
        description), its close box collapses it, a results row's click opens its description (and closes the open
        one; the same row's or the close box's click closes it). A selected dot the view no longer holds is
        unselected."""
        state = {**HOME, **(state if isinstance(state, dict) else {})}
        if state["view"] not in (DECKS_VIEW, TOPICS_VIEW, CARDS_VIEW) or state["panel"] not in (None, *PANEL_LETTERS):
            state = dict(HOME)
        if trigger == "map":
            points = (click_data or {}).get("points") or []
            dot = points[0].get("customdata") if points and isinstance(points[0], dict) else None
            if dot in self.dots(state):
                state["selected"] = dot
        elif trigger in ENTER_TRIGGERS:
            state = self.enter(state)
        elif trigger == "back":
            state = self.back(state)
        elif isinstance(trigger, str) and trigger.startswith("panel-") and trigger[6:] in PANEL_LETTERS:
            if state["panel"] != trigger[6:]:
                state["measure"] = None
            state["panel"] = trigger[6:]
        elif isinstance(trigger, str) and trigger.startswith("close-") and trigger[6:] == state["panel"]:
            state["panel"] = None
            state["measure"] = None
        elif trigger == MEASURE_INPUT:
            key = self.measure_of(measure)
            state["measure"] = None if key is None or key == state["measure"] else key
        if state.get("selected") is not None and state["selected"] not in self.dots(state):
            state["selected"] = None
        return state


def register_callback(app: Dash, page: Page):
    """The page's one callback; returns the function (for tests). Its outputs: the map, window A, the view, the
    line under the top bar, Open's and Back's disabled state, the four panels' contents and their classes, then the
    difficulty window's content and class. Its inputs end with the hidden input of a results row's click and the
    view's state."""

    @app.callback(
        Output("map", "figure"),
        Output("dot-panel", "children"),
        Output("view", "data"),
        Output("where", "children"),
        Output("open", "disabled"),
        Output("back", "disabled"),
        *[Output(f"content-{letter}", "children") for letter in PANEL_LETTERS],
        *[Output(f"box-{letter}", "className") for letter in PANEL_LETTERS],
        Output("difficulty-window", "children"),
        Output("difficulty-window", "className"),
        Input("map", "clickData"),
        Input("open", "n_clicks"),
        Input("back", "n_clicks"),
        Input("dot-panel", "n_clicks"),
        *[Input(f"panel-{letter}", "n_clicks") for letter in PANEL_LETTERS],
        *[Input(f"close-{letter}", "n_clicks") for letter in PANEL_LETTERS],
        Input(MEASURE_INPUT, "value"),
        State("view", "data"),
    )
    def update(click_data, open_clicks, back_clicks, panel_clicks, *rest):
        from dash import ctx

        state = page.step(rest[-1], ctx.triggered_id, click_data, measure=rest[-2])
        contents, classes = page.panels(state)
        window, window_class_name = page.difficulty_window(state)
        return (
            map_figure(page, state),
            page.dot_panel(state),
            state,
            page.where(state),
            not page.can_enter(state),
            state["view"] == DECKS_VIEW and not state.get("selected"),
            *contents,
            *classes,
            window,
            window_class_name,
        )

    return update


def create_app(page: Page) -> Dash:
    """The Dash app of the page, in the viewer's theme (assets/style.css), with its callback."""
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.layout = page_layout(HOME)
    register_callback(app, page)
    return app
