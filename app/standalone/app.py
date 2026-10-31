"""The standalone page's models and its one callback (Dash; nothing of the viewer).

Page fits the course model and D1 on the course table once, as the viewer's serving process does, and answers what
the three views show; everything shown is the model's, nothing is stored:

* decks view: the decks at the model's positions on the sphere, the clustering's connections below the preset
  (parts of at most DECK_PART decks, models.clustering.preset_rank);
* topics view (an entered deck): the deck's model topics spaced evenly on a ring around the centroid in file order
  (the model has no topic positions; step 19's topic vectors are the upgrade), no connections;
* cards view (an entered topic): the topic's cards at the model's directions on the surface of its sphere (a card
  without one, in a topic of a single card, at an even spread), the connections below the preset (parts of at most
  CARD_PART cards).

In every view the map's readout shows the course model C1's own measures of it (Page.measures): the layouts'
neighbour measures, the connections the preset keeps and T1's margins, over the course, the deck or the topic. What
needs the scripted topics as its reference (T1's agreement with them, the moved cards) is a comparison and is left
out.

The panels, the map's hover and the line under the top bar (Page.where) name the dots and the views by id
(Page.scope): no learning material, no card's text, no deck's description and no deck's name. A topic's name, the
one name that is no id, shows nowhere either: the map's hover and the line under the top bar give its label instead
(Page.topic_label: its topic id, then a hash of its name, made at runtime and stored nowhere).

The callback (register_callback) keeps the view in the `view` store: a click on a dot selects it, Open or a click on
a selected deck's or topic's panel enters it, Back goes up a level with the dot it came from selected (in the decks
view it unselects). No keys, groups, editing, progress, quiz or image window.
"""

import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np
from dash import Dash, Input, Output, State
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..course_table import CourseTable
from ..figures import dendrogram_figure, number
from ..models import clustering
from ..models.course_model import CARD_NEIGHBOURS, DECK_NEIGHBOURS, CourseModel, Point, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.layout import as_dense, fibonacci_sphere
from .layout import (
    CARDS_VIEW, DECKS_VIEW, TOPICS_VIEW, card_panel, deck_panel, map_figure, metrics_panel, model_readout, overview_panel,
    page_layout, topic_panel,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TITLE = "Guided learning: the models"
DECK_PART = 5  # the decks view's preset: parts of at most five decks (as the viewer's Helper presets them)
CARD_PART = 4  # a topic's preset: parts of at most four cards
RING_RADIUS = 0.75  # the topics' ring around the deck's centroid
TOPIC_HASH_BYTES = 3  # a topic label's hash: six hex digits
HOME = {"view": DECKS_VIEW, "deck": None, "topic": None, "selected": None}
ENTER_TRIGGERS = ("open", "dot-panel")
# The readout's hints, a row's tooltip
TRUSTWORTHINESS = "scikit-learn's trustworthiness of the map for the text vectors: 1 when none of a dot's k nearest on the map is a stranger by text."
OVERLAP = "The share of a dot's k most similar by text that are also among its k nearest on the map, averaged over the dots."
MEAN_OVERLAP = (
    f"The mean over the topics of more than {CARD_NEIGHBOURS + 1} cards: in a smaller one every other card is a neighbour "
    "(k = n − 1), so the overlap is 1 by construction."
)
MEAN_TRUSTWORTHINESS = "The mean over the topics of more than 2k cards, where scikit-learn defines it."
BETWEEN = "The cards whose topic T1 chose by a margin below {margin} (the best centroid's cosine minus the second best's)."


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
    """The models fitted on the course table, and what the three views show of them."""

    def __init__(
        self, table: CourseTable, model: Optional[CourseModel] = None, difficulty: Optional[DifficultyReport] = None
    ) -> None:
        self.table = table
        self.model = model if model is not None else CourseModel(table.cards, table.deck_ids, table.seeds)
        self.difficulty = difficulty if difficulty is not None else fit_difficulty(table.cards, self.model.index)
        self.card_by_id = {card.id: card for card in table.cards}
        self.deck_by_id = {deck.deck_id: deck for deck in table.decks}
        self._row_of = {card_id: row for row, card_id in enumerate(self.model.index.ids)}
        self.deck_rank = clustering.preset_rank(self.model.deck_links, DECK_PART)
        self.between = self.model.hidden  # the cards T1 placed by a margin below its threshold

    # --- the decks view ----------------------------------------------------------------------------------------

    def deck_points(self) -> dict[str, Point]:
        """The decks of the table the model placed, in deck id order."""
        return {deck.deck_id: self.model.deck_points[deck.deck_id] for deck in self.table.decks if deck.deck_id in self.model.deck_points}

    def deck_links(self) -> list[tuple[str, str]]:
        """The deck connections below the preset: the first merges, until a part would hold more than DECK_PART."""
        return list(self.model.deck_links[: self.deck_rank])

    def deck_tree(self):
        """(the merges, the deck ids of their rows) of the decks' clustering; None below two decks."""
        ids = [self.model.deck_ids[name] for name in self.model.deck_names]
        return (merges_of(self.model.deck_vectors), ids) if len(ids) > 1 else None

    # --- the topics view ---------------------------------------------------------------------------------------

    def topics(self, deck_id: Optional[str]) -> list[TopicLayout]:
        """The deck's model topics in file order."""
        return self.model.topics_of_deck(deck_id) if deck_id else []

    @staticmethod
    def topic_dot(topic: TopicLayout) -> str:
        """A topic's dot id on the map: the deck id and the topic id."""
        return f"{topic.deck_id}:{topic.topic_id}"

    @staticmethod
    def topic_label(topic: TopicLayout) -> str:
        """A topic's label in place of its name: the topic id, then a hash of the name (blake2s, so the same on
        every run), made at runtime and stored nowhere."""
        digest = hashlib.blake2s(topic.name.encode("utf-8"), digest_size=TOPIC_HASH_BYTES).hexdigest()
        return f"{topic.topic_id}-{digest}"

    def topic_points(self, deck_id: Optional[str]) -> dict[str, Point]:
        """The deck's topics spaced evenly on a ring around its centroid, in file order."""
        topics = self.topics(deck_id)
        return {
            self.topic_dot(topic): (
                RING_RADIUS * math.cos(2 * math.pi * index / len(topics)),
                RING_RADIUS * math.sin(2 * math.pi * index / len(topics)),
                0.0,
            )
            for index, topic in enumerate(topics)
        }

    def topic(self, deck_id: Optional[str], topic_id: Optional[str]) -> Optional[TopicLayout]:
        return self.model.topics.get((deck_id, topic_id))

    def topic_of_dot(self, deck_id: Optional[str], dot: Optional[str]) -> Optional[TopicLayout]:
        return next((topic for topic in self.topics(deck_id) if self.topic_dot(topic) == dot), None)

    # --- the cards view ----------------------------------------------------------------------------------------

    def card_points(self, topic: TopicLayout) -> dict[str, Point]:
        """The topic's cards at the model's directions; a card without one (a topic of a single card) at an even
        spread over the sphere."""
        missing = [card_id for card_id in topic.ids if card_id not in topic.positions]
        spread = {card_id: tuple(float(value) for value in point) for card_id, point in zip(missing, fibonacci_sphere(len(missing)))}
        return {card_id: topic.positions.get(card_id) or spread[card_id] for card_id in topic.ids}

    def card_links(self, topic: Optional[TopicLayout]) -> list[tuple[str, str]]:
        """The topic's connections below the preset: the first merges, until a part would hold more than CARD_PART."""
        if topic is None:
            return []
        return list(topic.links[: clustering.preset_rank(topic.links, CARD_PART)])

    def card_tree(self, topic: Optional[TopicLayout]):
        """(the merges, the card ids of their rows) of the topic's clustering; None below two cards."""
        if topic is None or len(topic.links) < 1:
            return None
        vectors = as_dense(self.model.index.vectors[[self._row_of[card_id] for card_id in topic.ids]])
        return merges_of(vectors), list(topic.ids)

    # --- what a view shows -------------------------------------------------------------------------------------

    def dots(self, state: dict) -> list[str]:
        """The dot ids of the view in `state`."""
        if state["view"] == DECKS_VIEW:
            return list(self.deck_points())
        if state["view"] == TOPICS_VIEW:
            return list(self.topic_points(state["deck"]))
        topic = self.topic(state["deck"], state["topic"])
        return list(topic.ids) if topic is not None else []

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
        """Window A: the selected dot's ids and counts, or the dendrogram of the view's connections with the preset's
        line, its leaves by id."""
        view, selected = state["view"], state.get("selected")
        if view == DECKS_VIEW and selected in self.deck_by_id:
            topics = self.topics(selected)
            return deck_panel(selected, len(topics), sum(len(topic.ids) for topic in topics))
        if view == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected)
            if topic is not None:
                return topic_panel(topic, self.card_by_id, rest=self._rest(topic))
        if view == CARDS_VIEW and selected in self.card_by_id:
            return card_panel(
                self.card_by_id[selected], state["deck"], state["topic"], self.difficulty.predictions.get(selected),
                between=selected in self.between,
            )
        if view == DECKS_VIEW:
            tree = self.deck_tree()
            kept = self.deck_rank
            title, subtitle = "Constellations of decks", "The model's deck connections, cut into parts of at most five decks."
        elif view == TOPICS_VIEW:
            tree, kept, title, subtitle = None, None, self.scope(state), None
        else:
            topic = self.topic(state["deck"], state["topic"])
            tree = self.card_tree(topic)
            kept = len(self.card_links(topic))
            title = self.scope(state)
            subtitle = "The model's card connections, cut into parts of at most four cards."
        figure = None
        if tree is not None:
            merges, ids = tree
            figure = dendrogram_figure(merges, ids, {}, kept)  # no labels: the leaves by id
        return overview_panel(title, figure, view, subtitle)

    def metrics(self, state: dict):
        """Window B: D1's metrics of the view: the course, the entered deck, or the entered topic, the last with the
        cards' colour legend."""
        if state["view"] == DECKS_VIEW:
            return metrics_panel(self.difficulty, self.scope(state))
        deck = self.deck_by_id.get(state["deck"])
        part = self.difficulty.for_deck(deck.name if deck else "")
        topic = self.topic(state["deck"], state["topic"])
        if state["view"] == TOPICS_VIEW or topic is None:
            return metrics_panel(part, f"deck {state['deck']}")
        return metrics_panel(part.for_topic(topic.name, topic.ids), self.scope(state), legend=True)

    def measures(self, state: dict) -> tuple[str, list]:
        """The map's readout: C1's own measures of the view, as (what they cover, [(heading, [(label, value, hint)])]),
        over the course, the entered deck or the entered topic."""
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
            ring = ("dot positions", "none yet", "The model places no topic: the dots stand on a ring in file order.")
            return f"This deck: {len(topics)} topics, {sum(len(topic.ids) for topic in topics)} cards", [
                ("Topics (T1)", self._topic_measures(topics) + [ring]),
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
                ("connections kept", f"{len(self.card_links(topic))} of {len(topic.links)}", f"Below the cut-off: parts of at most {CARD_PART} cards."),
            ]
        between = ("between two topics", f"{len(self.between.intersection(topic.ids))} of {count}", BETWEEN.format(margin=self.model.topic_model.margin))
        return f"This topic: {count} cards", [("Cards on the sphere", sphere), ("Topic (T1)", [between])]

    def _topic_measures(self, topics: list[TopicLayout]) -> list[tuple[str, str, str]]:
        """T1 over these topics: how many keep cards, and how many of their cards it placed between two topics."""
        ids = [card_id for topic in topics for card_id in topic.ids]
        return [
            ("with cards", f"{sum(bool(topic.ids) for topic in topics)} of {len(topics)}", "A topic T1 left without cards keeps its dot."),
            ("between two topics", _share(len(self.between.intersection(ids)), len(ids)), BETWEEN.format(margin=self.model.topic_model.margin)),
        ]

    @staticmethod
    def _layout_measures(topics: list[TopicLayout]) -> list[tuple[str, str, str]]:
        """The card layouts of these topics: the mean overlap where k leaves cards out, the mean trustworthiness where
        it is defined."""
        overlaps = [topic.overlap for topic in topics if len(topic.ids) - 1 > CARD_NEIGHBOURS]
        trusted = [topic.trustworthiness for topic in topics if not math.isnan(topic.trustworthiness)]
        return [("top-k overlap", _mean(overlaps), MEAN_OVERLAP), ("trustworthiness", _mean(trusted), MEAN_TRUSTWORTHINESS)]

    def _rest(self, topic: TopicLayout) -> bool:
        return any(seed.rest for seed in self.model.topic_model.seeds if seed.topic_id == topic.topic_id and seed.deck == topic.deck)

    # --- the moves ---------------------------------------------------------------------------------------------

    def enter(self, state: dict) -> dict:
        """Open: a selected deck's topics view, a selected topic's cards view; nothing else."""
        selected = state.get("selected")
        if state["view"] == DECKS_VIEW and selected in self.deck_by_id:
            return {"view": TOPICS_VIEW, "deck": selected, "topic": None, "selected": None}
        if state["view"] == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected)
            if topic is not None:
                return {"view": CARDS_VIEW, "deck": state["deck"], "topic": topic.topic_id, "selected": None}
        return state

    def back(self, state: dict) -> dict:
        """Back: up a level, with the dot it came from selected; in the decks view the selection goes."""
        if state["view"] == CARDS_VIEW:
            topic = self.topic(state["deck"], state["topic"])
            return {"view": TOPICS_VIEW, "deck": state["deck"], "topic": None, "selected": self.topic_dot(topic) if topic else None}
        if state["view"] == TOPICS_VIEW:
            return {**HOME, "selected": state["deck"]}
        return {**state, "selected": None}

    def can_enter(self, state: dict) -> bool:
        return self.enter(state) is not state

    def step(self, state: Optional[dict], trigger: Optional[str], click_data: Optional[dict] = None) -> dict:
        """The view after a trigger: a click on the map selects a dot of the view, Open or a click on the dot's
        panel enters, Back goes up."""
        state = {**HOME, **(state if isinstance(state, dict) else {})}
        if state["view"] not in (DECKS_VIEW, TOPICS_VIEW, CARDS_VIEW):
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
        return state


def register_callback(app: Dash, page: Page):
    """The page's one callback; returns the function (for tests)."""

    @app.callback(
        Output("map", "figure"),
        Output("dot-panel", "children"),
        Output("metrics-panel", "children"),
        Output("view", "data"),
        Output("where", "children"),
        Output("open", "disabled"),
        Output("back", "disabled"),
        Output("model-readout", "children"),
        Input("map", "clickData"),
        Input("open", "n_clicks"),
        Input("back", "n_clicks"),
        Input("dot-panel", "n_clicks"),
        State("view", "data"),
    )
    def update(click_data, open_clicks, back_clicks, panel_clicks, state):
        from dash import ctx

        state = page.step(state, ctx.triggered_id, click_data)
        return (
            map_figure(page, state),
            page.dot_panel(state),
            page.metrics(state),
            state,
            page.where(state),
            not page.can_enter(state),
            state["view"] == DECKS_VIEW and not state.get("selected"),
            model_readout(*page.measures(state)),
        )

    return update


def create_app(page: Page) -> Dash:
    """The Dash app of the page, in the viewer's theme (assets/style.css), with its callback."""
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.layout = page_layout(HOME)
    register_callback(app, page)
    return app
