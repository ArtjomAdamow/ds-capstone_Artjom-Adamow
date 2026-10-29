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

The callback (register_callback) keeps the view in the `view` store: a click on a dot selects it, Open or a click on
a selected deck's or topic's panel enters it, Back goes up a level with the dot it came from selected (in the decks
view it unselects). No keys, groups, editing, progress, quiz or image window.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
from dash import Dash, Input, Output, State
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..course_table import CourseTable
from ..figures import dendrogram_figure
from ..models import clustering
from ..models.course_model import CourseModel, Point, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.layout import as_dense, fibonacci_sphere
from .layout import (
    CARDS_VIEW, DECKS_VIEW, TOPICS_VIEW, card_panel, deck_panel, map_figure, metrics_panel, overview_panel, page_layout,
    topic_panel,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TITLE = "Guided learning: the models"
DECK_PART = 5  # the decks view's preset: parts of at most five decks (as the viewer's Helper presets them)
CARD_PART = 4  # a topic's preset: parts of at most four cards
RING_RADIUS = 0.75  # the topics' ring around the deck's centroid
HOME = {"view": DECKS_VIEW, "deck": None, "topic": None, "selected": None}
ENTER_TRIGGERS = ("open", "dot-panel")


def merges_of(vectors) -> np.ndarray:
    """scipy's linkage of the rows, as models.clustering.merge_links makes it (the same distances and method), so
    its merges are the connections' merge order with their real distances."""
    return linkage(squareform(clustering.cosine_distances(vectors), checks=False), method=clustering.METHOD)


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
        """The line under the top bar: the number of decks, the entered deck, or the deck and the topic."""
        if state["view"] == DECKS_VIEW:
            return f"{len(self.deck_points())} decks"
        deck = self.deck_by_id.get(state["deck"])
        name = deck.name if deck else ""
        if state["view"] == TOPICS_VIEW:
            return name
        topic = self.topic(state["deck"], state["topic"])
        return f"{name} › {topic.name if topic else ''}"

    def dot_panel(self, state: dict):
        """Window A: the selected dot's panel, or the dendrogram of the view's connections with the preset's line."""
        view, selected = state["view"], state.get("selected")
        if view == DECKS_VIEW and selected in self.deck_by_id:
            return deck_panel(self.deck_by_id[selected], len(self.topics(selected)))
        if view == TOPICS_VIEW and selected:
            topic = self.topic_of_dot(state["deck"], selected)
            if topic is not None:
                return topic_panel(topic, self.card_by_id, rest=self._rest(topic))
        if view == CARDS_VIEW and selected in self.card_by_id:
            return card_panel(self.card_by_id[selected], self.difficulty.predictions.get(selected))
        if view == DECKS_VIEW:
            tree = self.deck_tree()
            kept = self.deck_rank
            title, subtitle = "Constellations of decks", "The model's deck connections, cut into parts of at most five decks."
        elif view == TOPICS_VIEW:
            tree, kept, title, subtitle = None, None, self.where(state), None
        else:
            topic = self.topic(state["deck"], state["topic"])
            tree = self.card_tree(topic)
            kept = len(self.card_links(topic))
            title = topic.name if topic else ""
            subtitle = "The model's card connections, cut into parts of at most four cards."
        figure = None
        if tree is not None:
            merges, ids = tree
            label_of = self._labels(view, ids)
            figure = dendrogram_figure(merges, ids, label_of, kept)
        return overview_panel(title, figure, view, subtitle)

    def metrics(self, state: dict):
        """Window B: D1's metrics of the view: the course, the entered deck, or the entered topic."""
        if state["view"] == DECKS_VIEW:
            return metrics_panel(self.difficulty, "all decks")
        deck = self.deck_by_id.get(state["deck"])
        name = deck.name if deck else ""
        part = self.difficulty.for_deck(name)
        if state["view"] == TOPICS_VIEW:
            return metrics_panel(part, name)
        topic = self.topic(state["deck"], state["topic"])
        if topic is None:
            return metrics_panel(part, name)
        return metrics_panel(part.for_topic(topic.name, topic.ids), f"{name} › {topic.name}")

    def _rest(self, topic: TopicLayout) -> bool:
        return any(seed.rest for seed in self.model.topic_model.seeds if seed.topic_id == topic.topic_id and seed.deck == topic.deck)

    def _labels(self, view: str, ids) -> dict[str, str]:
        if view == DECKS_VIEW:
            return {deck_id: self.deck_by_id[deck_id].name if deck_id in self.deck_by_id else deck_id for deck_id in ids}
        return {card_id: self.card_by_id[card_id].question for card_id in ids}

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
        )

    return update


def create_app(page: Page) -> Dash:
    """The Dash app of the page, in the viewer's theme (assets/style.css), with its callback."""
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.layout = page_layout(HOME)
    register_callback(app, page)
    return app
