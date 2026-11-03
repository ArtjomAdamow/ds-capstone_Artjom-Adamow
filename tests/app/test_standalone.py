"""The standalone page (app.standalone) on the test course as a course table, with nothing of the viewer:
the three views through the callback (select, Open, a click on the panel, Back), the four model panels (one
expanded at a time, the close box), the map per panel (random positions and the scripted topics with every panel
collapsed, C1's positions with the connections, the centroid and the axes with C, T1's markers with T, the difficulty
colours with D, the predicted quizzes as crosses in the predicted colour and the legend inside the map), the dot's
ids in its panel, the dendrogram of the view's membership with the preset's line and its leaves by id, D1's scores
in the D panel and its matrix and shares in the difficulty window under window A, C1's readout in the C panel, the
results tables of the newest run with their floor, a row's description opened by a click (one at a time, the close
box), the decks' hover and the line under the top bar by id, no learning material and no deck's or topic's name in
any view or panel, and that neither the page nor the table, the figures or the mode question import the viewer or
the glue (a subprocess with the viewer blocked). The real course's table serves the three views too (skipped
without it)."""

import model_support

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plotly.utils import PlotlyJSONEncoder

from app.course_table import load_table, missing_files
from app.figures import GOLD, PREDICTED_COLOR
from app.models.course_model import TopicLayout
from app.models.topic_model import FOCUS_MARGIN
from app.standalone import main as standalone_main
from app.standalone.app import (
    ASSETS_DIR, CARD_PART, DECK_PART, HOME, MEASURE_INPUT, PANEL_LETTERS, Page, create_app, register_callback,
)
from app.standalone.layout import (
    AXIS_RANGE, AXIS_TICKS, CROSS_SYMBOL, EMPTIED_SYMBOL, GREY, LEVEL_COLORS, MOVED_IN_COLOR, MOVED_IN_SYMBOL, MOVED_OUT_COLOR,
    MOVED_OUT_SYMBOL, PREDICTED_SYMBOL, TOPICS_NOTE, WINDOW_NOTE, box_class, margin_color, page_layout, window_class,
)
from app.standalone.results import NO_FOLDER, Results, course_note, read_glossary

OUT = {"figure": 0, "panel": 1, "view": 2, "where": 3, "open_disabled": 4, "back_disabled": 5}
for _index, _letter in enumerate(PANEL_LETTERS):
    OUT[f"content_{_letter}"] = 6 + _index
    OUT[f"class_{_letter}"] = 6 + len(PANEL_LETTERS) + _index
OUT["window_d"] = 6 + 2 * len(PANEL_LETTERS)  # the difficulty window's content, then its class
OUT["class_d"] = OUT["window_d"] + 1
INPUTS = 5 + 2 * len(PANEL_LETTERS)  # the map, Open, Back, window A, the four rows, the four close boxes, the measure input


def measure_value(key) -> str:
    """What assets/measure_rows.js writes into the hidden input: the row's key (None for the close box) and a
    sequence number."""
    measure_value.seq = getattr(measure_value, "seq", 0) + 1
    return json.dumps({"measure": key, "seq": measure_value.seq})


def measure_keys(content) -> list[str]:
    """The rows' keys of a panel's content, in order."""
    found = []

    def walk(component):
        children = getattr(component, "children", None)
        key = getattr(component, "data-measure", None)
        if key is not None:
            found.append(key)
        for child in children if isinstance(children, list) else ([children] if children is not None else []):
            if hasattr(child, "children"):
                walk(child)

    for component in content if isinstance(content, list) else [content]:
        walk(component)
    return found


def readout_values(page: Page, state: dict) -> tuple[str, dict]:
    """The readout's scope and its values by (heading, label)."""
    scope, sections = page.measures({**HOME, **state})
    return scope, {(heading, label): value for heading, rows in sections for label, value, _hint in rows}


def id_rows(panel) -> dict:
    """The selected dot's ids and counts in window A, by label."""
    table = next(child for child in panel.children if getattr(child, "className", None) == "readout-table id-table")
    return {row.children[0].children: row.children[1].children for row in table.children.children}


def dendrogram(panel):
    """The dendrogram's figure in window A."""
    return next(child.figure for child in panel.children if getattr(child, "className", None) == "dendrogram-graph")


def dumped(value) -> str:
    return json.dumps(value, cls=PlotlyJSONEncoder)


# The test course as a course table (model_support writes it, as export_course.py does from the course file)
TOPIC_OF, DESCRIPTIONS, write_test_table = model_support.TOPIC_OF, model_support.DESCRIPTIONS, model_support.write_test_table


class RecordingApp:
    def callback(self, *dependencies):
        return lambda function: function


class PageDriver(unittest.TestCase):
    """The callback driven by triggers, the view kept between calls."""

    page: Page

    def setUp(self):
        self.update = register_callback(RecordingApp(), self.page)
        self.state = None

    def call(self, trigger, click=None, measure=0):
        with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
            result = self.update(click, *([0] * (INPUTS - 2)), measure, self.state)
        self.state = result[OUT["view"]]
        return result

    def click(self, dot):
        return self.call("map", {"points": [{"customdata": dot}]})

    def expand(self, letter):
        return self.call(f"panel-{letter}")

    def close(self, letter):
        return self.call(f"close-{letter}")

    def describe(self, key):
        """A click on a results row (its key), or on the open description's close box (None)."""
        return self.call(MEASURE_INPUT, measure=measure_value(key))

    def trace(self, result, name):
        return next((trace for trace in result[OUT["figure"]].data if trace.name == name), None)

    def legend(self, result) -> list[str]:
        """The legend inside the map: its entries' names, in order; none without it."""
        figure = result[OUT["figure"]]
        return [trace.name for trace in figure.data if trace.showlegend] if figure.layout.showlegend else []

    def colors(self, result):
        trace = self.trace(result, "Cards")
        return dict(zip(trace.customdata, trace.marker.color))

    def symbols(self, result):
        trace = self.trace(result, "Cards")
        symbols = trace.marker.symbol
        return dict(zip(trace.customdata, symbols if isinstance(symbols, (list, tuple)) else [symbols] * len(trace.customdata)))

    def points(self, result, name):
        trace = self.trace(result, name)
        return {dot: (x, y, z) for dot, x, y, z in zip(trace.customdata, trace.x, trace.y, trace.z)}

    def axes_visible(self, result) -> bool:
        scene = result[OUT["figure"]].layout.scene
        return bool(scene.xaxis.visible) and bool(scene.yaxis.visible) and bool(scene.zaxis.visible)

    def classes(self, result) -> dict:
        return {letter: result[OUT[f"class_{letter}"]] for letter in PANEL_LETTERS}

    def enter_topic(self, deck_id, topic_id):
        self.call(None)
        self.click(deck_id)
        self.call("open")
        self.click(f"{deck_id}:{topic_id}")
        return self.call("dot-panel")  # a click on the topic's panel enters it as Open does


class StandaloneTest(PageDriver):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.table = load_table(write_test_table(Path(cls._tmp.name)))
        cls.page = Page(cls.table, results_dir=Path(cls._tmp.name) / "results")  # no results files: the hints

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_table_is_the_models_fixture(self):
        self.assertEqual(self.table.cards, model_support.CARDS)
        self.assertEqual(self.table.seeds, model_support.SEEDS)
        self.assertEqual(self.table.deck_ids, model_support.DECK_IDS)

    def test_with_every_panel_collapsed_the_decks_stand_at_random_without_connections_or_axes(self):
        result = self.call(None)
        self.assertEqual(result[OUT["view"]], HOME)
        self.assertEqual(result[OUT["where"]], "3 decks")
        self.assertEqual((result[OUT["open_disabled"]], result[OUT["back_disabled"]]), (True, True))
        self.assertEqual(self.classes(result), {letter: box_class(False) for letter in PANEL_LETTERS})
        decks = self.trace(result, "Decks")
        self.assertEqual(list(decks.customdata), ["01", "02", "03"])
        self.assertEqual(list(decks.text), ["01", "02", "03"])  # the hover: the deck id, not its name
        self.assertEqual(self.points(result, "Decks"), Page.random_points(["01", "02", "03"]))
        for point in self.points(result, "Decks").values():
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in point)), 1.0)  # on the sphere
        self.assertNotEqual(self.points(result, "Decks")["01"], self.page.model.deck_points["01"])
        self.assertIsNone(self.trace(result, "Connections"))
        self.assertFalse(self.axes_visible(result))
        tree = dendrogram(result[OUT["panel"]])  # the dendrogram's window is no model's panel: the tree shows
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["01", "02", "03"])  # the leaves by id, not by name
        self.assertEqual(DECK_PART, 5)

    def test_c_shows_the_models_positions_its_connections_below_the_preset_and_the_axes(self):
        self.call(None)
        result = self.expand("C")
        self.assertEqual(self.state["panel"], "C")
        self.assertEqual(self.classes(result)["C"], box_class(True))
        self.assertEqual(
            self.points(result, "Decks"), {deck_id: self.page.model.deck_points[deck_id] for deck_id in ("01", "02", "03")}
        )
        links = self.trace(result, "Connections")
        self.assertEqual((links.line.color, links.line.dash), (GOLD, "dash"))
        self.assertEqual(self.page.deck_links(), self.page.model.deck_links)  # three decks fit into one part of five
        self.assertEqual(list(links.x).count(None), len(self.page.model.deck_links))
        scene = result[OUT["figure"]].layout.scene
        for name, axis in (("x", scene.xaxis), ("y", scene.yaxis), ("z", scene.zaxis)):
            self.assertTrue(axis.visible)
            self.assertEqual((list(axis.range), list(axis.tickvals), axis.title.text), (AXIS_RANGE, AXIS_TICKS, name))
            self.assertFalse(axis.showspikes)  # the hover stays the dot's id
        content = str(result[OUT["content_C"]])
        self.assertIn("C1 · course model", content)
        self.assertIn("All 3 decks, 13 cards", content)  # C1's readout moved into the panel

    def test_one_panel_is_expanded_at_a_time_and_the_close_box_collapses_it(self):
        self.call(None)
        self.expand("S")
        result = self.expand("T")
        self.assertEqual(self.state["panel"], "T")
        self.assertEqual(self.classes(result), {letter: box_class(letter == "T") for letter in PANEL_LETTERS})
        self.assertEqual(result[OUT["content_S"]], [])  # a collapsed panel has no content
        self.assertIn("T1 · topics", str(result[OUT["content_T"]]))
        result = self.close("S")  # not the expanded one: nothing changes
        self.assertEqual(self.state["panel"], "T")
        result = self.close("T")
        self.assertIsNone(self.state["panel"])
        self.assertEqual(self.classes(result), {letter: box_class(False) for letter in PANEL_LETTERS})
        self.assertTrue(all(result[OUT[f"content_{letter}"]] == [] for letter in PANEL_LETTERS))

    def test_a_click_selects_a_deck_and_open_enters_its_topics_at_random_positions(self):
        self.call(None)
        result = self.click("02")
        self.assertEqual(self.state["selected"], "02")
        self.assertFalse(result[OUT["open_disabled"]])
        self.assertEqual(id_rows(result[OUT["panel"]]), {"deck_id": "02", "topics": 2, "cards": 4})
        self.assertNotIn("Beta", dumped(result))  # nor its name anywhere
        self.assertIsNotNone(self.trace(result, "Selected"))

        result = self.call("open")
        self.assertEqual(self.state, {**HOME, "view": "topics", "deck": "02"})
        self.assertEqual(result[OUT["where"]], "deck 02")  # by id, not by name
        topics = self.trace(result, "Topics")
        self.assertEqual(list(topics.customdata), ["02:01", "02:02"])  # file order
        self.assertEqual(list(topics.text), [Page.topic_label(self.page.topic("02", topic_id)) for topic_id in ("01", "02")])
        for x, y, z in zip(topics.x, topics.y, topics.z):
            self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 1.0)  # at random on the sphere, never a ring
        self.assertEqual([trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"], [])  # no model: no centroid
        self.assertIsNone(self.trace(result, "Connections"))
        self.assertIn("No connections between topics", str(result[OUT["panel"]]))

    def test_the_topics_view_shows_the_centroid_with_the_model_and_never_draws_connections(self):
        self.call(None)
        self.click("02")
        self.call("open")
        for letter in ("C", "T"):
            result = self.expand(letter)
            centroid = [trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"]
            self.assertEqual([trace.marker.size for trace in centroid], [16, 10, 4], letter)
            self.assertEqual({trace.marker.color for trace in centroid}, {GREY})  # a 3D marker ignores a colour's alpha
            self.assertAlmostEqual(1 - math.prod(1 - trace.marker.opacity for trace in centroid), 0.4)  # at the core
            self.assertIsNone(self.trace(result, "Connections"), letter)
            self.assertTrue(self.axes_visible(result), letter)
            self.assertEqual(self.points(result, "Topics"), Page.random_points(["02:01", "02:02"]))  # the same draw
        self.assertIn(TOPICS_NOTE, str(result[OUT["content_C"]]) if self.state["panel"] == "C" else str(self.expand("C")[OUT["content_C"]]))
        for letter in ("S", "D"):
            result = self.expand(letter)
            self.assertEqual([trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"], [], letter)
            self.assertFalse(self.axes_visible(result), letter)

    def test_an_emptied_topic_is_an_open_circle(self):
        self.call(None)
        self.click("01")
        result = self.call("open")
        topics = self.trace(result, "Topics")
        symbols = dict(zip(topics.customdata, topics.marker.symbol))
        self.assertEqual(symbols["01:03"], EMPTIED_SYMBOL)  # "Empty": no cards
        self.assertEqual(symbols["01:01"], "circle")
        result = self.click("01:03")
        self.assertEqual(
            id_rows(result[OUT["panel"]]),
            {"deck_id": "01", "topic_id": "03", "rest topic": "no", "cards": 0, "flashcards": 0, "quizzes": 0, "rated": 0},
        )
        self.assertIn("No cards", str(result[OUT["panel"]]))
        result = self.click("01:01")
        self.assertEqual(
            id_rows(result[OUT["panel"]]),
            {"deck_id": "01", "topic_id": "01", "rest topic": "no", "cards": 3, "flashcards": 2, "quizzes": 1, "rated": 0},
        )

    def test_the_cards_view_follows_the_panel_grey_at_random_coloured_with_d_placed_and_connected_with_c(self):
        result = self.enter_topic("01", "02")  # Dictionaries: the rated quiz and a flashcard of the quiz file
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "01", "topic": "02"})
        self.assertEqual(result[OUT["where"]], f"deck 01 › topic {Page.topic_label(self.page.topic('01', '02'))}")
        self.assertEqual(self.trace(result, "Surface").opacity, 0.1)
        self.assertEqual(self.colors(result), {"301003": GREY, "401005": GREY})  # no colours without D
        self.assertEqual(self.legend(result), [])
        self.assertEqual(self.points(result, "Cards"), Page.random_points(["301003", "401005"]))
        self.assertIsNone(self.trace(result, "Connections"))
        result = self.expand("D")
        self.assertEqual(self.colors(result), {"301003": LEVEL_COLORS["easy"], "401005": GREY})  # the rated quiz, the flashcard
        self.assertEqual(self.symbols(result), {"301003": "circle", "401005": "circle"})
        self.assertEqual(self.legend(result), ["easy", "flashcard"])  # inside the map: the classes the view holds
        legend = result[OUT["figure"]].layout.legend
        self.assertEqual((legend.x, legend.y, legend.itemclick), (0.01, 0.99, False))
        self.assertEqual(self.points(result, "Cards"), Page.random_points(["301003", "401005"]))  # the same draw as with none
        self.assertIsNone(self.trace(result, "Connections"))
        result = self.expand("C")
        self.assertEqual(self.legend(result), [])
        topic = self.page.topic("01", "02")
        self.assertEqual(self.points(result, "Cards"), {card_id: topic.positions[card_id] for card_id in topic.ids})
        self.assertEqual(self.colors(result), {"301003": GREY, "401005": GREY})
        links = self.trace(result, "Connections")
        self.assertEqual(links.line.dash, "dash")
        self.assertEqual(list(links.x).count(None), 1)  # the topic's one connection
        self.assertTrue(self.axes_visible(result))
        self.assertEqual(CARD_PART, 4)

    def test_t_colours_the_cards_by_their_margin_and_marks_a_card_between_two_topics_with_a_cross(self):
        self.enter_topic("01", "01")  # Lists: three cards, one of them between Lists and Dictionaries
        result = self.expand("T")
        topic = self.page.topic("01", "01")
        self.assertEqual(self.points(result, "Cards"), {card_id: topic.positions[card_id] for card_id in topic.ids})  # C1's positions
        margins = self.page.topic_model.margin_of
        self.assertEqual(self.colors(result), {card_id: margin_color(margins[card_id], self.page.margin_scale) for card_id in topic.ids})
        self.assertEqual(self.symbols(result), {"101001": "circle", "201002": "circle", model_support.HIDDEN: CROSS_SYMBOL})
        self.assertLess(margins[model_support.HIDDEN], FOCUS_MARGIN)
        cards = self.trace(result, "Cards")
        hover = dict(zip(cards.customdata, cards.text))
        self.assertEqual(hover["101001"], f"101001 · margin {margins['101001']:.3f}")
        self.assertIsNone(self.trace(result, "Connections"))  # the connections stay C's
        self.assertTrue(self.axes_visible(result))
        content = str(result[OUT["content_T"]])
        self.assertIn("T1 · topics", content)
        self.assertIn("margin-histogram", content)
        self.assertIn("moved in by T1", content)  # the legend
        result = self.click(model_support.HIDDEN)
        rows = id_rows(result[OUT["panel"]])
        self.assertEqual(rows["between two topics"], "yes")
        self.assertEqual(rows["margin"], f"{margins[model_support.HIDDEN]:.2f}")
        self.assertEqual(rows["runner-up topic"], Page.topic_label(self.page.topic("01", "02")))  # by label, not by name

    def test_margin_color_runs_from_pale_to_teal(self):
        self.assertEqual(margin_color(0.0, 0.2), "#ccfbf1")
        self.assertEqual(margin_color(0.2, 0.2), "#14b8a6")
        self.assertEqual(margin_color(0.5, 0.2), "#14b8a6")  # above the scale: full
        self.assertEqual(margin_color(0.1, 0.0), "#ccfbf1")  # a course without margins

    def test_a_selected_card_shows_its_ids_not_its_text(self):
        self.enter_topic("01", "01")
        result = self.click("201002")
        self.assertEqual(id_rows(result[OUT["panel"]]), {
            "comb_id": "201002", "deck_id": "01", "topic_id": "01", "stack": "2 · unrated quiz", "difficulty": "–",
            "between two topics": "no",
        })
        cards = self.trace(result, "Cards")
        self.assertEqual(list(cards.text), list(cards.customdata))  # the hover: the comb_id, not the question
        self.assertNotIn("How do you slice a Python list?", str(result))
        self.assertEqual(id_rows(self.click("101004")[OUT["panel"]])["between two topics"], "yes")
        self.click("201002")
        self.assertTrue(result[OUT["open_disabled"]])  # a card is not entered
        self.assertEqual(self.call("dot-panel")[OUT["view"]]["view"], "cards")  # nor by a click on its panel

    def test_back_goes_up_a_level_with_the_dot_it_came_from_selected_and_keeps_the_panel(self):
        self.enter_topic("02", "01")
        self.expand("D")
        self.call("back")
        self.assertEqual(self.state, {**HOME, "view": "topics", "deck": "02", "selected": "02:01", "panel": "D"})
        self.call("back")
        self.assertEqual(self.state, {**HOME, "selected": "02", "panel": "D"})
        result = self.call("back")
        self.assertEqual(self.state, {**HOME, "panel": "D"})
        self.assertTrue(result[OUT["back_disabled"]])

    def test_a_click_on_no_dot_of_the_view_changes_nothing(self):
        self.call(None)
        self.click("101001")  # a card is no dot of the decks view
        self.assertEqual(self.state, HOME)
        self.call("map", {"points": [{"x": 0}]})  # the surface or a connection: no id
        self.assertEqual(self.state, HOME)

    def test_the_scores_live_in_the_d_panel_and_the_difficulty_window_opens_with_it_and_follows_the_view(self):
        result = self.call(None)
        self.assertEqual((result[OUT["window_d"]], result[OUT["class_d"]]), ([], window_class(False)))  # collapsed without D
        result = self.expand("D")
        self.assertIn("Difficulty model D1 · all decks", str(result[OUT["content_D"]]))
        self.assertIn(self.page.difficulty.note, str(result[OUT["content_D"]]))  # the test course is too small for D1
        self.assertEqual(result[OUT["class_d"]], window_class(True))
        self.assertIn("Difficulty model D1 · all decks", str(result[OUT["window_d"]]))
        self.assertIn(self.page.difficulty.note, str(result[OUT["window_d"]]))  # nothing measured: the note, no figure
        self.assertNotIn("Graph", str(result[OUT["window_d"]]))
        self.assertNotIn("color-legend", dumped(result))  # no legend in a panel or a window: the map carries it
        self.click("01")
        result = self.call("open")
        self.assertIn("Difficulty model D1 · deck 01", str(result[OUT["content_D"]]))
        self.assertIn("Difficulty model D1 · deck 01", str(result[OUT["window_d"]]))
        result = self.enter_topic("01", "01")
        self.assertEqual(self.state["panel"], "D")
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(result[OUT["content_D"]]))
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(result[OUT["window_d"]]))
        result = self.expand("S")
        self.assertEqual((result[OUT["window_d"]], result[OUT["class_d"]]), ([], window_class(False)))  # another panel: collapsed

    def test_the_axes_show_with_the_models_positions_only(self):
        for letter, visible in ((None, False), ("S", False), ("C", True), ("T", True), ("D", False)):
            self.state = None
            if letter:
                self.expand(letter)
            for result in (self.call(None), self.click("01"), self.call("open"), self.enter_topic("01", "02")):
                self.assertEqual(self.axes_visible(result), visible, letter)

    def test_the_readout_follows_the_view(self):
        self.assertIn("All 3 decks, 13 cards", str(self.expand("C")[OUT["content_C"]]))
        self.click("01")
        self.assertIn("This deck: 3 topics, 5 cards", str(self.call("open")[OUT["content_C"]]))
        readout = str(self.enter_topic("01", "01")[OUT["content_C"]])
        self.assertIn("This topic: 3 cards", readout)
        self.assertNotIn("agreement", readout)  # a comparison with the scripted topics, not C1's own measure

    def test_the_readout_shows_the_course_models_own_measures(self):
        self.assertEqual(self.page.between, {"101004"})  # the one card T1 placed by a margin below 0.05
        scope, values = readout_values(self.page, {})
        self.assertEqual(values[("Decks (k = 3)", "top-3 overlap")], "1.00")  # three decks: each the others' neighbours
        self.assertEqual(values[("Decks (k = 3)", "connections kept")], "2 of 2")
        self.assertEqual(values[("Topics (T1)", "with cards")], "6 of 7")  # the emptied topic keeps its dot
        self.assertEqual(values[("Topics (T1)", "between two topics")], "1 (8%)")
        self.assertEqual(values[("Cards on their topic's sphere", "top-k overlap")], "–")  # no topic of more than six cards
        scope, values = readout_values(self.page, {"view": "topics", "deck": "01"})
        self.assertEqual(values[("Topics (T1)", "with cards")], "2 of 3")
        self.assertEqual(values[("Topics (T1)", "between two topics")], "1 (20%)")
        self.assertEqual(values[("Topics (T1)", "dot positions")], "none yet")  # at random: not the model's
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "01"})
        self.assertEqual(values[("Cards on the sphere", "neighbours k")], "2 of 2")
        self.assertEqual(values[("Cards on the sphere", "top-k overlap")], "trivial")  # k = n - 1
        self.assertEqual(values[("Cards on the sphere", "trustworthiness")], "–")  # not defined for 2k cards or fewer
        self.assertEqual(values[("Cards on the sphere", "connections kept")], "2 of 2")
        self.assertEqual(values[("Topic (T1)", "between two topics")], "1 of 3")
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "03"})  # the emptied topic
        self.assertEqual((scope, values[("Cards on the sphere", "layout")]), ("This topic: 0 cards", "none"))

    def test_the_s_panel_shows_s1s_live_measures_and_the_hint_without_a_results_file(self):
        result = self.expand("S")
        content = str(result[OUT["content_S"]])
        self.assertIn("S1 · similarity", content)
        self.assertIn("vocabulary", content)
        self.assertIn("No results.csv in", content)  # the hint: how to make one
        self.assertIn("uv run evaluate.py", content)
        self.assertNotIn("Columns", content)

    def test_a_scripted_topics_dendrogram_is_the_clustering_of_its_cards(self):
        result = self.enter_topic("01", "01")  # every panel collapsed: the scripted membership
        self.assertIsNone(self.state["panel"])
        tree = dendrogram(result[OUT["panel"]])
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["101001", "101004", "201002"])
        topic = self.page.entered(self.state)
        self.assertEqual(topic.positions, {})  # the scripted layout has no positions
        self.assertEqual(len(self.page.links_of(topic, self.state)), 2)
        self.assertEqual(len(self.page.card_links(topic, self.state)), 2)  # three cards fit a part of four
        self.assertIn("cut into parts of at most four cards", str(result[OUT["panel"]]))

    def test_a_topics_label_is_its_id_and_a_hash_of_its_name(self):
        lists, commits = self.page.topic("01", "01"), self.page.topic("02", "01")  # the same topic id in two decks
        label = Page.topic_label(lists)
        self.assertRegex(label, r"^01-[0-9a-f]{6}$")
        self.assertEqual(label, Page.topic_label(lists))
        self.assertNotEqual(label, Page.topic_label(commits))  # the names differ, so do the hashes
        self.assertNotIn("Lists", label)
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); from app.models.course_model import TopicLayout; "
            "from app.standalone.app import Page; "
            "print(Page.topic_label(TopicLayout('Alpha', '01', '01', 'Lists', [], {}, [], 1.0, 1.0)))"
        )
        other_run = subprocess.run([sys.executable, "-B", "-c", script, str(model_support.MODULE_DIR)], capture_output=True, text=True)
        self.assertEqual(other_run.stdout.strip(), label, other_run.stderr)  # made at runtime, the same on every run

    def test_no_view_or_panel_shows_learning_material_or_a_name(self):
        """No card's question, answer or explanation, no deck's description and no deck's or topic's name, in any
        view with any dot selected and any panel expanded: not in the panels, the line under the top bar, the map's
        hover nor the dendrogram's leaves."""
        material = [text for card in self.table.cards for text in (card.question, card.answer, card.explain) if text]
        material += [text for deck in self.table.decks for text in (deck.description, deck.name)]
        material += [name for name, _rest in model_support.TOPIC_NAMES.values()]
        outputs = []
        for letter in (None,) + PANEL_LETTERS:
            self.state = None
            outputs.append(self.expand(letter) if letter else self.call(None))
            for deck in self.page.deck_points():
                outputs += [self.click(deck), self.call("open")]
                for dot in self.page.dots(self.state):
                    outputs += [self.click(dot), self.call("open")]
                    outputs += [self.click(card_id) for card_id in self.page.dots(self.state)]
                    self.call("back")
                self.call("back")
        self.assertGreater(len(outputs), 150)
        shown = dumped(outputs)
        for text in material:
            self.assertNotIn(json.dumps(text)[1:-1], shown)

    def test_the_layouts_means_leave_out_the_trivial_overlaps_and_the_undefined_trustworthiness(self):
        def topic(count, overlap, trustworthiness):
            return TopicLayout("Alpha", "01", "01", "Topic", [str(index) for index in range(count)], {}, [], overlap, trustworthiness)

        rows = Page._layout_measures([topic(7, 0.5, math.nan), topic(12, 0.7, 0.9), topic(6, 1.0, math.nan)])
        self.assertEqual({label: value for label, value, _hint in rows}, {"top-k overlap": "0.60 (2 topics)", "trustworthiness": "0.90 (1 topic)"})


class MovingCourseTest(PageDriver):
    """The moving course: T1 moves Gamma's p value (103002) from Intervals (03:01) to Tests (03:02)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.table = load_table(write_test_table(Path(cls._tmp.name), model_support.moving_cards(), model_support.MOVING_SCRIPTED))
        cls.page = Page(cls.table, results_dir=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_membership_follows_the_panel_and_the_view_is_kept(self):
        moved = model_support.MOVED
        result = self.enter_topic("03", "02")  # Tests, scripted: 103003 and 103005
        self.assertEqual(list(self.trace(result, "Cards").customdata), ["103003", "103005"])
        result = self.expand("T")  # T1's membership: the p value moved in
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "03", "topic": "02", "panel": "T"})
        self.assertEqual(set(self.trace(result, "Cards").customdata), {moved, "103003", "103005"})
        self.assertEqual(self.symbols(result)[moved], MOVED_IN_SYMBOL)
        self.assertEqual(self.colors(result)[moved], MOVED_IN_COLOR)
        result = self.click(moved)
        self.assertEqual(self.state["selected"], moved)
        rows = id_rows(result[OUT["panel"]])
        self.assertEqual(rows["scripted topic"], Page.topic_label(self.page.topic("03", "01")))
        self.assertIn("runner-up topic", rows)
        result = self.close("T")  # back to the scripted membership: the view stays, the card that left is unselected
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "03", "topic": "02"})
        self.assertEqual(list(self.trace(result, "Cards").customdata), ["103003", "103005"])
        result = self.click("103003")
        self.expand("C")
        self.assertEqual(self.state["selected"], "103003")  # a card held by both memberships stays selected

    def test_a_moved_out_card_is_an_open_pink_diamond_at_its_similar_members_mean_direction(self):
        moved = model_support.MOVED
        self.enter_topic("03", "01")  # Intervals, scripted: 103001, 103002 and 103004
        result = self.expand("T")
        cards = self.trace(result, "Cards")
        self.assertEqual(set(cards.customdata), {"103001", "103004", moved})  # T1's members, plus the card that left
        self.assertEqual(self.symbols(result)[moved], MOVED_OUT_SYMBOL)
        self.assertEqual(self.colors(result)[moved], MOVED_OUT_COLOR)
        points = self.points(result, "Cards")
        topic = self.page.topic("03", "01")
        self.assertEqual(points["103001"], topic.positions["103001"])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in points[moved])), 1.0)  # on the sphere
        mean = [sum(topic.positions[card_id][axis] for card_id in topic.ids) for axis in range(3)]
        length = math.sqrt(sum(value * value for value in mean))
        for axis in range(3):
            self.assertAlmostEqual(points[moved][axis], mean[axis] / length)  # the mean direction of its two members
        rows = id_rows(self.click(moved)[OUT["panel"]])
        self.assertEqual(rows["T1's topic"], Page.topic_label(self.page.topic("03", "02")))
        self.assertIn("moved out", str(self.expand("T")[OUT["content_T"]]))
        result = self.expand("C")  # with C the card that left is not drawn
        self.assertEqual(set(self.trace(result, "Cards").customdata), {"103001", "103004"})
        self.assertIsNone(self.state["selected"])

    def test_the_topic_model_keeps_the_runner_up_topic(self):
        self.assertEqual(self.page.topic_model.second_of[model_support.MOVED], "01")
        self.assertTrue(all(second != topic for (card_id, second), topic in zip(
            self.page.topic_model.second_of.items(), (self.page.topic_model.topic_of[card_id] for card_id in self.page.topic_model.second_of)) if second))


def write_results(folder: Path, note: str) -> None:
    """Three small results files: an older run without a run column, a 48-deck row without the note, and the newest
    run, with a deck's and a topic's name in its cells."""
    import csv

    folder.mkdir(parents=True, exist_ok=True)

    def write(name, columns, rows):
        with (folder / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})

    columns = ["date", "run", "model", "variant", "params", "seed", "precision_at_5", "kept_precision", "hand_precision",
               "judged_cards", "trust_decks", "overlap_k2", "lift_k2", "seconds", "notes"]
    write("results.csv", columns, [
        {"date": "2026-10-22", "model": "S1", "variant": "chosen: old", "precision_at_5": "0.1", "notes": "the 48-deck course"},
        {"date": "2026-10-26", "model": "S1", "variant": "unigrams", "precision_at_5": "0.4", "notes": f"x; {note}"},
        {"date": "2026-10-26", "model": "S1", "variant": "chosen: uni+bigrams", "precision_at_5": "0.5", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "B0", "variant": "random positions", "seed": "42", "precision_at_5": "0.2", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "B0", "variant": "random positions", "seed": "1", "precision_at_5": "0.4", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S1", "variant": "unigrams", "params": "a", "precision_at_5": "0.5", "kept_precision": "0.55", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S1", "variant": "chosen: uni+bigrams", "params": "b", "precision_at_5": "0.6", "kept_precision": "0.65", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "C1", "variant": "model", "params": "k=3", "trust_decks": "0.8", "overlap_k2": "0.7", "lift_k2": "0.5", "notes": f"deck Alpha and Lists; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "C1", "variant": "random", "params": "k=3", "seed": "42", "trust_decks": "0.4", "overlap_k2": "0.3", "lift_k2": "0.000", "notes": f"deck Alpha and Lists; {note}"},
    ])
    columns = ["date", "run", "model", "scope", "cards", "topics", "agreement", "kept_share", "moved", "notes"]
    write("topics.csv", columns, [
        {"date": "2026-10-26", "model": "T1", "scope": "all", "cards": "13", "agreement": "0.9", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "all", "cards": "13", "topics": "7", "agreement": "1.000", "moved": "0", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "chance", "scope": "all", "cards": "13", "topics": "7", "agreement": "0.400", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "deck 01 Alpha", "cards": "5", "topics": "3", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "deck 01", "cards": "5", "topics": "3", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "chance", "scope": "deck 01", "cards": "5", "topics": "3", "agreement": "0.300", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "stack 1", "cards": "6", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "margin >= 0.05", "cards": "12", "kept_share": "0.923", "agreement": "1.000", "notes": note},
    ])
    columns = ["date", "run", "model", "variant", "features", "label_rule", "folds", "repeats", "seed", "macro_f1", "macro_f1_sd",
               "accuracy", "hand_judged", "hand_within_deck", "hand_within_pairs", "hand_diff", "hand_diff_low", "hand_diff_high", "notes"]
    write("difficulty.csv", columns, [
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "features": "text", "label_rule": "deck shares",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.593", "macro_f1_sd": "0.018", "accuracy": "0.7", "hand_within_deck": "0.75", "hand_within_pairs": "77",
         "hand_diff": "0.000", "hand_diff_low": "0.000", "hand_diff_high": "0.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion", "features": "premise", "label_rule": "deck shares",
         "folds": "0", "repeats": "0", "macro_f1": "0.410", "macro_f1_sd": "0.000", "accuracy": "0.5", "hand_within_deck": "0.80", "hand_within_pairs": "77",
         "hand_diff": "0.052", "hand_diff_low": "-0.020", "hand_diff_high": "0.120", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion", "features": "premise", "label_rule": "most likely class",
         "folds": "0", "repeats": "0", "macro_f1": "0.380", "macro_f1_sd": "0.000", "accuracy": "0.45", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "features": "text", "label_rule": "most likely class",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.530", "macro_f1_sd": "0.017", "accuracy": "0.62", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "baseline", "variant": "always easy", "label_rule": "constant",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.27", "macro_f1_sd": "0.00", "accuracy": "0.667", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "hand", "variant": "the criterion against the authored levels", "label_rule": "authored",
         "hand_judged": "24", "notes": note},
    ])


class ResultsTest(unittest.TestCase):
    """The results files read for the panels: the current course's rows, the newest run, the transposed tables with
    their floor and the chosen variant, the names replaced, the glossary; the hints without them."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls._tmp.name)
        cls.table = load_table(write_test_table(cls.folder / "course"))
        cls.note = course_note(cls.table.deck_ids)
        write_results(cls.folder / "results", cls.note)
        cls.page = Page(cls.table, results_dir=cls.folder / "results")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_note_marks_the_course(self):
        self.assertEqual(self.note, "decks 01 to 03")

    def test_s1_stands_next_to_b0s_mean_in_the_newest_run_with_the_chosen_variant_marked(self):
        (table,) = self.page.results.tables("S")
        self.assertEqual(table.heads, ["B0 random positions (mean of 2 seeds)", "unigrams", "chosen: uni+bigrams"])
        self.assertEqual(table.chosen, [False, False, True])
        self.assertEqual(table.subtitle, "run 2026-11-01T10:00")
        rows = dict(table.rows)
        self.assertEqual(rows["precision_at_5"], ["0.300", "0.5", "0.6"])
        self.assertEqual(rows["kept_precision"], ["", "0.55", "0.65"])
        self.assertNotIn("hand_precision", rows)  # a measure the run left empty is no row
        self.assertEqual(table.measures, ["precision_at_5", "kept_precision"])
        self.assertNotIn("old", dumped(table.heads))  # the 48-deck row and the older run stay out

    def test_c1s_table_replaces_a_decks_and_a_topics_name_and_marks_the_model(self):
        (table,) = self.page.results.tables("C")
        self.assertEqual((table.heads, table.chosen), (["model", "random"], [True, False]))
        self.assertEqual(table.subtitle, "run 2026-11-01T10:00 · params k=3")  # the seed differs: not shared
        self.assertEqual(dict(table.rows)["lift_k2"], ["0.5", "0.000"])
        label = Page.topic_label(self.page.topic("01", "01"))
        self.assertEqual(self.page.results.by_ids("deck Alpha and Lists; x"), f"deck 01 and {label}; x")
        self.assertEqual(self.page.results.by_ids(""), "")

    def test_t1s_tables_are_the_course_or_the_entered_deck_with_the_stacks_and_the_margins(self):
        course, stacks, margins = self.page.results.tables("T")
        self.assertEqual((course.title, course.heads, course.chosen), ("T1 next to the judges, the course", ["T1", "chance"], [True, False]))
        self.assertEqual(dict(course.rows)["agreement"], ["1.000", "0.400"])
        self.assertEqual((stacks.heads, dict(stacks.rows)["cards"]), (["stack 1"], ["6"]))
        self.assertEqual((margins.heads, dict(margins.rows)["kept_share"]), (["margin >= 0.05"], ["0.923"]))
        deck, _stacks, _margins = self.page.results.tables("T", "01")
        self.assertEqual((deck.title, deck.heads), ("T1 next to the judges, deck 01", ["T1", "chance"]))
        self.assertEqual(dict(deck.rows)["agreement"], ["1.000", "0.300"])  # the row "deck 01 Alpha" is no deck scope

    def test_d1s_table_has_a_column_per_row_of_the_run(self):
        table, _tags = self.page.results.tables("D")
        self.assertEqual(table.heads, [
            "text (chosen) · deck shares", "text (chosen) · most likely class", "baseline: always easy",
            "hand labels: the criterion against the authored levels",
        ])  # D2's rows are the second table's
        self.assertEqual(table.chosen, [True, False, False, False])
        rows = dict(table.rows)
        self.assertEqual(rows["macro_f1 ± sd"], ["0.593 ± 0.018", "0.530 ± 0.017", "0.27 ± 0.00", ""])
        self.assertEqual(rows["hand_within_deck (pairs)"], ["0.75 (77)", "", "", ""])
        self.assertEqual(rows["hand_diff [low, high]"], ["0.000 [0.000, 0.000]", "", "", ""])
        self.assertIn("folds 5", table.subtitle)
        self.assertEqual(table.measures[:2], ["macro_f1", "macro_f1_sd"])

    def test_d2s_table_stands_next_to_d1s_chosen_row(self):
        _table, tags = self.page.results.tables("D")
        self.assertEqual(tags.title, "D2 next to D1's chosen row")
        self.assertEqual(tags.heads, ["D1 text (chosen) · deck shares", "D2 criterion · deck shares", "D2 criterion · most likely class"])
        self.assertEqual(tags.chosen, [True, False, False])
        rows = dict(tags.rows)
        self.assertEqual(rows["macro_f1 ± sd"], ["0.593 ± 0.018", "0.410 ± 0.000", "0.380 ± 0.000"])
        self.assertEqual(rows["hand_within_deck (pairs)"], ["0.75 (77)", "0.80 (77)", ""])
        self.assertEqual(rows["hand_diff [low, high]"], ["0.000 [0.000, 0.000]", "0.052 [-0.020, 0.120]", ""])
        self.assertEqual(tags.subtitle, "run 2026-11-01T10:00 · features premise · folds 0 · repeats 0")

    def test_the_glossary_has_a_row_for_every_measure_column_and_none_for_the_bookkeeping(self):
        glossary = read_glossary()
        import evaluate  # next to the package, as the tests run

        bookkeeping = {"date", "run", "model", "variant", "scope", "params", "features", "label_rule", "folds", "repeats", "seed", "seconds", "notes"}
        for columns in (evaluate.COLUMNS, evaluate.TOPICS_COLUMNS, evaluate.DIFFICULTY_COLUMNS):
            for column in columns:
                self.assertEqual(column in glossary, column not in bookkeeping, column)
        self.assertEqual(glossary["lift_k2"], glossary["lift_k3"])  # one row for the columns of one measure
        self.assertEqual(glossary["lift_k2"][0], "lift_k2, lift_k3")
        self.assertIn("(overlap − random) / (1 − random)", glossary["lift_k2"][1])
        table = self.page.results.tables("D")[0]
        self.assertEqual(table.row_columns[0], ["macro_f1", "macro_f1_sd"])  # the columns behind a row, its description's key

    def test_the_panels_show_the_tables_by_ids_and_a_click_on_a_row_opens_its_description(self):
        update = register_callback(RecordingApp(), self.page)
        state = None

        def call(trigger, measure=0):
            nonlocal state
            with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
                result = update(None, *([0] * (INPUTS - 2)), measure, state)
            state = result[OUT["view"]]
            return result

        result = call("panel-S")
        content = dumped(result[OUT["content_S"]])
        self.assertIn("chosen: uni+bigrams", content)
        self.assertIn("B0 random positions (mean of 2 seeds)", content)
        self.assertNotIn("Columns", content)  # no separate row of the columns: each row opens its own description
        self.assertNotIn("measure-description", content)
        keys = measure_keys(result[OUT["content_S"]])
        self.assertEqual(keys, ["S:0:precision_at_5", "S:0:kept_precision"])  # the panel, the table, the row's first column
        result = call(MEASURE_INPUT, measure_value("S:0:precision_at_5"))
        self.assertEqual(state["measure"], "S:0:precision_at_5")
        content = dumped(result[OUT["content_S"]])
        self.assertEqual(content.count("measure-description"), 1)
        self.assertIn(read_glossary()["precision_at_5"][1], content)  # what the measure is, from columns.md
        self.assertIn("Range:", content)
        self.assertIn("measure-close", content)
        self.assertIn("measure-row open", content)
        result = call(MEASURE_INPUT, measure_value("S:0:kept_precision"))  # another row: the description moves
        self.assertEqual(state["measure"], "S:0:kept_precision")
        content = dumped(result[OUT["content_S"]])
        self.assertEqual(content.count("measure-description"), 1)
        self.assertIn(read_glossary()["kept_precision"][1], content)
        self.assertNotIn(read_glossary()["precision_at_5"][1], content)
        call(MEASURE_INPUT, measure_value("S:0:kept_precision"))  # the open row again: closed
        self.assertIsNone(state["measure"])
        call(MEASURE_INPUT, measure_value("S:0:kept_precision"))
        result = call(MEASURE_INPUT, measure_value(None))  # the close box
        self.assertIsNone(state["measure"])
        self.assertNotIn("measure-description", dumped(result[OUT["content_S"]]))
        call(MEASURE_INPUT, measure_value("S:0:kept_precision"))
        result = call("panel-C")  # another panel: the open description goes
        self.assertIsNone(state["measure"])
        content = dumped(result[OUT["content_C"]])
        self.assertNotIn("Alpha", content)
        self.assertNotIn("Lists", content)
        self.assertIn("C:0:lift_k2", measure_keys(result[OUT["content_C"]]))
        result = call(MEASURE_INPUT, measure_value("C:0:lift_k2"))
        self.assertIn("lift_k2, lift_k3", dumped(result[OUT["content_C"]]))  # the glossary's names for the row's columns
        self.assertIsNone(Page.measure_of("not json"))
        self.assertIsNone(Page.measure_of(0))

    def test_without_a_file_or_a_folder_the_section_says_how_to_make_one(self):
        results = Results(self.folder / "nowhere", self.note)
        self.assertIsNone(results.tables("S"))
        self.assertIn("No results.csv in", results.hint("results.csv"))
        self.assertIn("uv run evaluate.py", results.hint("results.csv"))
        self.assertEqual(Results(None, self.note).hint("topics.csv"), NO_FOLDER)


class MeasuredMetricsTest(unittest.TestCase):
    """The page with a measured D1 report (model_support's difficulty course, whose decks are named alike)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.report = model_support.difficulty_report({"201002": "hard"})
        cls.page = Page(load_table(write_test_table(Path(cls._tmp.name))), difficulty=cls.report, results_dir=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def graphs(self, panel):
        return [child.className for child in panel.children if type(child).__name__ == "Graph"]

    def test_the_scores_are_the_d_panels_and_the_matrix_and_the_shares_the_difficulty_windows(self):
        panel = self.page.metrics(HOME)
        self.assertIn("All decks: 36 rated quizzes", str(panel))
        self.assertEqual(self.graphs(panel), [])  # no figure in the panel
        self.assertIn(WINDOW_NOTE, str(panel))  # it says where they are
        (window,), class_name = self.page.difficulty_window({**HOME, "panel": "D"})
        self.assertEqual(class_name, window_class(True))
        self.assertEqual(self.graphs(window), ["difficulty-confusion", "difficulty-shares"])
        self.assertNotIn("color-legend", str(window))  # the legend is the map's
        deck = self.page.metrics({**HOME, "view": "topics", "deck": "01"})
        self.assertIn("This deck: its 12 rated quizzes", str(deck))
        topic = self.page.metrics({**HOME, "view": "cards", "deck": "01", "topic": "01"})
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(topic))
        self.assertIn("This topic:", str(topic))
        (window,), _class = self.page.difficulty_window({**HOME, "view": "cards", "deck": "01", "topic": "01", "panel": "D"})
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(window))

    def test_a_predicted_card_says_so_and_is_a_cross_in_the_predicted_colour_with_d(self):
        panel = self.page.dot_panel({**HOME, "view": "cards", "deck": "01", "topic": "01", "selected": "201002"})
        self.assertEqual(id_rows(panel)["difficulty"], "hard (predicted by D1)")
        state = {**HOME, "view": "cards", "deck": "01", "topic": "01", "panel": "D"}
        from app.standalone.layout import map_figure

        figure = map_figure(self.page, state)
        cards = next(trace for trace in figure.data if trace.name == "Cards")
        colors = dict(zip(cards.customdata, cards.marker.color))
        symbols = dict(zip(cards.customdata, cards.marker.symbol))
        self.assertEqual((colors["201002"], symbols["201002"]), (PREDICTED_COLOR, PREDICTED_SYMBOL))  # as predicted, not as hard
        self.assertEqual((colors["101001"], symbols["101001"]), (GREY, "circle"))  # a flashcard
        self.assertEqual(PREDICTED_SYMBOL, "cross")
        self.assertTrue(figure.layout.showlegend)
        self.assertEqual([trace.name for trace in figure.data if trace.showlegend], ["predicted", "flashcard"])
        self.assertEqual(PREDICTED_COLOR, "#c98500")  # the viewer's predicted group's colour too (test_difficulty_groups.py)


class AppTest(unittest.TestCase):
    def test_the_app_has_one_callback_and_the_pages_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Page(load_table(write_test_table(Path(folder))), results_dir=None)
            app = create_app(page)
        self.assertEqual(len(app.callback_map), 1)
        self.assertTrue(Path(app.config.assets_folder, "style.css").is_file())
        self.assertIn("--map-side", Path(app.config.assets_folder, "map_side.js").read_text(encoding="utf-8"))
        rows = Path(app.config.assets_folder, "measure_rows.js").read_text(encoding="utf-8")
        self.assertIn(f'"{MEASURE_INPUT}"', rows)  # the hidden input the rows' clicks set
        self.assertIn("set_props", rows)
        self.assertIn(".measure-close", rows)

    def test_the_map_is_a_square_with_window_a_the_difficulty_window_and_the_panels_beside_them(self):
        layout = page_layout(HOME)
        self.assertTrue(layout["map"].responsive)  # plotly follows the frame when the window resizes
        style = Path(ASSETS_DIR, "style.css").read_text(encoding="utf-8")
        self.assertIn(".map-frame { grid-column: 1; grid-row: 1; width: var(--map-side); max-width: 100%; aspect-ratio: 1;", style)
        self.assertIn("grid-template-rows: auto auto auto 1fr;", style)  # the last row takes the panels' height: no gap under the map
        self.assertIn(".model-panels { grid-column: 2; grid-row: 1 / span 4;", style)
        self.assertIn(".window-d.collapsed { display: none; }", style)
        self.assertIn("--panel-font: 2;", style)  # the panels' text at double size
        self.assertIn("@media (orientation: portrait)", style)  # window A beside the map, the panels under both
        workspace = layout.children[2]
        self.assertEqual([child.className for child in workspace.children], ["map-frame", "window-a", window_class(False), "model-panels"])
        self.assertEqual(workspace.children[2].id, "difficulty-window")
        self.assertEqual((layout.children[-1].id, layout.children[-1].value), (MEASURE_INPUT, ""))
        boxes = workspace.children[3].children
        self.assertEqual([box.id for box in boxes], [f"box-{letter}" for letter in PANEL_LETTERS])
        self.assertEqual({box.className for box in boxes}, {box_class(False)})
        head = boxes[0].children[0]
        self.assertEqual([button.id for button in head.children], ["panel-S", "close-S"])
        self.assertEqual(head.children[0].children, "S – Similarity")
        self.assertEqual(boxes[0].children[1].id, "content-S")

    def test_main_without_a_table_stops_and_names_the_missing_files(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit) as stop:
            standalone_main(Path(folder))
        self.assertIn("cards.csv", str(stop.exception.code))

    def test_main_serves_the_page_on_the_table_and_the_results_folder(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("dash.Dash.run") as run, mock.patch("builtins.print") as printed:
            standalone_main(write_test_table(Path(folder)), Path(folder) / "results")
        run.assert_called_once_with(debug=False)
        self.assertIn("results", printed.call_args[0][0])

    def test_main_takes_the_port_the_launcher_gives_the_page_next_to_the_viewer(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("dash.Dash.run") as run, mock.patch("builtins.print"):
            standalone_main(write_test_table(Path(folder)), None, port=8051)
        run.assert_called_once_with(debug=False, port=8051)


NO_VIEWER = """
import sys
sys.modules["project"] = None  # the viewer is not there: any import of it fails
sys.path.insert(0, {module_dir!r})
sys.path.insert(0, {tests_dir!r})
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import test_standalone
from app import launch
from app.standalone.app import Page, register_callback
from app.course_table import load_table
with tempfile.TemporaryDirectory() as folder:
    page = Page(load_table(test_standalone.write_test_table(Path(folder))), results_dir=None)
update = register_callback(test_standalone.RecordingApp(), page)
state = None
for trigger, click in ((None, None), ("map", {{"points": [{{"customdata": "01"}}]}}), ("open", None),
                       ("map", {{"points": [{{"customdata": "01:01"}}]}}), ("open", None), ("panel-T", None)):
    with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
        state = update(click, *([0] * (test_standalone.INPUTS - 1)), state)[test_standalone.OUT["view"]]
assert state["view"] == "cards" and state["panel"] == "T", state
assert not launch.viewer_available()
loaded = sorted(name for name in sys.modules if (name == "project" or name.startswith(("project.", "app.glue"))) and sys.modules[name] is not None)
print("LOADED", loaded)
"""


class NoViewerTest(unittest.TestCase):
    def test_the_page_runs_with_the_viewer_blocked_and_imports_none_of_it_or_the_glue(self):
        tests_dir, module_dir = Path(__file__).resolve().parent, model_support.MODULE_DIR
        script = NO_VIEWER.format(module_dir=str(module_dir), tests_dir=str(tests_dir))
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, cwd=str(module_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LOADED []", result.stdout)


TABLE_DIR = model_support.COURSE_TABLE_DIR
RESULTS_DIR = model_support.MODULE_DIR / "results"


@unittest.skipUnless(not missing_files(TABLE_DIR), "the course table is not there (export_course.py writes it)")
class RealTableTest(unittest.TestCase):
    def test_the_real_course_table_serves_the_three_views_and_the_panels(self):
        page = Page(load_table(TABLE_DIR), results_dir=RESULTS_DIR)
        update = register_callback(RecordingApp(), page)
        state = None
        deck = page.table.decks[0].deck_id
        topic = next(topic for topic in page.topics(deck) if len(topic.ids) > 1)
        for trigger, click in (
            (None, None), ("panel-C", None), ("map", {"points": [{"customdata": deck}]}), ("open", None),
            ("map", {"points": [{"customdata": page.topic_dot(topic)}]}), ("open", None), ("panel-T", None),
        ):
            with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
                result = update(click, *([0] * (INPUTS - 1)), state)
            state = result[OUT["view"]]
        self.assertEqual(state, {**HOME, "view": "cards", "deck": deck, "topic": topic.topic_id, "panel": "T"})
        self.assertIn("dendrogram-graph", str(result[OUT["panel"]]))
        self.assertEqual(len(page.deck_links()), 9)  # the decks' tree is cut after 9 of its 23 merges
        self.assertEqual(len(page.model.deck_links), 23)
        self.assertEqual(readout_values(page, {})[1][("Decks (k = 3)", "connections kept")], "9 of 23")
        if (RESULTS_DIR / "topics.csv").is_file():  # a checkout that never ran evaluate.py holds only the hand files
            content = dumped(result[OUT["content_T"]])
            self.assertIn("T1 next to the judges, deck 01", content)  # the entered deck's row, by id
            for deck_row in page.table.decks:
                self.assertNotIn(deck_row.name, content)



class EmbeddingsPanelTest(unittest.TestCase):
    """The S panel with S2's hybrid as the app's vectoriser (the fake encoder), and the S2 table of the results."""

    @classmethod
    def setUpClass(cls):
        from app.models import embeddings, similar

        cls._tmp = tempfile.TemporaryDirectory()
        folder = Path(cls._tmp.name)
        cls.table = load_table(write_test_table(folder / "course"))
        note = course_note(cls.table.deck_ids)
        write_results(folder / "results", note)
        import csv

        path = folder / "results" / "results.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            columns = next(csv.reader(handle))
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writerow({"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S2", "variant": "MiniLM", "params": "m",
                             "precision_at_5": "0.7", "kept_precision": "0.75", "notes": f"x; {note}"})
            writer.writerow({"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S2", "variant": "hybrid: chosen + MiniLM",
                             "params": "h", "precision_at_5": "0.8", "kept_precision": "0.85", "notes": f"x; {note}"})
        hybrid = embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader))
        cls.page = Page(cls.table, results_dir=folder / "results", vectoriser=hybrid, min_score=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_s_panels_live_measures_name_the_hybrid_its_minima_and_weights(self):
        contents, _classes = self.page.panels({**HOME, "panel": "S"})
        content = str(contents[PANEL_LETTERS.index("S")])
        self.assertIn("S2 · similarity", content)
        self.assertIn("hybrid: TF-IDF + MiniLM (16 dimensions)", content)
        for label in ("vectoriser", "vocabulary", "embedding dimensions", "embeddings", "device", "hybrid minima", "hybrid weights", "minimum score"):
            self.assertIn(label, content)
        self.assertIn(f"{self.page.model.index.min_score:.2f}", content)  # the index's own minimum, as the readout rounds it
        self.assertIn("encoded in", content)  # no cache: encoded at the start
        self.assertIn(f"s on {self.page.model.index.vectoriser.second.device}", content)  # and on which device

    def test_s2s_table_stands_next_to_s1s_chosen_variant(self):
        s1, s2 = self.page.results.tables("S")
        self.assertEqual(s1.heads, ["B0 random positions (mean of 2 seeds)", "unigrams", "chosen: uni+bigrams"])
        self.assertEqual(s2.title, "S2 next to S1's chosen variant")
        self.assertEqual(s2.heads, ["S1 chosen: uni+bigrams", "MiniLM", "hybrid: chosen + MiniLM"])
        self.assertEqual(s2.chosen, [True, False, False])
        self.assertEqual(dict(s2.rows)["precision_at_5"], ["0.6", "0.7", "0.8"])
        contents, _classes = self.page.panels({**HOME, "panel": "S"})
        self.assertIn("S2 next to S1", str(contents[PANEL_LETTERS.index("S")]))


if __name__ == "__main__":
    unittest.main()
