"""The standalone page (app.standalone) on the test course as a course table, with nothing of the viewer:
the three views through the callback (select, Open, the tag strip's Open, Back), the map by layers (the check boxes'
states for every check of their inputs: on, on its fallback, not functioning; each layer's channel: the positions
and the connections from C, C on the scripted topics with T off, the membership and its markers from T, D's rings,
S's lines, P's target; the bare course with nothing checked), the legend's strip with one group per layer and T's
colour bar along the full height, no axes, the dots' sizes, the tab bar (a tab opened and closed by its second
click, P's among them), the generation tabs with the "app" mark, the section (the line with no tab open; each
generation's two numbers from the results rows, the app's pick marked in S2's table; T's line of the view; P's
request box, a request navigating
the view by ids, the decision line, the three nearest a click away, the box disabled with S off), Details (collapsed,
holding today's content, opening in the figure slot's place, closing on a tab or generation change; the results
tables of the newest run with their floor, a row's description opened by a click, one at a time), the tag strip
(hidden with nothing selected, its Open entering a deck or a topic, a card's tags), the figure slot's figure per tab,
the breadcrumb and the dendrogram's leaves without a topic's hash (the hover keeping it), no learning material and no
deck's or topic's name in any view or region, and that neither the page nor the table, the figures or the mode
question import the viewer or the glue (a subprocess with the viewer blocked). The real course's table serves the
three views too (skipped without it). The results files, the section's numbers and the parts the page shares with
the dashboard are tested on their own: test_results.py, test_headlines.py, test_panels.py and test_figures.py."""

import model_support

import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plotly.utils import PlotlyJSONEncoder

from app import headlines as hl
from app.course_table import load_table, missing_files
from app.figures import GOLD, PREDICTED_COLOR, FONT_SIZE
from app.models.course_model import TopicLayout
from app.models.topic_model import FOCUS_MARGIN
from app.panels import DECK_PART, PANEL_LETTERS
from app.results import course_note, glossary_entry, read_glossary
from app.standalone import main as standalone_main
from app.standalone.app import (
    ASSETS_DIR, CARD_PART, FALLBACK, HOME, NO_ENCODER, NO_TAGS, OFF, ON, ROW_INPUT, Page, create_app, register_callback,
)
from app.standalone.layout import (
    BETWEEN, CROSS_SYMBOL, DISAGREE, DOT_SIZE, EMPTIED, EMPTIED_SYMBOL, GENERATIONS, GREY, GROWN, LEGEND_STRIP, LEGEND_TITLES,
    LEVEL_COLORS, MOVED_IN, MOVED_IN_COLOR, MOVED_IN_SYMBOL, MOVED_OUT, MOVED_OUT_COLOR, MOVED_OUT_SYMBOL, NEIGHBOUR_PAIRS, NOT_RATED,
    PREDICTED, PROMPT_HINT, PROMPT_OFF, PROMPT_PLACEHOLDER, SELECTED_LINE, SELECTED_PAIRS, SELECTED_SIZE, TARGET,
    details_class, faded, layer_options, margin_color, page_layout, prompt_class, tab_class, tag_class,
)
from test_panels import RecordingApp
from test_results import PROMPT_ROWS, write_results

OUT = {"figure": 0, "view": 1, "where": 2, "open_disabled": 3, "back_disabled": 4, "options": 5, "checked": 6}
for _index, _letter in enumerate(PANEL_LETTERS):
    OUT[f"tab_{_letter}"] = 7 + _index
for _index, _name in enumerate(("levels", "head", "live", "prompt_class", "prompt_disabled", "details_class", "tag_body", "tag_class", "slot")):
    OUT[_name] = 7 + len(PANEL_LETTERS) + _index
# The map, the check boxes, Open, Back, the tag strip's Open, the tabs, Details, Go, Enter, the row input
INPUTS = 10 + len(PANEL_LETTERS)


def measure_value(key) -> str:
    """What assets/panel_rows.js writes into the hidden input for a results row: the row's key (None for the close
    box) and a sequence number."""
    measure_value.seq = getattr(measure_value, "seq", 0) + 1
    return json.dumps({"seq": measure_value.seq, "measure": key})


def level_value(level) -> str:
    """What assets/panel_rows.js writes into the hidden input for a generation's tab."""
    measure_value.seq = getattr(measure_value, "seq", 0) + 1
    return json.dumps({"seq": measure_value.seq, "level": level})


def target_value(target) -> str:
    """What assets/panel_rows.js writes into the hidden input for one of P1's targets."""
    measure_value.seq = getattr(measure_value, "seq", 0) + 1
    return json.dumps({"seq": measure_value.seq, "target": target})


def props(component) -> dict:
    return component.to_plotly_json()["props"]


def level_tabs(levels) -> dict:
    """The generation tabs: generation -> whether it is the open one."""
    return {props(tab)["data-level"]: "open" in tab.className for tab in levels}


def app_marked(levels) -> list[str]:
    """The generations whose tab carries the "app" mark."""
    return [props(tab)["data-level"] for tab in levels if any(getattr(child, "className", "") == "app-mark" for child in tab.children)]


def walk(content, attribute: str) -> list[str]:
    """The values of `attribute` (a data attribute) in a region's content, in order."""
    found = []

    def visit(component):
        value = getattr(component, attribute, None)
        if value is not None:
            found.append(value)
        children = getattr(component, "children", None)
        for child in children if isinstance(children, list) else ([children] if children is not None else []):
            if hasattr(child, "children"):
                visit(child)

    for component in content if isinstance(content, list) else [content]:
        visit(component)
    return found


def measure_keys(content) -> list[str]:
    return walk(content, "data-measure")


def targets_of(content) -> list[str]:
    return walk(content, "data-target")


def readout_values(page: Page, state: dict) -> tuple[str, dict]:
    """The readout's scope and its values by (heading, label)."""
    scope, sections = page.measures({**HOME, **state})
    return scope, {(heading, label): value for heading, rows in sections for label, value, _hint in rows}


def tags_of(body) -> tuple[str, list[str]]:
    """The tag strip's title and its tags' texts."""
    if not body:
        return "", []
    title = body[0].children
    return title, [span.children for span in body[1].children]


def tag_classes(body) -> dict[str, str]:
    return {span.children: span.className for span in body[1].children} if body else {}


def numbers_of(head) -> list[dict]:
    """The section's numbers: each block's parts by class."""
    blocks = next((child.children for child in head if getattr(child, "className", "") == "numbers"), [])
    found = []
    for block in blocks:
        parts = {}
        for part in block.children:
            parts[part.className] = part.children
        found.append(parts)
    return found


def slot_graph(slot):
    """The one graph of the figure slot, or None."""
    return next((child for child in slot if type(child).__name__ == "Graph"), None)


def dumped(value) -> str:
    return json.dumps(value, cls=PlotlyJSONEncoder)


# The test course as a course table (model_support writes it, as export_course.py does from the course file)
TOPIC_OF, DESCRIPTIONS, write_test_table = model_support.TOPIC_OF, model_support.DESCRIPTIONS, model_support.write_test_table


class PageDriver(unittest.TestCase):
    """The callback driven by triggers, the view kept between calls."""

    page: Page

    def setUp(self):
        self.update = register_callback(RecordingApp(), self.page)
        self.state = None

    def call(self, trigger, click=None, measure=0, request=None, layers=None):
        with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
            result = self.update(click, layers, *([0] * (INPUTS - 5)), measure, request, self.state)
        self.state = result[OUT["view"]]
        return result

    def click(self, dot):
        return self.call("map", {"points": [{"customdata": dot}]})

    def tab(self, letter):
        return self.call(f"tab-{letter}")

    def check(self, *letters):
        """The check boxes set to these layers."""
        return self.call("layers", layers=list(letters))

    def details(self):
        return self.call("details-toggle")

    def describe(self, key):
        """A click on a results row (its key), or on the open description's close box (None)."""
        return self.call(ROW_INPUT, measure=measure_value(key))

    def generation(self, level):
        """A click on a generation's tab of the open tab."""
        return self.call(ROW_INPUT, measure=level_value(level))

    def trace(self, result, name):
        return next((trace for trace in result[OUT["figure"]].data if trace.name == name), None)

    def legend(self, result) -> list[str]:
        """The legend's strip: its entries' names, in order; none without it."""
        figure = result[OUT["figure"]]
        return [trace.name for trace in figure.data if trace.showlegend] if figure.layout.showlegend else []

    def legend_groups(self, result) -> dict[str, str]:
        """The legend's groups: an entry's name -> its group's title."""
        return {trace.name: trace.legendgrouptitle.text for trace in result[OUT["figure"]].data if trace.showlegend}

    def colorbar(self, result):
        """The colour bar inside the map: (its title, its top, its length); None without it."""
        scale = self.trace(result, "Scale")
        return (scale.marker.colorbar.title.text, scale.marker.cmax, scale.marker.colorbar.len) if scale is not None else None

    def colors(self, result):
        trace = self.trace(result, "Cards")
        return dict(zip(trace.customdata, trace.marker.color))

    def rings(self, result):
        trace = self.trace(result, "Cards")
        colors = trace.marker.line.color
        return dict(zip(trace.customdata, colors if isinstance(colors, (list, tuple)) else [colors] * len(trace.customdata)))

    def symbols(self, result):
        trace = self.trace(result, "Cards")
        symbols = trace.marker.symbol
        return dict(zip(trace.customdata, symbols if isinstance(symbols, (list, tuple)) else [symbols] * len(trace.customdata)))

    def points(self, result, name):
        trace = self.trace(result, name)
        return {dot: (x, y, z) for dot, x, y, z in zip(trace.customdata, trace.x, trace.y, trace.z)}

    def axes_visible(self, result) -> bool:
        scene = result[OUT["figure"]].layout.scene
        return bool(scene.xaxis.visible) or bool(scene.yaxis.visible) or bool(scene.zaxis.visible)

    def tabs(self, result) -> dict:
        return {letter: result[OUT[f"tab_{letter}"]] for letter in PANEL_LETTERS}

    def labels(self, result) -> dict[str, str]:
        """The check boxes' labels by letter."""
        return {option["value"]: option["label"] for option in result[OUT["options"]]}

    def enter_topic(self, deck_id, topic_id):
        self.call(None)
        self.click(deck_id)
        self.call("open")
        self.click(f"{deck_id}:{topic_id}")
        return self.call("tag-open")  # the tag strip's Open enters it as the top bar's does



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

    def test_nothing_checked_is_the_bare_course_at_random_with_no_tab_open(self):
        result = self.call(None)
        self.assertEqual(result[OUT["view"]], HOME)
        self.assertEqual(HOME["layers"], [])
        self.assertFalse(HOME["details"])
        self.assertEqual(result[OUT["where"]], "3 decks")
        self.assertEqual((result[OUT["open_disabled"]], result[OUT["back_disabled"]]), (True, True))
        self.assertEqual(self.tabs(result), {letter: tab_class(False) for letter in PANEL_LETTERS})
        self.assertEqual(result[OUT["levels"]], [])
        self.assertEqual([child.className for child in result[OUT["head"]]], ["section-none"])
        self.assertIn("Five models", result[OUT["head"]][0].children)
        self.assertEqual(result[OUT["live"]], [])
        self.assertEqual(result[OUT["details_class"]], details_class(False, False))  # hidden with no tab
        self.assertEqual(result[OUT["prompt_class"]], prompt_class(False))
        self.assertEqual(result[OUT["tag_class"]], tag_class(False, False))  # hidden with nothing selected
        self.assertEqual(self.labels(result), {letter: letter for letter in PANEL_LETTERS})  # every box plain
        decks = self.trace(result, "Decks")
        self.assertEqual(list(decks.customdata), ["01", "02", "03"])
        self.assertEqual(list(decks.text), ["01", "02", "03"])  # the hover: the deck id, not its name
        self.assertEqual(self.points(result, "Decks"), Page.random_points(["01", "02", "03"]))
        for point in self.points(result, "Decks").values():
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in point)), 1.0)  # on the sphere
        self.assertNotEqual(self.points(result, "Decks")["01"], self.page.model.deck_points["01"])
        self.assertEqual(decks.marker.size, DOT_SIZE)
        self.assertEqual((DOT_SIZE, SELECTED_SIZE), (13, 20))
        self.assertIsNone(self.trace(result, "Connections"))
        self.assertFalse(self.axes_visible(result))
        self.assertEqual(self.legend(result), [])
        self.assertEqual(list(result[OUT["figure"]].layout.scene.domain.y), [0.0, 1.0])  # no strip without a legend
        self.assertEqual(result[OUT["figure"]].layout.hoverlabel.font.size, 16)
        tree = slot_graph(result[OUT["slot"]]).figure  # the slot with no tab open: the dendrogram
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["01", "02", "03"])  # the leaves by id, not by name
        self.assertEqual(DECK_PART, 5)

    def test_the_layers_states_for_every_check_of_their_inputs(self):
        states = Page.layer_states
        self.assertEqual(states([]), {})
        self.assertEqual(states(["S"]), {"S": ON})
        self.assertEqual(states(["D"]), {"D": ON})
        self.assertEqual(states(["T"]), {"T": OFF})  # needs S
        self.assertEqual(states(["S", "T"]), {"S": ON, "T": ON})
        self.assertEqual(states(["C"]), {"C": OFF})
        self.assertEqual(states(["S", "C"]), {"S": ON, "C": FALLBACK})  # T off: the scripted topics
        self.assertEqual(states(["S", "T", "C"]), {"S": ON, "T": ON, "C": ON})
        self.assertEqual(states(["P"]), {"P": OFF})
        self.assertEqual(states(["S", "P"]), {"S": ON, "P": FALLBACK})
        self.assertEqual(states(["S", "T", "P"]), {"S": ON, "T": ON, "P": ON})
        self.assertEqual(states(["S", "C", "T", "D", "P"]), {letter: ON for letter in "SDTCP"})
        self.assertEqual(states(["C", "T", "D", "P"]), {"D": ON, "T": OFF, "C": OFF, "P": OFF})
        options = layer_options(states(["S", "C", "P", "D"]))
        self.assertEqual([option["label"] for option in options], ["S", "C · scripted topics", "T", "D", "P · scripted topics"])
        options = layer_options(states(["C", "T", "P"]))
        self.assertEqual([option["label"] for option in options], ["S", "C · needs S", "T · needs S", "D", "P · needs S"])
        self.assertEqual([option["value"] for option in options], list(PANEL_LETTERS))
        result = self.check("T", "C")
        self.assertEqual(self.state["layers"], ["C", "T"])  # kept in the letters' order
        self.assertEqual(result[OUT["checked"]], ["C", "T"])  # the boxes follow the store
        self.assertEqual(self.labels(result)["T"], "T · needs S")
        self.assertEqual(self.points(result, "Decks"), Page.random_points(["01", "02", "03"]))  # C not functioning draws nothing
        self.assertIsNone(self.trace(result, "Connections"))
        result = self.check("T", "C", "S")
        self.assertEqual(self.labels(result), {letter: letter for letter in PANEL_LETTERS})
        self.assertEqual(self.points(result, "Decks"), {deck_id: self.page.model.deck_points[deck_id] for deck_id in ("01", "02", "03")})

    def test_c_draws_the_positions_and_the_connections_below_the_preset_without_axes(self):
        self.call(None)
        result = self.check("S", "T", "C")
        self.assertEqual(self.page.positions_of(self.state), "model")
        links = self.trace(result, "Connections")
        self.assertEqual((links.line.color, links.line.dash), (GOLD, "dash"))
        self.assertEqual(self.page.deck_links(), self.page.model.deck_links)  # three decks fit into one part of five
        self.assertEqual(list(links.x).count(None), len(self.page.model.deck_links))
        self.assertFalse(self.axes_visible(result))  # no x, y, z axes in any state
        scene = result[OUT["figure"]].layout.scene
        self.assertFalse(scene.xaxis.showspikes)  # the hover stays the dot's id
        self.assertEqual(list(scene.xaxis.range), [-1.15, 1.15])  # the sphere keeps its size

    def test_a_tab_opens_its_model_and_a_second_click_closes_it_p_among_them(self):
        self.call(None)
        self.tab("S")
        result = self.tab("T")
        self.assertEqual(self.state["panel"], "T")
        self.assertEqual(self.tabs(result), {letter: tab_class(letter == "T") for letter in PANEL_LETTERS})
        self.assertEqual(level_tabs(result[OUT["levels"]]), {"T1": True})
        self.assertIn("Sorts each card", result[OUT["head"]][0].children[0].children)
        self.assertEqual(result[OUT["details_class"]], details_class(False, True))  # Details collapsed, its button shown
        result = self.tab("T")  # the open one again: closed
        self.assertIsNone(self.state["panel"])
        self.assertEqual(self.tabs(result), {letter: tab_class(False) for letter in PANEL_LETTERS})
        self.assertEqual(result[OUT["levels"]], [])
        self.assertEqual([child.className for child in result[OUT["head"]]], ["section-none"])
        result = self.tab("P")
        self.assertEqual(self.state["panel"], "P")
        self.assertEqual(level_tabs(result[OUT["levels"]]), {"P1": True})
        self.assertEqual(result[OUT["prompt_class"]], prompt_class(True))  # the request box with P's tab
        self.tab("C")
        self.assertEqual(self.call(None)[OUT["prompt_class"]], prompt_class(False))

    def test_the_layers_hold_while_the_tabs_change_and_the_tabs_hold_the_numbers_alone(self):
        self.call(None)
        self.check("S", "T", "C")
        placed = self.tab("S")
        self.assertEqual(self.state["layers"], ["S", "C", "T"])
        self.assertEqual(self.points(placed, "Decks"), {deck_id: self.page.model.deck_points[deck_id] for deck_id in ("01", "02", "03")})
        self.tab("D")  # another tab: the map is the layers', unchanged
        self.assertEqual(self.points(self.call(None), "Decks"), self.points(placed, "Decks"))
        self.check()
        self.assertEqual(self.points(self.call(None), "Decks"), Page.random_points(["01", "02", "03"]))
        self.assertEqual(self.state["panel"], "D")  # the tab stays open

    def test_a_click_selects_a_deck_and_the_tag_strips_open_enters_its_topics_at_random(self):
        self.call(None)
        result = self.click("02")
        self.assertEqual(self.state["selected"], "02")
        self.assertFalse(result[OUT["open_disabled"]])
        self.assertEqual(result[OUT["tag_class"]], tag_class(True, True))  # shown, with Open
        self.assertEqual(tags_of(result[OUT["tag_body"]]), ("Deck 02", ["2 topics", "4 cards"]))
        self.assertNotIn("Beta", dumped(result))  # nor its name anywhere
        self.assertIsNotNone(self.trace(result, "Selected"))
        self.assertEqual(self.trace(result, "Selected").marker.size, SELECTED_SIZE)

        result = self.call("tag-open")
        self.assertEqual(self.state, {**HOME, "view": "topics", "deck": "02"})
        self.assertEqual(result[OUT["where"]], "Deck 02")  # the breadcrumb, by id
        self.assertEqual(result[OUT["tag_class"]], tag_class(False, False))
        topics = self.trace(result, "Topics")
        self.assertEqual(list(topics.customdata), ["02:01", "02:02"])  # file order
        self.assertEqual(list(topics.text), [Page.topic_label(self.page.topic("02", topic_id)) for topic_id in ("01", "02")])  # the hover: the label with its hash
        for x, y, z in zip(topics.x, topics.y, topics.z):
            self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 1.0)  # at random on the sphere
        self.assertEqual([trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"], [])  # no C: no centroid
        self.assertIsNone(self.trace(result, "Connections"))  # no line on the map: the topics stand at random
        tree = slot_graph(result[OUT["slot"]]).figure  # the deck's tree of its topics, its leaves by id without the hash
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["01", "02"])
        leaves = tree.data[1]
        self.assertEqual(sorted(leaves.text), sorted(Page.topic_label(self.page.topic("02", topic_id)) for topic_id in ("01", "02")))  # the hover keeps it
        self.assertEqual(tree.layout.meta["kept"], 1)  # no preset: the full tree, one merge of two topics
        self.assertEqual(tree.layout.annotations[0].text, "the full tree")  # labelled on the plot
        self.assertNotIn("the full tree", str(result[OUT["slot"]][0].className))  # no subtitle

    def test_the_topics_view_shows_the_centroid_with_c_drawing_and_never_connections(self):
        self.call(None)
        self.click("02")
        self.call("open")
        for layers in (("S", "C"), ("S", "T", "C")):
            result = self.check(*layers)
            centroid = [trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"]
            self.assertEqual([trace.marker.size for trace in centroid], [16, 10, 4], layers)
            self.assertEqual({trace.marker.color for trace in centroid}, {GREY})
            self.assertAlmostEqual(1 - math.prod(1 - trace.marker.opacity for trace in centroid), 0.4)  # at the core
            self.assertIsNone(self.trace(result, "Connections"), layers)
            self.assertEqual(self.points(result, "Topics"), Page.random_points(["02:01", "02:02"]))  # the same draw
        for layers in ((), ("S",), ("D",), ("C",)):
            result = self.check(*layers)
            self.assertEqual([trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"], [], layers)

    def test_an_emptied_topic_is_an_open_circle_and_t_grows_the_uncertain_topic(self):
        self.call(None)
        self.click("01")
        result = self.call("open")
        topics = self.trace(result, "Topics")
        symbols = dict(zip(topics.customdata, topics.marker.symbol))
        self.assertEqual(symbols["01:03"], EMPTIED_SYMBOL)  # "Empty": no cards
        self.assertEqual(symbols["01:01"], "circle")
        self.assertEqual(self.legend(result), [])
        result = self.check("S", "T")
        self.assertEqual(self.legend(result), [EMPTIED, GROWN])  # the emptied topic, the topic grown most
        self.assertEqual(self.legend_groups(result), {EMPTIED: LEGEND_TITLES["T"], GROWN: LEGEND_TITLES["T"]})
        self.assertEqual(LEGEND_TITLES, {"S": "Similar cards", "T": "Topic fit", "D": "Difficulty", "P": "Prompt"})
        self.assertEqual(GROWN.replace("<br>", " "), "larger = more uncertain cards")
        self.assertEqual(list(result[OUT["figure"]].layout.scene.domain.y), [LEGEND_STRIP, 1.0])  # the strip under the sphere
        legend = result[OUT["figure"]].layout.legend
        self.assertEqual((legend.orientation, legend.itemsizing, legend.itemclick, legend.font.size), ("h", "constant", False, 18))
        self.check()
        result = self.click("01:03")
        self.assertEqual(tags_of(result[OUT["tag_body"]]), ("Topic 03", ["0 cards", "0 flashcards", "0 quizzes", "0 rated", "no cards: the model moved every card elsewhere"]))
        result = self.click("01:01")
        self.assertEqual(tags_of(result[OUT["tag_body"]]), ("Topic 01", ["3 cards", "2 flashcards", "1 quizzes", "0 rated"]))
        self.assertEqual(result[OUT["tag_class"]], tag_class(True, True))

    def test_the_cards_view_grey_at_random_ringed_with_d_placed_and_connected_with_c(self):
        result = self.enter_topic("01", "02")  # Dictionaries: the rated quiz and a flashcard of the quiz file
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "01", "topic": "02"})
        self.assertEqual(result[OUT["where"]], "Deck 01 › Topic 02")  # without the hash
        self.assertEqual(self.trace(result, "Surface").opacity, 0.1)
        self.assertEqual(self.colors(result), {"301003": GREY, "401005": GREY})
        self.assertEqual(self.legend(result), [])
        self.assertEqual(self.points(result, "Cards"), Page.random_points(["301003", "401005"]))
        self.assertIsNone(self.trace(result, "Connections"))
        result = self.check("D")  # D draws D1 by default: the rated quiz's ring by its level, the flashcard none
        self.assertEqual(self.colors(result), {"301003": GREY, "401005": GREY})  # the fill stays T's: grey without T
        self.assertEqual(self.rings(result)["301003"], LEVEL_COLORS["easy"])
        self.assertEqual(self.rings(result)["401005"], "rgba(0, 0, 0, 0)")
        self.assertEqual(self.legend(result), ["easy", NOT_RATED])  # the classes the view holds
        self.assertEqual(self.legend_groups(result)["easy"], "Difficulty")
        self.assertEqual(self.points(result, "Cards"), Page.random_points(["301003", "401005"]))  # the same draw
        result = self.check("D", "S", "T", "C")
        topic = self.page.topic("01", "02")
        self.assertEqual(self.points(result, "Cards"), {card_id: topic.positions[card_id] for card_id in topic.ids})
        links = self.trace(result, "Connections")
        self.assertEqual(links.line.dash, "dash")
        self.assertEqual(list(links.x).count(None), 1)  # the topic's one connection
        self.assertEqual(self.rings(result)["301003"], LEVEL_COLORS["easy"])  # D's ring on T's fill
        self.assertNotEqual(self.colors(result)["301003"], GREY)
        self.assertEqual(CARD_PART, 4)

    def test_c_without_t_lays_out_the_scripted_topics_and_without_s_draws_nothing(self):
        self.enter_topic("01", "01")
        result = self.check("S", "C")  # C on its fallback: the scripted topic laid out and clustered
        self.assertEqual(self.page.layer_states(self.state["layers"])["C"], FALLBACK)
        self.assertEqual(self.page.positions_of(self.state), "scripted")
        self.assertEqual(self.labels(result)["C"], "C · scripted topics")
        topic = self.page.entered(self.state)
        self.assertEqual(topic.positions, {})  # the scripted layout itself has no positions
        laid_out = self.page.scripted_layout(topic)
        self.assertEqual(set(laid_out.positions), set(topic.ids))
        self.assertEqual(self.points(result, "Cards"), {card_id: laid_out.positions[card_id] for card_id in topic.ids})
        self.assertIs(self.page.scripted_layout(topic), laid_out)  # fitted once and kept
        self.assertIsNotNone(self.trace(result, "Connections"))
        self.assertEqual(list(self.trace(result, "Connections").x).count(None), 2)  # three cards fit a part of four
        result = self.check("C")  # S off: C does not function
        self.assertEqual(self.page.layer_states(self.state["layers"])["C"], OFF)
        self.assertEqual(self.labels(result)["C"], "C · needs S")
        self.assertEqual(self.points(result, "Cards"), Page.random_points(topic.ids))
        self.assertIsNone(self.trace(result, "Connections"))
        self.assertEqual(self.state["layers"], ["C"])  # it stays checked

    def test_t_colours_the_cards_by_their_margin_and_marks_a_card_between_two_topics_with_a_cross(self):
        self.enter_topic("01", "01")  # Lists: three cards, one of them between Lists and Dictionaries
        result = self.check("S", "T")
        self.assertEqual(self.points(result, "Cards"), Page.random_points(["101001", "201002", "101004"]))  # no C: at random
        topic = self.page.topic("01", "01")
        margins = self.page.topic_model.margin_of
        self.assertEqual(self.colors(result), {card_id: margin_color(margins[card_id], self.page.margin_scale) for card_id in topic.ids})
        self.assertEqual(self.symbols(result), {"101001": "circle", "201002": "circle", model_support.HIDDEN: CROSS_SYMBOL})
        self.assertLess(margins[model_support.HIDDEN], FOCUS_MARGIN)
        cards = self.trace(result, "Cards")
        hover = dict(zip(cards.customdata, cards.text))
        self.assertEqual(hover["101001"], f"101001 · margin {margins['101001']:.3f}")
        self.assertEqual(self.legend(result), [BETWEEN, NEIGHBOUR_PAIRS])  # the classes the view holds, T's then S's lines
        self.assertEqual(self.legend_groups(result), {BETWEEN: "Topic fit", NEIGHBOUR_PAIRS: "Similar cards"})
        self.assertEqual(self.colorbar(result), ("margin", self.page.margin_scale, 1.0 - LEGEND_STRIP))  # the sphere's full height, above the strip
        scale = self.trace(result, "Scale")
        self.assertEqual((scale.marker.colorbar.x, scale.marker.colorbar.y, scale.marker.colorbar.tickfont.size), (1.0, 1.0, 18))
        self.call("map", {"points": [{"customdata": "101001"}]})
        self.assertEqual(self.check("S", "T")[OUT["figure"]].layout.scene.domain.x[1], 0.9)
        self.assertEqual(list(result[OUT["figure"]].layout.scene.domain.x), [0.0, 0.9])  # the scene gives it the right edge
        result = self.check("T")  # S off: T does not function, the membership is the scripted
        self.assertEqual(self.labels(result)["T"], "T · needs S")
        self.assertEqual(self.colors(result), {card_id: GREY for card_id in topic.ids})
        self.assertIsNone(self.colorbar(result))
        self.assertEqual(self.legend(result), [])
        self.check("S", "T")
        result = self.click(model_support.HIDDEN)
        title, tags = tags_of(result[OUT["tag_body"]])
        self.assertEqual(title, f"Card {model_support.HIDDEN}")
        self.assertIn("between two topics", tags)
        self.assertIn(f"fits 01 by {margins[model_support.HIDDEN]:.2f} · runner-up 02", tags)  # by id, not by name
        self.assertEqual(tag_classes(result[OUT["tag_body"]])["between two topics"], "tag t")
        self.assertEqual(result[OUT["tag_class"]], tag_class(True, False))  # a card: no Open

    def test_margin_color_runs_from_pale_to_teal(self):
        self.assertEqual(margin_color(0.0, 0.2), "#ccfbf1")
        self.assertEqual(margin_color(0.2, 0.2), "#14b8a6")
        self.assertEqual(margin_color(0.5, 0.2), "#14b8a6")  # above the scale: full
        self.assertEqual(margin_color(0.1, 0.0), "#ccfbf1")  # a course without margins

    def test_a_selected_card_shows_its_tags_not_its_text(self):
        self.enter_topic("01", "01")
        result = self.click("201002")
        self.assertEqual(tags_of(result[OUT["tag_body"]]), ("Card 201002", ["unrated quiz"]))  # the stack's name, no number
        cards = self.trace(result, "Cards")
        self.assertEqual(list(cards.text), list(cards.customdata))  # the hover: the comb_id, not the question
        self.assertNotIn("How do you slice a Python list?", str(result))
        self.assertIn("between two topics", tags_of(self.click("101004")[OUT["tag_body"]])[1])
        self.click("201002")
        self.assertTrue(result[OUT["open_disabled"]])  # a card is not entered
        self.assertEqual(self.call("tag-open")[OUT["view"]]["view"], "cards")
        self.state = None
        self.enter_topic("01", "02")
        result = self.click("301003")
        self.assertEqual(tags_of(result[OUT["tag_body"]]), ("Card 301003", ["rated quiz", "easy"]))
        self.assertEqual(tag_classes(result[OUT["tag_body"]])["easy"], "tag level easy")  # a coloured tag

    def test_back_goes_up_a_level_with_the_dot_it_came_from_selected_and_keeps_the_tab_and_the_layers(self):
        self.enter_topic("02", "01")
        self.tab("D")
        self.check("S", "D")
        self.call("back")
        self.assertEqual(self.state, {**HOME, "view": "topics", "deck": "02", "selected": "02:01", "panel": "D", "layers": ["S", "D"]})
        self.call("back")
        self.assertEqual(self.state, {**HOME, "selected": "02", "panel": "D", "layers": ["S", "D"]})
        result = self.call("back")
        self.assertEqual(self.state, {**HOME, "panel": "D", "layers": ["S", "D"]})
        self.assertTrue(result[OUT["back_disabled"]])

    def test_a_click_on_no_dot_of_the_view_changes_nothing(self):
        self.call(None)
        self.click("101001")  # a card is no dot of the decks view
        self.assertEqual(self.state, HOME)
        self.call("map", {"points": [{"x": 0}]})  # the surface or a connection: no id
        self.assertEqual(self.state, HOME)

    def test_the_figure_slot_shows_one_figure_per_tab(self):
        result = self.call(None)
        self.assertEqual(slot_graph(result[OUT["slot"]]).className, "dendrogram-graph")  # no tab: the dendrogram
        for letter, expected in (("C", "dendrogram-graph"), ("P", "dendrogram-graph"), ("S", "dendrogram-graph"), ("T", "margin-histogram")):
            self.assertEqual(slot_graph(self.tab(letter)[OUT["slot"]]).className, expected, letter)
        graph = slot_graph(self.call(None)[OUT["slot"]])  # T's tab is open
        self.assertIsNone(graph.figure.layout.height)  # no fixed height: the slot's
        self.assertTrue(graph.responsive)
        self.assertEqual(graph.style, {"height": "100%", "width": "100%"})
        self.assertTrue(graph.figure.layout.annotations[0].text.startswith("between two topics below"))  # the threshold labelled on its line
        result = self.tab("D")  # D2 is the newest: no tags on this page
        self.assertIsNone(slot_graph(result[OUT["slot"]]))
        self.assertIn(NO_TAGS, str(result[OUT["slot"]]))
        result = self.generation("D1")
        self.assertIsNone(slot_graph(result[OUT["slot"]]))  # the test course is too small for D1: its note
        self.assertIn(self.page.difficulty.note, str(result[OUT["slot"]]))
        self.enter_topic("02", "01")
        self.tab("S")
        self.generation("S1")
        result = self.click("102001")
        self.assertEqual(slot_graph(result[OUT["slot"]]).className, "neighbours-graph")  # a card selected with S: its neighbours
        bars = result[OUT["slot"]][0].figure.data
        shown = {card_id for bar in bars for card_id in bar.y}
        self.assertEqual(shown, {other for other, _score in self.page.indexes["S1"].neighbours["102001"]})
        self.assertEqual(slot_graph(self.generation("S2")[OUT["slot"]]), None)  # S2 without the encoder
        self.assertIn(NO_ENCODER, str(self.call(None)[OUT["slot"]]))

    def test_details_is_collapsed_holds_todays_content_and_opens_in_the_slots_place(self):
        self.call(None)
        result = self.tab("C")
        self.assertFalse(self.state["details"])
        self.assertEqual(result[OUT["details_class"]], details_class(False, True))
        self.assertEqual(slot_graph(result[OUT["slot"]]).className, "dendrogram-graph")
        result = self.details()
        self.assertTrue(self.state["details"])
        self.assertEqual(result[OUT["details_class"]], details_class(True, True))
        (section,) = result[OUT["slot"]]
        self.assertEqual(section.className, "level-section details")  # in the slot's place
        content = str(section)
        self.assertIn("The course model", content)  # the description
        self.assertIn("All 3 decks, 13 cards", content)  # C1's readout
        self.assertIn("No results.csv in", content)  # the hint: how to make the file
        self.assertIn("uv run evaluate.py", content)
        self.assertNotIn("dendrogram-graph", content)  # the slot would show it: not twice
        self.assertIn("Select a deck", content)  # the view's hint moved in
        result = self.details()  # closed again: the figure comes back
        self.assertFalse(self.state["details"])
        self.assertEqual(slot_graph(result[OUT["slot"]]).className, "dendrogram-graph")
        self.details()
        self.tab("T")  # a tab change closes Details
        self.assertFalse(self.state["details"])
        self.details()
        self.assertTrue(self.state["details"])
        content = str(self.call(None)[OUT["slot"]])
        self.assertIn("dendrogram-graph", content)  # the slot shows the histogram: the tree moves in
        self.assertNotIn("is in the window", content)  # no sentence that only says where something is
        self.tab("S")
        self.details()
        self.generation("S1")  # a generation change closes Details
        self.assertFalse(self.state["details"])
        self.tab("S")  # closed: Details goes with it
        self.details()
        self.assertFalse(self.state["details"])

    def test_the_t_line_of_the_view_stays_on_top(self):
        self.call(None)
        result = self.tab("T")
        self.assertEqual([child.className for child in result[OUT["live"]]], ["live-line"])
        self.assertEqual(result[OUT["live"]][0].children, "course: 100 % agree with the script")
        self.click("01")
        result = self.call("open")
        self.assertEqual(result[OUT["live"]][0].children, "this deck: 100 % agree, course 100 %")
        result = self.enter_topic("01", "01")
        self.assertEqual(self.state["panel"], "T")
        self.assertEqual(result[OUT["live"]][0].children, "this topic: 100 % agree, course 100 %")
        self.assertEqual(self.tab("C")[OUT["live"]], [])

    def test_the_generation_tabs_open_one_at_a_time_with_the_app_mark(self):
        self.call(None)
        result = self.tab("S")
        self.assertEqual((self.state["level"], Page.level_of(self.state)), (None, "S2"))
        self.assertEqual(list(level_tabs(result[OUT["levels"]]).items()), [("S1", False), ("S2", True)])  # the older first
        self.assertEqual(app_marked(result[OUT["levels"]]), ["S1"])  # this page runs on TF-IDF: S1 is the app's
        self.assertEqual(self.page.app_levels, {"S": "S1", "C": "C1", "T": "T1", "D": "D1", "P": "P1"})
        result = self.generation("S1")
        self.assertEqual(self.state["level"], "S1")
        self.assertEqual(list(level_tabs(result[OUT["levels"]]).items()), [("S1", True), ("S2", False)])
        self.assertIn("by their words", result[OUT["head"]][0].children[0].children)  # S1's headline
        self.generation("S1")  # the open one again: it stays open
        self.assertEqual(self.state["level"], "S1")
        self.generation("D1")  # no generation of this tab: nothing changes
        self.assertEqual(self.state["level"], "S1")
        result = self.tab("D")
        self.assertEqual((self.state["level"], level_tabs(result[OUT["levels"]])), (None, {"D1": False, "D2": True}))
        self.assertEqual(app_marked(result[OUT["levels"]]), ["D1"])
        self.assertEqual(app_marked(self.tab("C")[OUT["levels"]]), ["C1"])
        self.assertEqual(app_marked(self.tab("P")[OUT["levels"]]), ["P1"])
        self.tab("S")
        self.assertEqual(Page.level_of(self.state), "S2")  # a tab opens with its newest again
        self.assertEqual(GENERATIONS, {"S": ("S1", "S2"), "C": ("C1",), "T": ("T1",), "D": ("D1", "D2"), "P": ("P1",)})
        self.assertIsNone(Page.generation_of(measure_value("S1:0:x")))
        self.assertIsNone(Page.measure_of(level_value("S1")))

    def test_s_draws_the_topics_pairs_as_lines_from_the_tabs_generation_else_the_apps(self):
        index = self.page.indexes["S1"]
        self.call(None)
        result = self.check("S")
        self.assertEqual(self.points(result, "Decks"), Page.random_points(["01", "02", "03"]))  # S places nothing
        decks = self.trace(result, "Decks")
        self.assertEqual(set(decks.marker.color), {"#9fb3c8"})  # S's share colours left the map
        self.assertIsNone(self.colorbar(result))
        result = self.enter_topic("02", "01")
        cards = self.trace(result, "Cards")
        topic = self.page.entered(self.state)
        pairs = {frozenset((card_id, other)) for card_id in topic.ids for other, _score in index.neighbours[card_id] if other in topic.ids}
        self.assertTrue(pairs)
        lines = self.trace(result, NEIGHBOUR_PAIRS)
        self.assertEqual(list(lines.x).count(None), len(pairs))  # each pair once, along the surface
        self.assertIsNone(self.trace(result, SELECTED_PAIRS))
        self.assertEqual(self.legend_groups(result), {NEIGHBOUR_PAIRS: "Similar cards"})
        self.assertEqual(set(cards.marker.color), {GREY})  # the fill is T's: grey without T
        self.assertIs(self.page.similarity_of(self.state), index)  # the app's generation: S1 here
        self.tab("S")
        self.assertIs(self.page.similarity_of(self.state), index)  # S2 without the encoder falls back on the app's
        self.generation("S1")
        self.assertIs(self.page.similarity_of(self.state), index)
        selected = sorted(next(iter(pairs)))[0]
        result = self.click(selected)
        own = self.trace(result, SELECTED_PAIRS)
        self.assertEqual(own.line.color, SELECTED_LINE["color"])
        self.assertEqual(list(own.x).count(None), sum(selected in pair for pair in pairs))
        title, tags = tags_of(result[OUT["tag_body"]])
        counts = next(tag for tag in tags if "here" in tag)
        self.assertRegex(counts, r"^\d+ here · \d+ this deck · \d+ elsewhere$")
        self.assertEqual(tag_classes(result[OUT["tag_body"]])[counts], "tag s")
        others = any(selected not in pair for pair in pairs)
        self.assertEqual(self.legend(result), [SELECTED_PAIRS] + ([NEIGHBOUR_PAIRS] if others else []))
        result = self.check()  # S off: no lines, no counts
        self.assertIsNone(self.trace(result, SELECTED_PAIRS))
        self.assertNotIn(counts, tags_of(result[OUT["tag_body"]])[1])

    def test_a_scripted_topics_dendrogram_is_the_clustering_of_its_cards_with_the_cut_labelled(self):
        result = self.enter_topic("01", "01")  # nothing checked: the scripted membership
        self.assertEqual(self.state["layers"], [])
        tree = slot_graph(result[OUT["slot"]]).figure
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["101001", "101004", "201002"])
        topic = self.page.entered(self.state)
        self.assertEqual(topic.positions, {})  # the scripted layout has no positions
        self.assertEqual(len(self.page.links_of(topic, self.state)), 2)
        self.assertEqual(len(self.page.card_links(topic, self.state)), 2)  # three cards fit a part of four
        self.assertEqual(tree.layout.annotations[0].text, "groups of ≤ 4 cards")  # on the plot, in place of the subtitle
        self.assertEqual(tree.layout.yaxis.title.text, "distance")
        self.assertEqual(tree.layout.yaxis.title.font.color, "#f7f2e8")  # in full colour
        self.assertEqual(tree.layout.yaxis.nticks, 4)
        decks = slot_graph(self.call("back")[OUT["slot"]] if self.call("back") else [])  # up to the decks
        self.assertEqual(decks.figure.layout.annotations[0].text, "groups of ≤ 5 decks")

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

    def test_no_view_or_region_shows_learning_material_or_a_name(self):
        """No card's question, answer or explanation, no deck's description and no deck's or topic's name, in any
        view with any dot selected, any tab open with Details and every layer checked: not in the regions, the
        breadcrumb, the map's hover nor the dendrogram's leaves."""
        material = [text for card in self.table.cards for text in (card.question, card.answer, card.explain) if text]
        material += [text for deck in self.table.decks for text in (deck.description, deck.name)]
        material += [name for name, _rest in model_support.TOPIC_NAMES.values()]
        material += [heading for headings, _sources in model_support.TOPIC_TEXTS.values() for heading in headings]
        outputs = []
        for letter, level in [(None, None)] + [(letter, level) for letter in PANEL_LETTERS for level in GENERATIONS[letter]]:
            self.state = None
            outputs.append(self.tab(letter) if letter else self.call(None))
            if level:
                outputs += [self.generation(level), self.details()]
            outputs.append(self.check(*PANEL_LETTERS))
            for deck in self.page.deck_points():
                outputs += [self.click(deck), self.call("open")]
                for dot in self.page.dots(self.state):
                    outputs += [self.click(dot), self.call("open")]
                    outputs += [self.click(card_id) for card_id in self.page.dots(self.state)]
                    self.call("back")
                self.call("back")
        self.assertGreater(len(outputs), 240)
        shown = dumped(outputs)
        for text in material:
            self.assertNotIn(json.dumps(text)[1:-1], shown)

    def test_the_layouts_means_leave_out_the_trivial_overlaps_and_the_undefined_trustworthiness(self):
        def topic(count, overlap, trustworthiness):
            return TopicLayout("Alpha", "01", "01", "Topic", [str(index) for index in range(count)], {}, [], overlap, trustworthiness)

        rows = Page._layout_measures([topic(7, 0.5, math.nan), topic(12, 0.7, 0.9), topic(6, 1.0, math.nan)])
        self.assertEqual({label: value for label, value, _hint in rows}, {"top-k overlap": "0.60 (2 topics)", "trustworthiness": "0.90 (1 topic)"})

    def test_the_readout_follows_the_view(self):
        self.assertEqual(self.page.between, {"101004"})  # the one card T1 placed by a margin below 0.05
        scope, values = readout_values(self.page, {})
        self.assertEqual(values[("Decks (k = 3)", "top-3 overlap")], "1.00")  # three decks: each the others' neighbours
        self.assertEqual(values[("Topics (T1)", "with cards")], "6 of 7")  # the emptied topic keeps its dot
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "01"})
        self.assertEqual(values[("Cards on the sphere", "top-k overlap")], "trivial")  # k = n - 1
        self.assertEqual(values[("Topic (T1)", "between two topics")], "1 of 3")
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "03"})  # the emptied topic
        self.assertEqual((scope, values[("Cards on the sphere", "layout")]), ("This topic: 0 cards", "none"))


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

    def test_the_membership_follows_ts_layer_and_the_view_is_kept(self):
        moved = model_support.MOVED
        result = self.enter_topic("03", "02")  # Tests, scripted: 103003 and 103005
        self.assertEqual(list(self.trace(result, "Cards").customdata), ["103003", "103005"])
        result = self.check("S", "T")  # T1's membership: the p value moved in
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "03", "topic": "02", "layers": ["S", "T"]})
        self.assertEqual(set(self.trace(result, "Cards").customdata), {moved, "103003", "103005"})
        self.assertEqual(self.symbols(result)[moved], MOVED_IN_SYMBOL)
        self.assertEqual(self.colors(result)[moved], MOVED_IN_COLOR)
        self.assertEqual(self.legend(result), [MOVED_IN, NEIGHBOUR_PAIRS])
        result = self.click(moved)
        self.assertEqual(self.state["selected"], moved)
        _title, tags = tags_of(result[OUT["tag_body"]])
        self.assertIn("scripted into 01", tags)
        self.assertTrue(any(tag.startswith("fits 02 by") and "runner-up 01" in tag for tag in tags))
        result = self.check("S")  # back to the scripted membership: the view stays, the card that left is unselected
        self.assertEqual(self.state, {**HOME, "view": "cards", "deck": "03", "topic": "02", "layers": ["S"]})
        self.assertEqual(list(self.trace(result, "Cards").customdata), ["103003", "103005"])
        result = self.click("103003")
        self.check("S", "T", "C")
        self.assertEqual(self.state["selected"], "103003")  # a card held by both memberships stays selected

    def test_a_moved_out_card_is_an_open_pink_diamond_at_its_similar_members_mean_direction(self):
        moved = model_support.MOVED
        self.enter_topic("03", "01")  # Intervals, scripted: 103001, 103002 and 103004
        result = self.check("S", "T", "C")
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
        _title, tags = tags_of(self.click(moved)[OUT["tag_body"]])
        self.assertIn("T1's topic 02", tags)
        self.assertEqual(self.legend(result), [MOVED_OUT, NEIGHBOUR_PAIRS])
        result = self.check("S", "C")  # C on the scripted topics: the card that left is a member again
        self.assertEqual(set(self.trace(result, "Cards").customdata), {"103001", "103004", moved})
        self.assertNotEqual(self.symbols(result)[moved], MOVED_OUT_SYMBOL)  # a plain member
        self.assertEqual(self.state["selected"], moved)

    def test_the_topic_model_keeps_the_runner_up_topic(self):
        self.assertEqual(self.page.topic_model.second_of[model_support.MOVED], "01")



class ResultsTest(PageDriver):
    """The results in the page (the files and the numbers themselves: test_results.py, test_headlines.py): the section
    with the two numbers and Details with the tables whose rows open their description, the scope of the view, a
    deck's and a topic's name replaced by the deck's id and the topic's label, the entered deck's table."""

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











    def test_c1s_table_replaces_a_decks_and_a_topics_name_and_marks_the_model(self):
        (table,) = self.page.results.tables("C1")
        self.assertEqual((table.heads, table.chosen), (["model", "random"], [True, False]))
        self.assertEqual(dict(table.rows)["lift_k2"], ["0.5", "0.000"])
        label = Page.topic_label(self.page.topic("01", "01"))
        self.assertEqual(self.page.results.by_ids("deck Alpha and Lists; x"), f"deck 01 and {label}; x")


    def test_d1s_table_has_a_column_per_row_of_the_run_and_the_entered_decks_rows_a_second_table(self):
        (table,) = self.page.results.tables("D1")
        self.assertEqual(table.heads, [
            "text (chosen) · deck shares", "text (chosen) · most likely class", "baseline: always easy",
            "hand labels: the criterion against the authored levels",
        ])
        self.assertEqual(table.chosen, [True, False, False, False])
        rows = dict(table.rows)
        self.assertEqual(rows["macro_f1 ± sd"], ["0.593 ± 0.018", "0.530 ± 0.017", "0.27 ± 0.00", ""])
        self.assertEqual(rows["hand_within_deck (pairs)"], ["0.75 (77)", "", "", ""])
        self.assertIn("folds 5", table.subtitle)
        course, deck = self.page.results.tables("D1", "01")
        self.assertEqual((deck.title, deck.heads, deck.chosen), ("D1 within deck 01: the hand order", ["text (chosen) · deck shares", "text+options · deck shares"], [True, False]))
        self.assertEqual(len(self.page.results.tables("D1", "02")), 1)  # no rows of deck 02: the course's table alone
        # Details shows the deck's table with a deck entered, and the course's alone in the decks view
        section = str(self.page.details({**HOME, "view": "topics", "deck": "01", "panel": "D", "level": "D1", "details": True}))
        self.assertIn("D1 within deck 01: the hand order", section)
        self.assertNotIn("within deck", str(self.page.details({**HOME, "panel": "D", "level": "D1", "details": True})))


    def test_the_section_shows_the_two_numbers_and_details_the_tables_whose_rows_open_their_description(self):
        result = self.tab("S")  # S2, the newest: no S2 rows in this run
        self.assertIn(self.page.results.no_rows("S2"), dumped(result[OUT["head"]]))
        self.assertEqual(numbers_of(result[OUT["head"]]), [])
        result = self.generation("S1")
        numbers = numbers_of(result[OUT["head"]])
        self.assertEqual([block["number-value"] for block in numbers], ["60 %", "92 %"])
        self.assertEqual([block["number-label"] for block in numbers], ["same-deck precision@5", "judged precision@1"])
        self.assertEqual(numbers[0]["number-contrast-line"][0].children, "against random positions 30 %")
        self.assertEqual(numbers[0]["number-verdict"][0].children, "above the floor by generic criteria")
        self.assertIn("half of the way", numbers[0]["number-verdict"][1].children)
        self.assertEqual(numbers[1]["number-contrast-line"], [])  # a hit rate without a contrast
        self.assertIn("Of each card's five nearest", numbers[0]["number-sentence"])
        variant = next(child for child in result[OUT["head"]] if getattr(child, "className", "") == "variant")
        self.assertEqual(variant.children[1].children, "chosen: uni+bigrams")  # a label without plain words shows as it is
        self.assertNotIn("measure-row", dumped(result[OUT["head"]]))  # the tables are Details'
        result = self.details()
        content = dumped(result[OUT["slot"]])
        self.assertIn("chosen: uni+bigrams", content)
        self.assertIn("B0 random positions (mean of 2 seeds)", content)
        self.assertNotIn("measure-description", content)
        keys = measure_keys(result[OUT["slot"]])
        self.assertEqual(keys, ["S1:0:precision_at_5", "S1:0:kept_precision", "S1:0:hand_precision", "S1:0:hand_p1"])
        result = self.describe("S1:0:precision_at_5")
        self.assertEqual(self.state["measure"], "S1:0:precision_at_5")
        self.assertTrue(self.state["details"])  # a row's click keeps Details open
        content = dumped(result[OUT["slot"]])
        self.assertEqual(content.count("measure-description"), 1)
        self.assertIn(read_glossary()["precision_at_5"][1], content)  # what the measure is, from columns.md
        self.assertIn("Range:", content)
        self.assertIn("measure-row open", content)
        result = self.describe("S1:0:kept_precision")  # another row: the description moves
        self.assertEqual(self.state["measure"], "S1:0:kept_precision")
        content = dumped(result[OUT["slot"]])
        self.assertEqual(content.count("measure-description"), 1)
        self.assertIn(read_glossary()["kept_precision"][1], content)
        self.describe("S1:0:kept_precision")  # the open row again: closed
        self.assertIsNone(self.state["measure"])
        self.describe("S1:0:kept_precision")
        result = self.describe(None)  # the close box
        self.assertIsNone(self.state["measure"])
        self.assertNotIn("measure-description", dumped(result[OUT["slot"]]))
        self.describe("S1:0:kept_precision")
        self.generation("S2")  # another generation: the open description goes, Details closes
        self.assertEqual((self.state["measure"], self.state["details"]), (None, False))
        self.generation("S1")
        self.details()
        self.describe("S1:0:kept_precision")
        result = self.tab("C")  # another tab: the same
        self.assertEqual((self.state["measure"], self.state["details"]), (None, False))
        numbers = numbers_of(result[OUT["head"]])
        self.assertEqual([block["number-value"] for block in numbers], ["0.90", "0.80"])
        result = self.details()
        content = dumped(result[OUT["slot"]])
        self.assertNotIn("Alpha", content)
        self.assertNotIn("Lists", content)
        self.assertIn("C1:0:lift_k2", measure_keys(result[OUT["slot"]]))
        result = self.describe("C1:0:lift_k2")
        self.assertIn("within-topic overlap at k = 2 or 3", dumped(result[OUT["slot"]]))  # the shared row of the two columns
        self.assertIsNone(Page.measure_of("not json"))
        self.assertIsNone(Page.measure_of(0))

    def test_the_sections_scope_is_the_views(self):
        result = self.tab("T")
        head = result[OUT["head"]][0]
        self.assertEqual([child.className for child in head.children], ["headline-text", "headline-scope"])
        self.assertEqual(head.children[1].children, "all decks")
        self.click("01")
        self.assertEqual(self.call("open")[OUT["head"]][0].children[1].children, "deck 01")
        numbers = numbers_of(self.call(None)[OUT["head"]])
        self.assertEqual([block["number-value"] for block in numbers], ["64 %", "89 %"])




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

    def graphs(self, content):
        return [child.className for child in walk_graphs(content)]

    def test_the_matrix_is_the_slots_and_the_scores_with_the_shares_are_details(self):
        state = {**HOME, "panel": "D", "level": "D1"}
        (graph,) = self.page.figure_slot(state)
        self.assertEqual(graph.className, "difficulty-confusion")
        self.assertIsNone(graph.figure.layout.height)
        self.assertEqual(graph.figure.layout.annotations[0].font.size, 18)  # the cells at 18 px
        (details,) = self.page.details({**state, "details": True})
        content = str(details)
        self.assertIn("All decks: 36 rated quizzes", content)
        self.assertIn("difficulty-shares", content)  # the level shares' figure moved in
        self.assertNotIn("difficulty-confusion", content)  # the matrix is the slot's: not twice
        self.assertIn("macro-F1 equals the balanced accuracy", content)
        self.assertNotIn("is in the window", content)  # WINDOW_NOTE went
        self.assertIn("dendrogram-graph", content)  # the tree moved in while the slot shows the matrix
        self.assertNotIn("color-legend", content)  # the legend is the map's
        topic = {**state, "view": "cards", "deck": "01", "topic": "01"}
        self.assertIn("This topic:", str(self.page.details({**topic, "details": True})))
        self.assertEqual(self.page.figure_slot(topic)[0].className, "difficulty-confusion")

    def test_a_predicted_card_gets_the_amber_ring_and_says_so_in_the_tag_strip(self):
        from app.standalone.layout import map_figure

        state = {**HOME, "view": "cards", "deck": "01", "topic": "01", "layers": ["D"]}
        body, _class = self.page.tag_strip({**state, "selected": "201002"})
        self.assertEqual(tags_of(body), ("Card 201002", ["unrated quiz", "hard · predicted by D1"]))
        self.assertEqual(tag_classes(body)["hard · predicted by D1"], "tag level predicted")
        figure = map_figure(self.page, state)
        cards = next(trace for trace in figure.data if trace.name == "Cards")
        rings = dict(zip(cards.customdata, cards.marker.line.color))
        self.assertEqual(rings["201002"], PREDICTED_COLOR)  # as predicted, not as hard
        self.assertEqual(rings["101001"], "rgba(0, 0, 0, 0)")  # a flashcard: no ring
        self.assertEqual(set(cards.marker.color), {GREY})  # the fill is T's
        self.assertEqual(dict(zip(cards.customdata, cards.text))["201002"], "201002 · D1: hard")  # its level on the hover
        self.assertEqual([trace.name for trace in figure.data if trace.showlegend], [PREDICTED, NOT_RATED])  # no ring without D2
        legend = next(trace for trace in figure.data if trace.name == PREDICTED)
        self.assertEqual((legend.marker.color, legend.marker.line.color), (GREY, PREDICTED_COLOR))  # drawn as the dot is
        self.assertEqual(legend.marker.symbol, "circle")  # a circle with the amber ring, as the dot, no cross
        self.assertEqual(PREDICTED_COLOR, "#c98500")  # the viewer's predicted group's colour too (test_difficulty_groups.py)

    def test_a_quiz_the_two_generations_put_at_different_levels_gets_a_ring_and_the_tag_strip_names_both(self):
        from app.standalone.layout import DISAGREE_SYMBOL, map_figure

        table = load_table(write_test_table(Path(self._tmp.name) / "again"))
        tags = {card.id: (0.6, 0.3, 0.1) for card in table.cards if card.stack in (2, 3)}  # D2 reads every quiz easy
        page = Page(table, difficulty=self.report, results_dir=None, tags=tags)
        self.assertEqual(page.reports["D2"].predictions, {"201002": "easy"})  # one extra quiz of a deck: easy; D1 says hard
        self.assertEqual(page.disagreements(["201002", "101001"], page.reports["D1"]), {"201002": "easy"})
        self.assertEqual(self.page.disagreements(["201002"], self.page.reports["D1"]), {})  # the page without tags: no other generation
        state = {**HOME, "view": "cards", "deck": "01", "topic": "01", "layers": ["D"]}
        figure = map_figure(page, state)
        cards = next(trace for trace in figure.data if trace.name == "Cards")
        self.assertEqual(dict(zip(cards.customdata, cards.text))["201002"], "201002 · D1: hard (D2: easy)")  # both levels on the hover
        ring = next(trace for trace in figure.data if trace.name == DISAGREE)
        self.assertEqual((ring.marker.symbol, ring.showlegend, ring.hoverinfo, ring.legendgrouptitle.text), (DISAGREE_SYMBOL, True, "skip", "Difficulty"))
        self.assertEqual([trace.name for trace in figure.data if trace.showlegend], [PREDICTED, NOT_RATED, DISAGREE])
        self.assertEqual(list(figure.data).index(ring) + 1, list(figure.data).index(cards))  # the ring behind the dots
        figure = map_figure(page, {**state, "panel": "D", "level": "D2"})  # D's tab on D2: the layer draws D2
        cards = next(trace for trace in figure.data if trace.name == "Cards")
        self.assertEqual(dict(zip(cards.customdata, cards.text))["201002"], "201002 · D2: easy (D1: hard)")
        body, _class = page.tag_strip({**state, "selected": "201002"})
        self.assertIn("hard · predicted by D1 · D2: easy", tags_of(body)[1])
        body, _class = page.tag_strip({**state, "panel": "D", "level": "D2", "selected": "201002"})
        self.assertIn("easy · predicted by D2 · D1: hard", tags_of(body)[1])
        page.reports["D2"].predictions["201002"] = "hard"
        self.assertFalse(any(trace.name == DISAGREE for trace in map_figure(page, state).data))  # the same level: no ring
        self.assertEqual(page.figure_slot({**state, "panel": "D", "level": "D1"})[0].className, "difficulty-confusion")  # D1's matrix of the part
        self.assertEqual(page.figure_slot({**state, "panel": "D", "level": "D2"})[0].className, "slot-note")  # D2 on this course: too few rated quizzes, its note


def walk_graphs(content) -> list:
    found = []

    def visit(component):
        if type(component).__name__ == "Graph":
            found.append(component)
        children = getattr(component, "children", None)
        for child in children if isinstance(children, list) else ([children] if children is not None else []):
            if hasattr(child, "children") or type(child).__name__ == "Graph":
                visit(child)

    for component in content if isinstance(content, list) else [content]:
        visit(component)
    return found



class PromptPanelTest(PageDriver):
    """P's tab: the request box, a request running P1 and navigating the page's view by ids, the decision line, the
    three nearest a click away, no name in the section, the newest run's numbers and table with the app's variant
    marked, and P's layer: its target on the map, its fallback on the scripted topics, the box disabled with S
    off."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        folder = Path(cls._tmp.name)
        cls.table = load_table(write_test_table(folder))
        (folder / "results").mkdir()
        (folder / "results" / "prompt.csv").write_text(PROMPT_ROWS, encoding="utf-8")
        cls.page = Page(cls.table, results_dir=folder / "results")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def ask(self, text, trigger="prompt-go"):
        return self.call(trigger, request=text)

    def test_ps_tab_shows_the_request_box_the_numbers_and_in_details_the_floors(self):
        layout = page_layout(HOME)
        self.assertEqual(layout["prompt-text"].placeholder, PROMPT_PLACEHOLDER)
        self.assertEqual(layout["prompt-go"].children, "Go")
        self.assertEqual(layout["prompt-box"].className, prompt_class(False))
        result = self.tab("P")
        self.assertEqual(result[OUT["prompt_class"]], prompt_class(True))
        self.assertFalse(result[OUT["prompt_disabled"]])
        self.assertEqual(level_tabs(result[OUT["levels"]]), {"P1": True})
        self.assertIn(PROMPT_HINT, dumped(result[OUT["live"]]))  # nothing asked yet
        numbers = numbers_of(result[OUT["head"]])
        self.assertEqual([block["number-label"] for block in numbers], ["share served right", "share landed wrong"])
        self.assertEqual([block["number-value"] for block in numbers], ["75 %", "20 %"])  # the app's variant: TF-IDF's card route here
        self.assertEqual(numbers[0]["number-contrast-line"], [])  # the page runs on TF-IDF: no contrast
        self.assertEqual(numbers[0]["number-verdict"][0].children, "good by generic criteria")
        self.assertEqual(numbers[1]["number-verdict"][0].children, "good by generic criteria")  # lower better: 20 %
        self.assertIn("the rest (5 %) got no match", numbers[1]["number-sentence"])
        self.assertNotIn("card route floor", dumped(result[OUT["head"]]))  # the floors are Details'
        shown = dumped(self.details()[OUT["slot"]])
        self.assertIn("card route floor", shown)
        self.assertIn(f"{self.page.prompt.floors.card:.3f}", shown)
        style = Path(ASSETS_DIR, "style.css").read_text(encoding="utf-8")
        self.assertIn(".prompt-box.hidden { display: none; }", style)


    def test_a_request_navigates_the_view_into_the_topic_and_selects_the_card_by_ids(self):
        self.tab("P")
        result = self.ask("null hypothesis")
        self.assertEqual((self.state["view"], self.state["deck"], self.state["topic"], self.state["selected"]), ("cards", "03", "02", "103003"))
        self.assertEqual(self.state["prompt"]["text"], "null hypothesis")
        self.assertEqual(self.state["prompt"]["target"], ["03", "02", "103003"])
        self.assertEqual(result[OUT["where"]], "Deck 03 › Topic 02")
        live = result[OUT["live"]]
        shown = dumped(live)
        self.assertRegex(shown, r"card selected \\u00b7 \d+\.\d ms")  # the dump escapes the middle dot
        targets = targets_of(live)
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0], "03:02:103003")  # the decision's own first, its card shown
        label = self.page.label_in_deck("03", "02")
        self.assertIn(json.dumps(f"deck 03 › {label} · card 103003")[1:-1], shown)
        self.assertNotIn("mean", shown)  # every route's score is Details'
        self.assertRegex(dumped(self.details()[OUT["slot"]]), r"card \d\.\d\d \\u00b7 centroid -?\d\.\d\d \\u00b7 text -?\d\.\d\d \\u00b7 mean -?\d\.\d\d")
        self.assertEqual(set(self.trace(result, "Cards").customdata), set(self.page.scripted[("03", "02")].ids))  # no T: the scripted membership

    def test_enter_is_go_and_an_empty_request_changes_nothing(self):
        self.tab("P")
        before = dict(self.ask("   ")[OUT["view"]])
        self.assertIsNone(before["prompt"])
        self.ask("git commit history", trigger="prompt-text")
        self.assertEqual((self.state["view"], self.state["deck"]), ("cards", "02"))

    def test_no_match_moves_nothing_and_lists_the_three_nearest(self):
        self.tab("P")
        result = self.ask("how to bake bread")
        self.assertEqual((self.state["view"], self.state["deck"], self.state["selected"]), ("decks", None, None))
        self.assertIsNone(self.state["prompt"]["target"])
        self.assertIn("no match, the three nearest below the floor", dumped(result[OUT["live"]]))
        self.assertEqual(len(targets_of(result[OUT["live"]])), 3)

    def test_a_click_on_a_target_goes_to_it(self):
        self.tab("P")
        self.ask("how to bake bread")
        self.call(ROW_INPUT, measure=target_value("01:02"))
        self.assertEqual((self.state["view"], self.state["deck"], self.state["topic"], self.state["selected"]), ("cards", "01", "02", None))
        self.call(ROW_INPUT, measure=target_value("03:02:103003"))
        self.assertEqual((self.state["topic"], self.state["selected"]), ("02", "103003"))
        self.call(ROW_INPUT, measure=target_value("09:09"))  # no such topic: nothing moves
        self.assertEqual((self.state["deck"], self.state["topic"]), ("03", "02"))
        self.assertEqual(Page.target_of(target_value("01")), None)
        self.assertEqual(self.state["prompt"]["text"], "how to bake bread")  # the answer stays until the next request

    def test_ps_layer_marks_the_target_answers_on_the_scripted_topics_without_t_and_needs_s(self):
        self.tab("P")
        self.check("S", "T", "P")
        result = self.ask("null hypothesis")
        self.assertEqual(self.page.target_dot(self.state), "103003")
        self.assertEqual(self.legend(result)[-1], TARGET)  # after S's lines of the selected card
        self.assertEqual(self.legend_groups(result)[TARGET], "Prompt")
        target = self.trace(result, TARGET)
        self.assertEqual((list(target.x), list(target.y), list(target.z)), tuple([value] for value in self.points(result, "Cards")["103003"]))
        self.call("back")
        result = self.call(None)
        self.assertEqual(self.page.target_dot(self.state), "03:02")  # the topic in the topics view
        self.assertEqual(self.legend(result), [TARGET])
        self.call("back")
        self.assertEqual(self.page.target_dot(self.state), "03")  # the deck in the decks view
        self.click("01")
        self.call("open")
        self.assertIsNone(self.page.target_dot(self.state))  # another deck: no target in the view
        result = self.check("S", "P")  # T off: P1 on the scripted topics
        self.assertEqual(self.labels(result)["P"], "P · scripted topics")
        fallback = self.page.prompt_of(self.state)
        self.assertIsNot(fallback, self.page.prompt)
        self.assertIs(self.page.prompt_of(self.state), fallback)  # fitted once and kept
        self.assertEqual(fallback.members, [[self.page._row_of[card_id] for card_id in seed.card_ids] for seed in self.page.topic_model.seeds])
        self.assertFalse(result[OUT["prompt_disabled"]])
        result = self.ask("null hypothesis")
        self.assertEqual((self.state["deck"], self.state["topic"], self.state["selected"]), ("03", "02", "103003"))
        result = self.check("P")  # S off: P does not function, the box is disabled
        self.assertEqual(self.labels(result)["P"], "P · needs S")
        self.assertTrue(result[OUT["prompt_disabled"]])
        self.assertIsNone(self.page.prompt_of(self.state))
        self.assertIn(PROMPT_OFF, dumped(result[OUT["live"]]))
        self.assertIsNone(self.trace(result, TARGET))
        before = dict(self.state)
        self.ask("git merge")
        self.assertEqual(self.state, before)  # nothing changes
        result = self.check()  # unchecked: P1 answers as the app's, the layer draws nothing
        self.assertFalse(result[OUT["prompt_disabled"]])
        self.assertIs(self.page.prompt_of(self.state), self.page.prompt)

    def test_no_name_of_a_card_or_a_topic_shows_in_the_section(self):
        material = [text for card in self.table.cards for text in (card.question, card.answer, card.explain) if text]
        material += [text for deck in self.table.decks for text in (deck.description, deck.name)]
        material += [name for name, _rest in model_support.TOPIC_NAMES.values()]
        material += [heading for headings, _sources in model_support.TOPIC_TEXTS.values() for heading in headings]
        self.tab("P")
        shown = dumped([self.ask(text)[OUT["live"]] for text in ("null hypothesis", "git merge", "how to bake bread", "python")])
        for text in material:
            self.assertNotIn(json.dumps(text)[1:-1], shown)

    def test_the_table_is_the_newest_runs_with_the_apps_variant_marked(self):
        (table,) = self.page.results.tables("P1")
        self.assertEqual(table.heads, ["TF-IDF: card", "TF-IDF: all", "hybrid-minilm: card"])
        self.assertEqual(table.chosen, [True, False, False])  # the card route on the app's vectoriser, TF-IDF here
        rows = dict(table.rows)
        self.assertEqual(rows["hand_right"], ["0.75", "0.5", "0.76"])  # the newest run's
        self.assertEqual(table.file, "prompt.csv")
        self.tab("P")
        result = self.details()
        self.assertIn("P1:0:hand_right", measure_keys(result[OUT["slot"]]))
        result = self.describe("P1:0:hand_diff")
        self.assertIn(glossary_entry(read_glossary(), "prompt.csv", "hand_diff")[1], dumped(result[OUT["slot"]]))

    def test_the_glossary_keeps_a_columns_row_per_file(self):
        glossary = read_glossary()
        rows = {name: glossary_entry(glossary, name, "hand_diff")[1] for name in ("results.csv", "difficulty.csv", "prompt.csv")}
        self.assertEqual(len(set(rows.values())), 3)
        self.assertEqual(glossary["hand_diff"], glossary["results.csv:hand_diff"])  # by name alone: the first file's
        self.assertEqual(self.page.results.tables("S1"), None)  # no results.csv here


class AppTest(unittest.TestCase):
    def test_the_app_has_one_callback_and_the_pages_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Page(load_table(write_test_table(Path(folder))), results_dir=None)
            app = create_app(page)
        self.assertEqual(len(app.callback_map), 1)
        self.assertTrue(Path(app.config.assets_folder, "style.css").is_file())
        side = Path(app.config.assets_folder, "map_side.js").read_text(encoding="utf-8")
        self.assertIn("--map-side", side)
        self.assertIn("clientHeight", side)  # the height under the top bar, as 100vh measures it
        self.assertIn("Plots.resize", side)  # plotly's figures follow a change of the side
        self.assertNotIn("MAP_SHARE", side)  # the 12.5 % share went
        rows = Path(app.config.assets_folder, "panel_rows.js").read_text(encoding="utf-8")
        self.assertIn(f'"{ROW_INPUT}"', rows)  # the hidden input the clicks set
        self.assertIn("set_props", rows)
        for selector in (".level-tab", ".measure-row", ".measure-close", ".prompt-target"):
            self.assertIn(selector, rows)
        self.assertNotIn(".level-row", rows)

    def test_the_layout_is_two_columns_with_the_regions_and_the_type_scale(self):
        layout = page_layout(HOME)
        self.assertTrue(layout["map"].responsive)
        self.assertEqual(layout["layers"].value, [])  # nothing checked at load
        self.assertEqual([option["value"] for option in layout["layers"].options], list(PANEL_LETTERS))
        self.assertEqual([tab.id for tab in layout["tabs"].children], [f"tab-{letter}" for letter in PANEL_LETTERS])
        self.assertEqual([child.className for child in layout["tabs"].children[0].children], ["tab-letter", "tab-name"])
        self.assertEqual(layout["tabs"].children[0].children[1].children, "Similarity")
        self.assertEqual([child.id for child in layout["section"].children], ["section-head", "prompt-box", "section-live", "details-toggle"])
        self.assertEqual(layout["details-toggle"].className, details_class(False, False))
        self.assertEqual([child.id for child in layout["tag-strip"].children], ["tag-body", "tag-open"])
        self.assertEqual(layout["tag-strip"].className, tag_class(False, False))
        workspace = layout.children[2]
        self.assertEqual([child.className for child in workspace.children], ["map-frame", "right-column"])
        self.assertEqual([child.id for child in workspace.children[0].children], ["map", "layers"])  # the check boxes inside the map
        self.assertEqual([child.id for child in workspace.children[1].children], ["tabs", "levels", "section", "tag-strip", "figure-slot"])
        self.assertEqual((layout.children[-1].id, layout.children[-1].value), (ROW_INPUT, ""))
        self.assertEqual(layout["where"].className, "breadcrumb")
        style = Path(ASSETS_DIR, "style.css").read_text(encoding="utf-8")
        self.assertIn("grid-template-columns: var(--map-side) minmax(0, 1fr);", style)  # the two columns
        self.assertIn("grid-template-rows: auto auto auto auto minmax(var(--figure-min), 1fr);", style)  # the slot takes the rest
        self.assertIn(".app-shell { height: 100vh;", style)  # no scroll bar at the top level
        self.assertIn("overflow: hidden;", style)
        self.assertIn(".details { flex: 1 1 auto; min-height: 0; overflow-y: auto;", style)  # the one scroll bar
        for token in ("--size-number: 3rem;", "--size-heading: 1.5rem;", "--size-body: 1.25rem;", "--size-details: 1rem;"):
            self.assertIn(token, style)
        self.assertNotIn("--panel-font", style)
        self.assertIn(".breadcrumb {", style)
        self.assertIn("font-size: 1.3rem", style)
        self.assertIn("@media (orientation: portrait), (max-width: 1000px)", style)  # the map at the full width, the column under it
        for size in re.findall(r"(\d*\.?\d+)(rem|px)", style):  # nothing below 16 px
            value, unit = float(size[0]), size[1]
            if unit == "rem":
                self.assertGreaterEqual(value, 1.0, size)
        for match in re.finditer(r"font(?:-size)?:\s*[^;]*?(\d*\.?\d+)px", style):
            self.assertGreaterEqual(float(match.group(1)), 16, match.group(0))
        self.assertNotIn("legend", style[style.index("*/"):])  # every legend is the map's

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

    def test_the_sputnik_link_stands_in_the_top_bar_with_the_viewers_url_and_nowhere_without(self):
        from app.standalone.layout import VIEWER_LINK

        layout = page_layout(HOME)
        actions = layout.children[0].children[1].children
        self.assertEqual([child.id for child in actions], ["back", "open"])  # none without a URL
        layout = page_layout(HOME, "http://127.0.0.1:8050/")
        actions = layout.children[0].children[1].children
        self.assertEqual([child.id for child in actions], ["back", "open", "viewer-link"])
        link = actions[2]
        self.assertEqual((link.children, link.href, link.target, link.rel), (VIEWER_LINK, "http://127.0.0.1:8050/", "_blank", "noopener"))
        self.assertEqual(VIEWER_LINK, "Sputnik")
        with tempfile.TemporaryDirectory() as folder:
            page = Page(load_table(write_test_table(Path(folder))), results_dir=None)
            self.assertNotIn("viewer-link", str(create_app(page).layout))
            app = create_app(page, "http://127.0.0.1:8050/")
            self.assertEqual(app.layout["viewer-link"].href, "http://127.0.0.1:8050/")
            with mock.patch("dash.Dash.run"), mock.patch("builtins.print"), mock.patch("app.standalone.create_app", wraps=create_app) as created:
                standalone_main(write_test_table(Path(folder) / "again"), None, port=8051, viewer_url="http://127.0.0.1:8050/")
            self.assertEqual(created.call_args[0][1], "http://127.0.0.1:8050/")


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
for trigger, click, layers in ((None, None, []), ("map", {{"points": [{{"customdata": "01"}}]}}, []), ("open", None, []),
                               ("map", {{"points": [{{"customdata": "01:01"}}]}}, []), ("open", None, []), ("tab-T", None, []), ("layers", None, ["S", "T"])):
    with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
        state = update(click, layers, *([0] * (test_standalone.INPUTS - 2)), state)[test_standalone.OUT["view"]]
assert state["view"] == "cards" and state["panel"] == "T" and state["layers"] == ["S", "T"], state
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
    def test_the_real_course_table_serves_the_three_views_the_layers_and_the_section(self):
        page = Page(load_table(TABLE_DIR), results_dir=RESULTS_DIR)
        update = register_callback(RecordingApp(), page)
        state = None
        deck = page.table.decks[0].deck_id
        topic = next(topic for topic in page.topics(deck) if len(topic.ids) > 1)
        for trigger, click, layers in (
            (None, None, []), ("layers", None, ["S", "T", "C"]), ("map", {"points": [{"customdata": deck}]}, []), ("open", None, []),
            ("map", {"points": [{"customdata": page.topic_dot(topic)}]}, []), ("open", None, []), ("tab-T", None, []),
        ):
            with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
                result = update(click, layers, *([0] * (INPUTS - 2)), state)
            state = result[OUT["view"]]
        self.assertEqual(state, {**HOME, "view": "cards", "deck": deck, "topic": topic.topic_id, "panel": "T", "layers": ["S", "C", "T"]})
        self.assertEqual(slot_graph(result[OUT["slot"]]).className, "margin-histogram")
        self.assertEqual(len(page.deck_links()), 9)  # the decks' tree is cut after 9 of its 23 merges
        self.assertEqual(len(page.model.deck_links), 23)
        self.assertEqual(readout_values(page, {})[1][("Decks (k = 3)", "connections kept")], "9 of 23")
        if (RESULTS_DIR / "topics.csv").is_file():  # a checkout that never ran evaluate.py holds only the hand files
            numbers = numbers_of(result[OUT["head"]])
            self.assertEqual([block["number-label"] for block in numbers], ["agreement with the script", "agreement of the 25 % most confident"])
            for level in ("S1", "S2", "C1", "T1", "D1", "D2", "P1"):
                found = hl.headlines(page.results, level)
                self.assertEqual(len(found), 2, level)
                for headline in found:
                    self.assertTrue(headline.verdict and headline.verdict != hl.NO_CONTRAST, (level, headline.label))
            for deck_row in page.table.decks:
                self.assertNotIn(deck_row.name, dumped(result[OUT["head"]]))


class EmbeddingsPanelTest(unittest.TestCase):
    """S with S2's hybrid as the app's vectoriser (the fake encoder): the "app" mark on S2, the S2 table with the
    app's variant marked next to the reference, S2's numbers as its gain over S1, and S1 as an index of its own."""

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
                             "precision_at_5": "0.7", "kept_precision": "0.75", "hand_precision": "0.72", "hand_p1": "0.94",
                             "hand_diff": "0.04", "hand_diff_low": "0.00", "hand_diff_high": "0.08", "notes": f"x; {note}"})
            writer.writerow({"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S2", "variant": "hybrid: chosen + MiniLM",
                             "params": "h", "precision_at_5": "0.8", "kept_precision": "0.85", "hand_precision": "0.734", "hand_p1": "0.955",
                             "hand_diff": "0.054", "hand_diff_low": "0.018", "hand_diff_high": "0.093", "notes": f"x; {note}"})
        hybrid = embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader))
        cls.page = Page(cls.table, results_dir=folder / "results", vectoriser=hybrid, min_score=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_s2_is_the_apps_generation_and_its_live_measures_name_the_hybrid(self):
        self.assertEqual(self.page.app_levels["S"], "S2")
        self.assertEqual(self.page.results.chosen["S2"], "hybrid: chosen + MiniLM")
        content = str(self.page.details({**HOME, "panel": "S", "details": True}))
        self.assertIn("hybrid: TF-IDF + MiniLM (16 dimensions)", content)
        self.assertIs(self.page.indexes["S2"], self.page.model.index)
        for label in ("vectoriser", "vocabulary", "embedding dimensions", "embeddings", "device", "hybrid minima", "hybrid weights", "minimum score"):
            self.assertIn(label, content)
        self.assertNotIn("neighbours-graph", content)  # the selected card's neighbours are the slot's alone
        self.assertNotIn("neighbours-graph", str(self.page.details({**HOME, "view": "cards", "deck": "02", "topic": "01", "selected": "102001", "panel": "S", "details": True})))

    def test_s1_is_a_tfidf_index_of_its_own_next_to_the_apps_and_each_generation_draws_its_own(self):
        s1 = self.page.indexes["S1"]
        self.assertIsNot(s1, self.page.model.index)
        self.assertEqual(s1.ids, self.page.model.index.ids)
        state = {**HOME, "view": "cards", "deck": "02", "topic": "01", "layers": ["S"]}
        self.assertIs(self.page.similarity_of(state), self.page.model.index)  # the app's: S2
        self.assertIs(self.page.similarity_of({**state, "panel": "S", "level": "S1"}), s1)  # S's tab on S1: the layer draws S1's
        points = self.page.card_points(self.page.entered(state), state)
        for level in ("S1", "S2"):
            index = self.page.indexes[level]
            pairs, _own = self.page.neighbour_pairs(points, {**state, "panel": "S", "level": level})
            self.assertTrue(all(other in dict(index.neighbours[card_id]) or card_id in dict(index.neighbours[other]) for card_id, other in pairs), level)

    def test_s2s_table_marks_the_apps_variant_next_to_s1s_and_its_numbers_are_its_gain(self):
        (s1,), (s2,) = self.page.results.tables("S1"), self.page.results.tables("S2")
        self.assertEqual(s1.heads, ["B0 random positions (mean of 2 seeds)", "unigrams", "chosen: uni+bigrams"])
        self.assertEqual(s2.heads, ["S1 chosen: uni+bigrams", "MiniLM", "hybrid: chosen + MiniLM"])
        self.assertEqual(s2.chosen, [True, False, True])  # the reference and the app's pick
        self.assertEqual(dict(s2.rows)["precision_at_5"], ["0.6", "0.7", "0.8"])
        precision, p1 = hl.headlines(self.page.results, "S2")
        self.assertEqual((precision.label, precision.shown, precision.contrast_shown, precision.gain), ("judged precision@5", "73 %", "S1 68 %", "+5.4 points (+1.8 to +9.3)"))
        self.assertEqual(precision.verdict, "better")
        self.assertEqual((p1.shown, p1.contrast_shown, p1.gain, p1.verdict), ("96 %", "S1 92 %", "+3.5 points", "very good by generic criteria"))
        self.assertEqual(hl.chosen_label(self.page.results, "S2"), "hybrid: chosen + MiniLM")
        head, _live = self.page.section({**HOME, "panel": "S"})
        variant = next(child for child in head if getattr(child, "className", "") == "variant")
        self.assertEqual(variant.children[1].children, "the sentence encoder MiniLM next to the words")


if __name__ == "__main__":
    unittest.main()
