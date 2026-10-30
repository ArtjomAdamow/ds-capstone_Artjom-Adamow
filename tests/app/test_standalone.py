"""The standalone page (app.standalone) on the test course as a course table, with nothing of the viewer:
the three views through the callback (select, Open, a click on the panel, Back), the dot's ids in its panel, the
dendrogram with the preset's line and its leaves by id, D1's metrics per view, the course model C1's readout per view
under the square map, the cards coloured by difficulty with no legend in any view, the faint centroid and
sphere, no learning material in any view, and that neither the page nor the table, the figures or the mode question
import the viewer or the glue (a subprocess with the viewer blocked). The real course's table serves the three views
too (skipped without it)."""

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

from app.course_table import flag, load_table, missing_files, write_table
from app.figures import GOLD
from app.models.course_model import TopicLayout
from app.standalone import main as standalone_main
from app.standalone.app import ASSETS_DIR, CARD_PART, DECK_PART, HOME, Page, create_app, register_callback
from app.standalone.layout import EMPTIED_SYMBOL, GREY, LEVEL_COLORS, page_layout

OUT = {"figure": 0, "panel": 1, "metrics": 2, "view": 3, "where": 4, "open_disabled": 5, "back_disabled": 6, "readout": 7}


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


TOPIC_OF = {card_id: key for key, ids in model_support.SCRIPTED.items() for card_id in ids}
DESCRIPTIONS = {"01": "Alpha's description.", "02": "Beta's description.", "03": "Gamma's description."}


def write_test_table(folder: Path) -> Path:
    """The test course of model_support as a course table, as export_course.py writes it from the course file."""
    cards = []
    for card in model_support.CARDS:
        deck_id, topic_id = TOPIC_OF[card.id]
        name, rest = model_support.TOPIC_NAMES[(deck_id, topic_id)]
        cards.append({
            "comb_id": card.id, "deck_id": deck_id, "deck": card.deck, "topic_id": topic_id, "topic_name": name,
            "rest": flag(rest), "stack": card.stack, "difficulty": card.difficulty, "question": card.question,
            "answer": card.answer, "explain": card.explain,
        })
    decks = [{"deck_id": deck_id, "name": name, "description": DESCRIPTIONS[deck_id]} for name, deck_id in model_support.DECK_IDS.items()]
    topics = [
        {"deck_id": deck_id, "topic_id": topic_id, "name": name, "rest": flag(rest)}
        for (deck_id, topic_id), (name, rest) in model_support.TOPIC_NAMES.items()
    ]
    write_table(folder, cards, decks, topics)
    return folder


class RecordingApp:
    def callback(self, *dependencies):
        return lambda function: function


class StandaloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.table = load_table(write_test_table(Path(cls._tmp.name)))
        cls.page = Page(cls.table)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.update = register_callback(RecordingApp(), self.page)
        self.state = None

    def call(self, trigger, click=None):
        with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
            result = self.update(click, 0, 0, 0, self.state)
        self.state = result[OUT["view"]]
        return result

    def click(self, dot):
        return self.call("map", {"points": [{"customdata": dot}]})

    def trace(self, result, name):
        return next((trace for trace in result[OUT["figure"]].data if trace.name == name), None)

    def colors(self, result):
        trace = self.trace(result, "Cards")
        return dict(zip(trace.customdata, trace.marker.color))

    def enter_topic(self, deck_id, topic_id):
        self.call(None)
        self.click(deck_id)
        self.call("open")
        self.click(f"{deck_id}:{topic_id}")
        return self.call("dot-panel")  # a click on the topic's panel enters it as Open does

    def test_the_table_is_the_models_fixture(self):
        self.assertEqual(self.table.cards, model_support.CARDS)
        self.assertEqual(self.table.seeds, model_support.SEEDS)
        self.assertEqual(self.table.deck_ids, model_support.DECK_IDS)

    def test_the_decks_view_shows_the_decks_at_the_models_positions_and_its_connections_below_the_preset(self):
        result = self.call(None)
        self.assertEqual(result[OUT["view"]], HOME)
        self.assertEqual(result[OUT["where"]], "3 decks")
        self.assertEqual((result[OUT["open_disabled"]], result[OUT["back_disabled"]]), (True, True))
        decks = self.trace(result, "Decks")
        self.assertEqual(list(decks.customdata), ["01", "02", "03"])
        self.assertEqual(
            [(x, y, z) for x, y, z in zip(decks.x, decks.y, decks.z)],
            [self.page.model.deck_points[deck_id] for deck_id in ("01", "02", "03")],
        )
        links = self.trace(result, "Connections")
        self.assertEqual((links.line.color, links.line.dash), (GOLD, "dash"))
        self.assertEqual(self.page.deck_links(), self.page.model.deck_links)  # three decks fit into one part of five
        self.assertEqual(list(links.x).count(None), len(self.page.model.deck_links))
        tree = dendrogram(result[OUT["panel"]])
        self.assertEqual(sorted(tree.layout.xaxis.ticktext), ["01", "02", "03"])  # the leaves by id, not by name
        self.assertEqual(DECK_PART, 5)

    def test_a_click_selects_a_deck_and_open_enters_its_topics_on_a_ring(self):
        self.call(None)
        result = self.click("02")
        self.assertEqual(self.state["selected"], "02")
        self.assertFalse(result[OUT["open_disabled"]])
        self.assertEqual(id_rows(result[OUT["panel"]]), {"deck_id": "02", "topics": 2, "cards": 4})
        self.assertNotIn("Beta", str(result[OUT["panel"]]))  # the name stays in the hover and the line under the top bar
        self.assertIsNotNone(self.trace(result, "Selected"))

        result = self.call("open")
        self.assertEqual(self.state, {"view": "topics", "deck": "02", "topic": None, "selected": None})
        self.assertEqual(result[OUT["where"]], "Beta")
        topics = self.trace(result, "Topics")
        self.assertEqual(list(topics.customdata), ["02:01", "02:02"])  # file order
        self.assertEqual(list(topics.text), [Page.topic_label(self.page.topic("02", topic_id)) for topic_id in ("01", "02")])
        for x, y, z in zip(topics.x, topics.y, topics.z):
            self.assertAlmostEqual((x * x + y * y) ** 0.5, 0.75)
            self.assertEqual(z, 0.0)
        centroid = [trace for trace in result[OUT["figure"]].data if trace.name == "Centroid"]
        self.assertEqual([trace.marker.size for trace in centroid], [32, 20, 8])
        self.assertEqual({trace.marker.color for trace in centroid}, {GREY})  # a 3D marker ignores a colour's alpha
        self.assertAlmostEqual(1 - math.prod(1 - trace.marker.opacity for trace in centroid), 0.4)  # at the core
        self.assertIsNone(self.trace(result, "Connections"))
        self.assertIn("No connections between topics", str(result[OUT["panel"]]))

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

    def test_the_cards_view_shows_the_topics_cards_on_its_sphere_coloured_by_difficulty(self):
        result = self.enter_topic("01", "02")  # Dictionaries: the rated quiz and a flashcard of the quiz file
        self.assertEqual(self.state, {"view": "cards", "deck": "01", "topic": "02", "selected": None})
        self.assertEqual(result[OUT["where"]], f"Alpha › {Page.topic_label(self.page.topic('01', '02'))}")
        self.assertEqual(self.trace(result, "Surface").opacity, 0.2)
        self.assertEqual(self.colors(result), {"301003": LEVEL_COLORS["easy"], "401005": GREY})
        topic = self.page.topic("01", "02")
        cards = self.trace(result, "Cards")
        for card_id, x, y, z in zip(cards.customdata, cards.x, cards.y, cards.z):
            self.assertEqual((x, y, z), topic.positions[card_id])
        links = self.trace(result, "Connections")
        self.assertEqual(links.line.dash, "dash")
        self.assertEqual(list(links.x).count(None), 1)  # the topic's one connection
        self.assertEqual(CARD_PART, 4)

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

    def test_back_goes_up_a_level_with_the_dot_it_came_from_selected(self):
        self.enter_topic("02", "01")
        self.call("back")
        self.assertEqual(self.state, {"view": "topics", "deck": "02", "topic": None, "selected": "02:01"})
        self.call("back")
        self.assertEqual(self.state, {**HOME, "selected": "02"})
        result = self.call("back")
        self.assertEqual(self.state, HOME)
        self.assertTrue(result[OUT["back_disabled"]])

    def test_a_click_on_no_dot_of_the_view_changes_nothing(self):
        self.call(None)
        self.click("101001")  # a card is no dot of the decks view
        self.assertEqual(self.state, HOME)
        self.call("map", {"points": [{"x": 0}]})  # the surface or a connection: no id
        self.assertEqual(self.state, HOME)

    def test_the_metrics_follow_the_view(self):
        result = self.call(None)
        self.assertIn("Difficulty model D1 · all decks", str(result[OUT["metrics"]]))
        self.assertIn(self.page.difficulty.note, str(result[OUT["metrics"]]))  # the test course is too small for D1
        self.click("01")
        self.assertIn("Difficulty model D1 · deck 01", str(self.call("open")[OUT["metrics"]]))
        result = self.enter_topic("01", "01")
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(result[OUT["metrics"]]))

    def test_no_view_has_a_difficulty_legend(self):
        for result in (self.call(None), self.click("01"), self.call("open"), self.enter_topic("01", "02")):
            self.assertEqual(len(result), len(OUT))  # no legend among the callback's outputs
            self.assertNotIn("Cards by difficulty", json.dumps(result, cls=PlotlyJSONEncoder))
        self.assertTrue(self.trace(self.enter_topic("01", "02"), "Cards").marker.color)  # the cards keep their colours

    def test_the_readout_follows_the_view(self):
        self.assertIn("All 3 decks, 13 cards", str(self.call(None)[OUT["readout"]]))
        self.click("01")
        self.assertIn("This deck: 3 topics, 5 cards", str(self.call("open")[OUT["readout"]]))
        readout = str(self.enter_topic("01", "01")[OUT["readout"]])
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
        self.assertEqual(values[("Topics (T1)", "dot positions")], "none yet")  # the ring is not the model's
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "01"})
        self.assertEqual(values[("Cards on the sphere", "neighbours k")], "2 of 2")
        self.assertEqual(values[("Cards on the sphere", "top-k overlap")], "trivial")  # k = n - 1
        self.assertEqual(values[("Cards on the sphere", "trustworthiness")], "–")  # not defined for 2k cards or fewer
        self.assertEqual(values[("Cards on the sphere", "connections kept")], "2 of 2")
        self.assertEqual(values[("Topic (T1)", "between two topics")], "1 of 3")
        scope, values = readout_values(self.page, {"view": "cards", "deck": "01", "topic": "03"})  # the emptied topic
        self.assertEqual((scope, values[("Cards on the sphere", "layout")]), ("This topic: 0 cards", "none"))

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

    def test_no_view_shows_learning_material_or_a_topics_name(self):
        """No card's question, answer or explanation, no deck's description and no topic's name, in any view with
        any dot selected: not in the panels, the readout, the line under the top bar, the map's hover nor the
        dendrogram's leaves."""
        material = [text for card in self.table.cards for text in (card.question, card.answer, card.explain) if text]
        material += [deck.description for deck in self.table.decks]
        material += [name for name, _rest in model_support.TOPIC_NAMES.values()]
        outputs = [self.call(None)]
        for deck in self.page.deck_points():
            outputs += [self.click(deck), self.call("open")]
            for dot in self.page.dots(self.state):
                outputs += [self.click(dot), self.call("open")]
                outputs += [self.click(card_id) for card_id in self.page.dots(self.state)]
                self.call("back")
            self.call("back")
        self.assertGreater(len(outputs), 30)
        shown = json.dumps(outputs, cls=PlotlyJSONEncoder)
        for text in material:
            self.assertNotIn(json.dumps(text)[1:-1], shown)

    def test_the_layouts_means_leave_out_the_trivial_overlaps_and_the_undefined_trustworthiness(self):
        def topic(count, overlap, trustworthiness):
            return TopicLayout("Alpha", "01", "01", "Topic", [str(index) for index in range(count)], {}, [], overlap, trustworthiness)

        rows = Page._layout_measures([topic(7, 0.5, math.nan), topic(12, 0.7, 0.9), topic(6, 1.0, math.nan)])
        self.assertEqual({label: value for label, value, _hint in rows}, {"top-k overlap": "0.60 (2 topics)", "trustworthiness": "0.90 (1 topic)"})


class MeasuredMetricsTest(unittest.TestCase):
    """The page with a measured D1 report (model_support's difficulty course, whose decks are named alike)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.report = model_support.difficulty_report({"201002": "hard"})
        cls.page = Page(load_table(write_test_table(Path(cls._tmp.name))), difficulty=cls.report)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def graphs(self, panel):
        return [child.className for child in panel.children if type(child).__name__ == "Graph"]

    def test_the_metrics_panels_draw_the_shared_figures(self):
        panel = self.page.metrics(HOME)
        self.assertIn("All decks: 36 rated quizzes", str(panel))
        self.assertEqual(self.graphs(panel), ["difficulty-confusion", "difficulty-shares"])
        deck = self.page.metrics({"view": "topics", "deck": "01", "topic": None, "selected": None})
        self.assertIn("This deck: its 12 rated quizzes", str(deck))
        topic = self.page.metrics({"view": "cards", "deck": "01", "topic": "01", "selected": None})
        self.assertIn("Difficulty model D1 · deck 01 › topic 01", str(topic))
        self.assertIn("This topic:", str(topic))

    def test_a_predicted_card_says_so_and_stays_grey(self):
        panel = self.page.dot_panel({"view": "cards", "deck": "01", "topic": "01", "selected": "201002"})
        self.assertEqual(id_rows(panel)["difficulty"], "hard (predicted by D1)")
        state = {"view": "cards", "deck": "01", "topic": "01", "selected": None}
        from app.standalone.layout import map_figure

        cards = next(trace for trace in map_figure(self.page, state).data if trace.name == "Cards")
        self.assertEqual(dict(zip(cards.customdata, cards.marker.color))["201002"], GREY)


class AppTest(unittest.TestCase):
    def test_the_app_has_one_callback_and_the_pages_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Page(load_table(write_test_table(Path(folder))))
            app = create_app(page)
        self.assertEqual(len(app.callback_map), 1)
        self.assertTrue(Path(app.config.assets_folder, "style.css").is_file())
        self.assertIn("--map-side", Path(app.config.assets_folder, "map_side.js").read_text(encoding="utf-8"))

    def test_the_map_is_a_square_with_the_readout_under_it_and_no_legend(self):
        layout = page_layout(HOME)
        self.assertTrue(layout["map"].responsive)  # plotly follows the frame when the window resizes
        style = Path(ASSETS_DIR, "style.css").read_text(encoding="utf-8")
        self.assertIn(".map-frame { width: var(--map-side); max-width: 100%; aspect-ratio: 1;", style)
        column = layout.children[2].children[0]
        self.assertEqual([getattr(child, "className", None) for child in column.children], ["map-frame", "readout-panel"])  # no legend

    def test_main_without_a_table_stops_and_names_the_missing_files(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit) as stop:
            standalone_main(Path(folder))
        self.assertIn("cards.csv", str(stop.exception.code))

    def test_main_serves_the_page_on_the_table(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("dash.Dash.run") as run, mock.patch("builtins.print"):
            standalone_main(write_test_table(Path(folder)))
        run.assert_called_once_with(debug=False)


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
    page = Page(load_table(test_standalone.write_test_table(Path(folder))))
update = register_callback(test_standalone.RecordingApp(), page)
state = None
for trigger, click in ((None, None), ("map", {{"points": [{{"customdata": "01"}}]}}), ("open", None),
                       ("map", {{"points": [{{"customdata": "01:01"}}]}}), ("open", None)):
    with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
        state = update(click, 0, 0, 0, state)[3]
assert state["view"] == "cards", state
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


@unittest.skipUnless(not missing_files(TABLE_DIR), "the course table is not there (export_course.py writes it)")
class RealTableTest(unittest.TestCase):
    def test_the_real_course_table_serves_the_three_views(self):
        page = Page(load_table(TABLE_DIR))
        update = register_callback(RecordingApp(), page)
        state = None
        deck = page.table.decks[0].deck_id
        topic = next(topic for topic in page.topics(deck) if len(topic.ids) > 1)
        for trigger, click in (
            (None, None), ("map", {"points": [{"customdata": deck}]}), ("open", None),
            ("map", {"points": [{"customdata": page.topic_dot(topic)}]}), ("open", None),
        ):
            with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
                result = update(click, 0, 0, 0, state)
            state = result[OUT["view"]]
        self.assertEqual(state, {"view": "cards", "deck": deck, "topic": topic.topic_id, "selected": None})
        self.assertIn("dendrogram-graph", str(result[OUT["panel"]]))
        self.assertEqual(len(page.deck_links()), 9)  # the decks' tree is cut after 9 of its 23 merges
        self.assertEqual(len(page.model.deck_links), 23)
        self.assertEqual(readout_values(page, {})[1][("Decks (k = 3)", "connections kept")], "9 of 23")


if __name__ == "__main__":
    unittest.main()
