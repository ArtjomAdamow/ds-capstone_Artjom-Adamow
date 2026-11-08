"""The dashboard (app.dashboard) on the test course as a course table, with nothing of the viewer: the closed start
and a restart with the explored cards and the shown card; what explores a card (Go's answer, a Related entry and a
Go entry when clicked, Back; nothing when Related is drawn); the two sides and the turn marking read, a quiz judged
right and wrong, Confirm disabled with nothing checked, Enter on the back; Related's order with the next card first
and again in its place, and each entry's place; Back on the trail, disabled on an empty one; a request's answer
showing the card, "no match" with its entries, an empty request; the diagram's highlights, its explored set and its
empty start, no click on it; the five tiles on the results fixtures and the line without a file, a tile opening its
panel, a second tile replacing it, the open tile closing it, Enter doing nothing while it is open; the layout's
regions and the stylesheet; the state file; the judgement's functions; and that nothing of the viewer or the glue
is imported (a subprocess with the viewer blocked). The real course's table serves a request too (skipped without
it)."""

import model_support

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_standalone

from app.course_table import load_table, missing_files
from app.dashboard import main as dashboard_main
from app.dashboard import quiz
from app.dashboard.app import ASSETS_DIR, HOME, INPUTS, OUT, OUTPUT_NAMES, Dashboard, create_app, register_callback
from app.dashboard.layout import (
    BACK_HINT, BEGIN_HINT, FLASHCARD_HINT, GO_SLOTS, KEY_INPUT, NEXT_MARK, NO_MATCH, NO_MATCH_NEAREST, PANEL_LETTERS, RELATED_SLOTS,
    THIS_DECK, THIS_TOPIC, page_layout, tile_class,
)
from app.dashboard.state import STATE_NAME, VERSION, DashboardState
from app.neighbours import rank_by_confidence, shorten
from app.standalone.headlines import WHAT
from app.standalone.results import course_note

SLICING = "201002"  # Alpha's quiz: how do you slice a list, option 0 correct
COMMIT = "102001"  # Beta's flashcard: what does git commit do
write_test_table = model_support.write_test_table


def props(component) -> dict:
    return component.to_plotly_json()["props"]


def texts(children) -> list[str]:
    """Every string in a region's content, in order."""
    found: list[str] = []

    def visit(component):
        if isinstance(component, str):
            found.append(component)
            return
        children = getattr(component, "children", None)
        for child in children if isinstance(children, list) else ([children] if children is not None else []):
            visit(child)

    for component in children if isinstance(children, list) else [children]:
        visit(component)
    return found


def dots(diagram) -> list[list[tuple[str, bool]]]:
    """The diagram's three rows: (the dot's id, whether it is the current one) per dot."""
    rows = []
    for row in diagram:
        spans = row.children[1].children
        rows.append([(props(span)["data-dot"], "current" in span.className) for span in spans])
    return rows


def entries(result, prefix: str, slots: int) -> list[tuple[str, list[str]]]:
    """The shown entries of the Related list or of Go's answer: (the slot's class, its texts)."""
    found = []
    for index in range(slots):
        class_name = result[OUT[f"{prefix}_{index}_class"]]
        if "hidden" not in class_name:
            found.append((class_name, texts(result[OUT[f"{prefix}_{index}"]])))
    return found


class Driver(unittest.TestCase):
    """The callback driven by triggers, the view and the checks kept between calls, a fresh state per test."""

    board: Dashboard

    def setUp(self):
        self.board.state = DashboardState()
        self.board.changed = False
        self.update = register_callback(test_standalone.RecordingApp(), self.board)
        self.view = None
        self.checks: list = []

    def call(self, trigger, checks=None, request=None):
        checks = self.checks if checks is None else checks
        with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
            result = self.update(*([0] * INPUTS), checks, request, self.view)
        self.view = result[OUT["view"]]
        self.checks = result[OUT["checks"]]
        return result

    def go(self, text):
        return self.call("prompt-go", request=text)

    def enter(self):
        return self.call(KEY_INPUT)

    def show(self, card_id):
        """A card shown as a selection shows it, then the regions drawn."""
        self.board.state.show(card_id)
        return self.call(None)

    @property
    def state(self) -> DashboardState:
        return self.board.state


class DashboardTest(Driver):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.table = load_table(write_test_table(Path(cls._tmp.name)))
        cls.board = Dashboard(cls.table, results_dir=Path(cls._tmp.name) / "results")  # no results files: the lines

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_closed_start_shows_nothing_but_how_to_begin(self):
        result = self.call(None)
        self.assertTrue(self.state.fresh)
        self.assertEqual(result[OUT["view"]], HOME)
        self.assertEqual(result[OUT["where"]], [])
        self.assertIn(BEGIN_HINT, texts(result[OUT["face"]]))
        self.assertEqual(result[OUT["face_class"]], "card-face empty")
        self.assertEqual(result[OUT["quiz_class"]], "quiz-control hidden")
        self.assertTrue(result[OUT["confirm_disabled"]])
        self.assertTrue(result[OUT["related_disabled"]])
        self.assertEqual(entries(result, "related", RELATED_SLOTS), [])
        self.assertEqual(result[OUT["answer_class"]], "prompt-answer hidden")
        self.assertEqual(dots(result[OUT["diagram"]]), [[], [], []])  # empty, its three rows there
        self.assertEqual([row.children[0].children for row in result[OUT["diagram"]]], ["Decks", "Topics", "Cards"])
        self.assertTrue(result[OUT["back_disabled"]])
        self.assertEqual([result[OUT[f"tile_{letter}"]] for letter in PANEL_LETTERS], [tile_class(False)] * 5)
        self.assertEqual(result[OUT["panel_class"]], "model-panel hidden")
        self.assertEqual(len(result), len(OUTPUT_NAMES))

    def test_a_request_shows_the_card_it_finds_and_explores_it_alone(self):
        found, entries_found = self.board.find("how do I slice a list")
        answer = self.board.prompt.find("how do I slice a list")
        self.assertTrue(answer.navigate)
        self.assertEqual((found, entries_found), (answer.card or answer.ranking[0].card, []))  # the selected card, else the topic's nearest
        self.assertIn(found, model_support.SCRIPTED[("01", "01")])  # a card of Lists
        result = self.go("how do I slice a list")
        self.assertEqual(self.state.shown, found)
        self.assertEqual(self.state.explored_ids, {found})
        self.assertFalse(self.state.is_read(found))  # shown is not turned
        self.assertEqual(texts(result[OUT["where"]]), ["Alpha › Lists"])
        self.assertIn(self.board.card_by_id[found].question, texts(result[OUT["face"]]))
        self.assertTrue(result[OUT["face_class"]].endswith("front"))
        self.assertFalse(result[OUT["related_disabled"]])
        self.assertEqual(result[OUT["answer_class"]], "prompt-answer hidden")  # a match: the card shows, no message
        self.assertEqual(dots(result[OUT["diagram"]]), [[("01", True)], [("01", True)], [(found, True)]])
        self.assertTrue(result[OUT["back_disabled"]])  # nothing shown before
        self.assertEqual(self.view["request"], {"text": "how do I slice a list", "match": True, "entries": []})
        result = self.show(SLICING)  # a quiz: its options and Confirm under the question
        self.assertEqual(result[OUT["face_class"]], "card-face quiz front")
        self.assertEqual(result[OUT["quiz_class"]], "quiz-control")
        self.assertEqual([option["value"] for option in result[OUT["options"]]], [0, 1, 2, 3])
        self.assertEqual([option["label"].children for option in result[OUT["options"]]], list(self.board.card_by_id[SLICING].options))
        self.assertEqual(result[OUT["checks"]], [])
        self.assertTrue(result[OUT["confirm_disabled"]])

    def test_no_match_lists_the_nearest_topics_cards_and_a_click_on_one_shows_and_explores_it(self):
        result = self.go("how to bake bread")
        self.assertIsNone(self.state.shown)
        self.assertEqual(self.state.explored_ids, set())  # nothing explored ahead
        self.assertEqual(result[OUT["answer_class"]], "prompt-answer")
        self.assertEqual(texts(result[OUT["message"]]), [NO_MATCH_NEAREST])
        listed = entries(result, "go", GO_SLOTS)
        self.assertEqual(len(listed), 2)  # Lists and Dictionaries; "Empty" has no card
        self.assertTrue(listed[0][1][0].startswith("Alpha › Lists · "))
        self.assertTrue(listed[1][1][0].startswith("Alpha › Dictionaries · "))
        stored = self.view["request"]["entries"]
        self.assertEqual([entry[:2] for entry in stored], [["01", "01"], ["01", "02"]])
        result = self.call("go-1")
        self.assertEqual(self.state.shown, stored[1][2])
        self.assertEqual(self.state.explored_ids, {stored[1][2]})
        self.assertEqual(result[OUT["answer_class"]], "prompt-answer")  # the entries stay until the next request
        self.assertEqual(len(entries(result, "go", GO_SLOTS)), 2)
        self.call("go-2")  # no such entry: nothing
        self.assertEqual(self.state.shown, stored[1][2])
        self.go("how do I slice a list")
        self.assertEqual(result[OUT["back_disabled"]], True)
        self.assertEqual(self.call(None)[OUT["answer_class"]], "prompt-answer hidden")

    def test_an_empty_request_does_nothing(self):
        self.go("how do I slice a list")
        first = self.state.shown
        for text in ("", "   ", None):
            result = self.go(text)
            self.assertEqual(self.state.shown, first)
            self.assertEqual(result[OUT["answer_class"]], "prompt-answer hidden")
        self.assertEqual(self.view["request"]["text"], "how do I slice a list")
        result = self.call("prompt-text", request="null hypothesis")  # Enter in the box submits too
        self.assertIn(self.state.shown, ("103002", "103003"))
        self.assertEqual(texts(result[OUT["where"]]), ["Gamma › Tests"])

    def test_a_flashcard_turns_by_a_click_or_enter_and_the_turn_marks_it_read(self):
        result = self.show(COMMIT)
        self.assertEqual(result[OUT["face_class"]], "card-face front")
        self.assertEqual(texts(result[OUT["face"]])[0], quiz.QUESTION_LABEL)
        self.assertIn(FLASHCARD_HINT, texts(result[OUT["face"]]))
        self.assertEqual(result[OUT["quiz_class"]], "quiz-control hidden")
        self.assertFalse(self.state.is_read(COMMIT))
        result = self.call("card-face")
        self.assertEqual(result[OUT["face_class"]], "card-face back")
        self.assertEqual(texts(result[OUT["face"]])[0], quiz.ANSWER_LABEL)
        self.assertIn("A git commit records a snapshot", str(result[OUT["face"]]))
        self.assertIn(BACK_HINT, texts(result[OUT["face"]]))
        self.assertTrue(self.state.is_read(COMMIT))
        result = self.enter()
        self.assertEqual(result[OUT["face_class"]], "card-face front")
        self.assertTrue(self.state.is_read(COMMIT))  # read stays
        self.assertEqual(self.enter()[OUT["face_class"]], "card-face back")
        self.assertEqual(self.state.explored_ids, {COMMIT})  # the turn explores nothing new

    def test_a_quiz_is_judged_by_confirm_or_enter_right_only_with_exactly_the_correct_options(self):
        result = self.show(SLICING)
        self.assertTrue(result[OUT["confirm_disabled"]])
        result = self.call("quiz-confirm", checks=[])  # nothing checked: no answer
        self.assertEqual(result[OUT["face_class"]], "card-face quiz front")
        self.assertFalse(self.state.is_read(SLICING))
        result = self.call("card-face", checks=[1])  # a click on a quiz's front does nothing
        self.assertEqual(result[OUT["face_class"]], "card-face quiz front")
        self.assertFalse(result[OUT["confirm_disabled"]])
        result = self.call("quiz-confirm", checks=[1])
        self.assertEqual(result[OUT["face_class"]], "card-face quiz back")
        shown = texts(result[OUT["face"]])
        self.assertEqual(shown[0], quiz.WRONG_LABEL)
        self.assertIn("list.slice(start, stop)", str(result[OUT["face"]]))
        self.assertIn("Lists have no slice method.", str(result[OUT["face"]]))
        self.assertNotIn("slice makes a slice object", str(result[OUT["face"]]))  # only the checked options
        self.assertEqual(result[OUT["quiz_class"]], "quiz-control hidden")
        self.assertTrue(self.state.is_read(SLICING))
        result = self.enter()  # Enter on the back: the front with the checks cleared
        self.assertEqual(result[OUT["face_class"]], "card-face quiz front")
        self.assertEqual(result[OUT["checks"]], [])
        self.assertTrue(result[OUT["confirm_disabled"]])
        result = self.call(KEY_INPUT, checks=[0, 1])  # a wrong one next to the correct one: wrong
        self.assertEqual(texts(result[OUT["face"]])[0], quiz.WRONG_LABEL)
        blocks = result[OUT["face"]][1].children
        self.assertEqual([block.className for block in blocks], ["quiz-feedback-block quiz-feedback-correct", "quiz-feedback-block quiz-feedback-wrong"])
        self.enter()
        result = self.call(KEY_INPUT, checks=[0])
        self.assertEqual(texts(result[OUT["face"]])[0], quiz.CORRECT_LABEL)
        self.assertIn("Slicing a list with start and stop indices returns a new list.", str(result[OUT["face"]]))
        self.assertEqual(self.call(KEY_INPUT, checks=[])[OUT["face_class"]], "card-face quiz front")
        self.assertEqual(self.call(KEY_INPUT, checks=[])[OUT["face_class"]], "card-face quiz front")  # Enter with nothing checked

    def test_related_lists_the_next_card_first_and_again_in_its_place_with_where_each_lies(self):
        result = self.show(COMMIT)
        self.assertEqual(result[OUT["related_class"]], "related-toggle")
        self.assertEqual(entries(result, "related", RELATED_SLOTS), [])
        result = self.call("related-toggle")
        self.assertEqual(result[OUT["related_class"]], "related-toggle open")
        neighbours = self.board.neighbours(COMMIT)
        self.assertTrue(neighbours)
        self.assertEqual([other for other, _score in self.board.model.related(COMMIT)], neighbours)  # the index's order
        listed = entries(result, "related", RELATED_SLOTS)
        self.assertEqual(len(listed), len(neighbours) + 1)
        self.assertEqual(listed[0][0], "related-entry marked")
        self.assertEqual(listed[0][1][0], NEXT_MARK)
        self.assertEqual([class_name for class_name, _texts in listed[1:]], ["related-entry"] * len(neighbours))
        self.assertEqual(listed[0][1][1:], next(found for found in listed[1:] if found[1] == listed[0][1][1:])[1])  # again in its place
        places = {other: self.board.place(COMMIT, other) for other in neighbours}
        self.assertEqual(places.get("102004"), THIS_TOPIC)  # Commits: the log card
        for other in ("102002", "102003"):
            if other in places:
                self.assertEqual(places[other], THIS_DECK)  # Branches
        for other in neighbours:
            if not other.startswith("102"):
                self.assertIn(places[other], ("Alpha", "Gamma"))
        self.assertEqual([found[1][-1] for found in listed[1:]], [places[other] for other in neighbours])
        self.assertEqual([found[1][0] for found in listed[1:]], [shorten(self.board.card_by_id[other].question) for other in neighbours])
        self.assertEqual(self.state.explored_ids, {COMMIT})  # drawing the list explores nothing
        self.assertEqual(self.board.next_card(COMMIT), neighbours[0])  # all unread, no quiz among Beta's: the neighbours' order
        self.assertEqual(self.call("related-toggle")[OUT["related_class"]], "related-toggle")  # a second click hides it

    def test_the_next_card_follows_the_read_cards_and_d1s_order_among_the_unread_quizzes(self):
        self.show(COMMIT)
        neighbours = self.board.neighbours(COMMIT)
        self.state.mark_read(neighbours[0])
        self.assertEqual(self.board.next_card(COMMIT), neighbours[1])  # the unread first
        for other in neighbours:
            self.state.mark_read(other)
        self.assertEqual(self.board.next_card(COMMIT), neighbours[0])  # all read: the neighbours' order
        self.assertEqual(rank_by_confidence(["f1", "q1", "f2", "q2"], {}, {}, {"q1": 1.6, "q2": 0.2}), ["f1", "q2", "f2", "q1"])  # D1's order among the quizzes

    def test_a_click_on_a_related_entry_shows_and_explores_that_card_and_hides_the_list(self):
        self.show(COMMIT)
        self.call("related-toggle")
        neighbours = self.board.neighbours(COMMIT)
        result = self.call("related-2")  # the second neighbour (slot 0 is the marked next card)
        self.assertEqual(self.state.shown, neighbours[1])
        self.assertEqual(self.state.explored_ids, {COMMIT, neighbours[1]})
        self.assertEqual(self.state.trail, [COMMIT])
        self.assertEqual(result[OUT["related_class"]], "related-toggle")
        self.assertEqual(entries(result, "related", RELATED_SLOTS), [])
        self.assertFalse(result[OUT["back_disabled"]])
        self.show(COMMIT)
        self.call("related-5")  # the list is not drawn: nothing
        self.assertEqual(self.state.shown, COMMIT)
        self.call("related-toggle")
        self.call(f"related-{len(neighbours) + 1}")  # beyond the entries: nothing
        self.assertEqual(self.state.shown, COMMIT)
        result = self.call("related-0")  # the marked next card
        self.assertEqual(self.state.shown, self.board.next_card(COMMIT))
        self.assertTrue(result[OUT["face_class"]].endswith("front"))

    def test_back_returns_to_the_card_shown_before_and_is_disabled_on_an_empty_trail(self):
        self.go("how do I slice a list")
        first = self.state.shown
        self.go("null hypothesis")
        second = self.state.shown
        self.go("the history of a repository")
        third = self.state.shown
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(self.state.trail, [first, second])
        self.call("card-face")  # the third card turned
        result = self.call("back")
        self.assertEqual(self.state.shown, second)
        self.assertEqual(self.state.trail, [first])
        self.assertFalse(self.view["back"])  # the front up again
        self.assertFalse(result[OUT["back_disabled"]])
        self.assertEqual(self.state.explored_ids, {first, second, third})
        result = self.call("back")
        self.assertEqual(self.state.shown, first)
        self.assertTrue(result[OUT["back_disabled"]])
        result = self.call("back")  # an empty trail: nothing
        self.assertEqual(self.state.shown, first)
        self.assertTrue(result[OUT["back_disabled"]])
        self.go("how do I slice a list")  # the same card again: the trail is unchanged
        self.assertEqual(self.state.trail, [])

    def test_the_diagram_draws_the_explored_set_with_the_shown_cards_place_highlighted_and_takes_no_click(self):
        self.go("how do I slice a list")
        result = self.go("null hypothesis")
        tests_card = self.state.shown
        decks, topics, cards = dots(result[OUT["diagram"]])
        self.assertEqual(decks, [("01", False), ("03", True)])  # the explored decks in deck id order
        self.assertEqual(topics, [("02", True)])  # Gamma's topics with an explored card: Tests alone
        self.assertEqual(cards, [(tests_card, True)])
        titles = [props(span)["title"] for row in result[OUT["diagram"]] for span in row.children[1].children]
        self.assertEqual(titles[:2], ["Alpha", "Gamma"])  # named on hover
        self.assertEqual(titles[2], "Tests")
        self.assertEqual(titles[3], shorten(self.board.card_by_id[tests_card].question))
        for row in result[OUT["diagram"]]:
            for span in row.children[1].children:
                self.assertEqual(type(span).__name__, "Span")  # no button, no click
                self.assertIsNone(getattr(span, "n_clicks", None))
        self.board.state.show("103001")  # Intervals, in the same deck
        decks, topics, cards = dots(self.call(None)[OUT["diagram"]])
        self.assertEqual(topics, [("01", True), ("02", False)])  # both topics hold an explored card now
        self.assertEqual(cards, [("103001", True)])  # Intervals' explored cards alone
        self.call("back")
        decks, topics, cards = dots(self.call(None)[OUT["diagram"]])
        self.assertEqual((decks[1], topics[1], cards), (("03", True), ("02", True), [(tests_card, True)]))

    def test_the_tiles_say_how_to_make_the_results_files_and_open_their_panel_one_at_a_time(self):
        result = self.call(None)
        for letter in PANEL_LETTERS:
            tile = self.board.tile(letter)
            self.assertEqual(tile.numbers, ())
            self.assertIn("uv run evaluate.py", tile.line)  # no file: how to make it
            self.assertEqual(tile.what, WHAT[tile.level])
        self.assertEqual([self.board.tile(letter).level for letter in PANEL_LETTERS], ["S1", "C1", "T1", "D1", "P1"])  # TF-IDF: S1 is the app's
        result = self.call("tile-S")
        self.assertEqual(self.view["panel"], "S")
        self.assertEqual(result[OUT["tile_S"]], tile_class(True))
        self.assertEqual(result[OUT["tile_T"]], tile_class(False))
        self.assertEqual(result[OUT["panel_class"]], "model-panel")
        self.assertIn(WHAT["S1"], texts(result[OUT["panel"]]))
        self.assertIn("No results.csv in", str(result[OUT["panel"]]))
        result = self.call("tile-T")  # another tile: its model in the panel's place
        self.assertEqual(self.view["panel"], "T")
        self.assertEqual((result[OUT["tile_S"]], result[OUT["tile_T"]]), (tile_class(False), tile_class(True)))
        self.assertIn(WHAT["T1"], texts(result[OUT["panel"]]))
        result = self.call("tile-T")  # the open tile again: closed
        self.assertIsNone(self.view["panel"])
        self.assertEqual(result[OUT["panel_class"]], "model-panel hidden")

    def test_enter_does_nothing_while_a_panel_is_open_and_the_panel_keeps_the_card(self):
        self.show(COMMIT)
        self.call("tile-D")
        result = self.enter()
        self.assertEqual(result[OUT["face_class"]], "card-face front")  # not turned
        self.assertFalse(self.state.is_read(COMMIT))
        self.assertEqual(self.view["panel"], "D")
        self.call("tile-D")
        self.assertEqual(self.enter()[OUT["face_class"]], "card-face back")

    def test_the_layout_holds_every_region_and_the_stylesheet_is_landscape_only(self):
        layout = page_layout(HOME, self.board.tiles())
        text = str(layout)
        for component_id in (
            "view", KEY_INPUT, "prompt-text", "prompt-go", "prompt-answer", "prompt-message", "back", "diagram", "card-where", "related-toggle",
            "card-face", "quiz-control", "quiz-options", "quiz-confirm", "model-panel",
            *[f"related-{index}" for index in range(RELATED_SLOTS)], *[f"go-{index}" for index in range(GO_SLOTS)],
            *[f"tile-{letter}" for letter in PANEL_LETTERS],
        ):
            self.assertIn(f"id='{component_id}'", text, component_id)
        for class_name in ("topbar", "workspace", "diagram", "card-column", "card-frame", "related-list", "slide-row", "model-panel hidden"):
            self.assertIn(f"className='{class_name}'", text, class_name)
        self.assertEqual(props(layout)["children"][2].className, "slide-row")  # along the bottom, after the workspace
        tile = props(layout)["children"][2].children[0]
        self.assertEqual(tile.id, "tile-S")
        self.assertIn("Similarity", texts(tile))
        css = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
        self.assertNotIn("portrait", css)  # landscape only, no stacked layout
        self.assertIn("--size-number: 3rem", css)
        self.assertIn("--size-heading: 1.5rem", css)
        self.assertIn("overflow-y: auto", css)  # the one scroll bar, inside the card's frame
        self.assertIn("--navy-deep: #071827", css)  # the tokens written anew
        keys = (ASSETS_DIR / "keys.js").read_text(encoding="utf-8")
        self.assertIn(f'"{KEY_INPUT}"', keys)
        self.assertIn('event.key !== "Enter"', keys)
        app = create_app(self.board)
        self.assertEqual(app.title, "Guided learning: the dashboard")
        self.assertEqual(str(ASSETS_DIR.name), "assets")

    def test_the_card_shows_nothing_of_the_models(self):
        result = self.go("how do I slice a list")
        shown = str(result[OUT["face"]]) + str(result[OUT["where"]])
        for word in ("easy", "medium", "hard", "predicted", "margin", "level", "score", "neighbour"):
            self.assertNotIn(word, shown.lower(), word)


class ResultsTilesTest(Driver):
    """The slide row on the results fixtures: the same numbers and words as the models view's section."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        folder = Path(cls._tmp.name)
        cls.table = load_table(write_test_table(folder))
        test_standalone.write_results(folder / "results", course_note(cls.table.deck_ids))
        cls.board = Dashboard(cls.table, results_dir=folder / "results")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_five_tiles_show_the_two_numbers_with_their_verdict_words_and_the_line_without_a_file(self):
        tiles = {tile.letter: tile for tile in self.board.tiles()}
        self.assertEqual([tile.shown for tile in tiles["S"].numbers], ["60 %", "92 %"])
        self.assertEqual([tile.label for tile in tiles["S"].numbers], ["same-deck precision@5", "hand precision@1"])
        self.assertEqual([tile.shown for tile in tiles["C"].numbers], ["0.90", "0.80"])
        self.assertEqual([tile.shown for tile in tiles["T"].numbers], ["64 %", "89 %"])
        self.assertEqual([tile.shown for tile in tiles["D"].numbers], ["75 %", "0.41"])
        self.assertEqual(tiles["D"].numbers[1].verdict, "moderate")
        self.assertEqual(tiles["P"].numbers, ())
        self.assertIn("No prompt.csv in", tiles["P"].line)
        layout = page_layout(HOME, self.board.tiles())
        row = props(layout)["children"][2]
        shown = texts(row.children[0])
        self.assertEqual(shown[:3], ["S", "Similarity", WHAT["S1"]])
        self.assertIn("60 %", shown)
        self.assertIn(tiles["S"].numbers[0].verdict, shown)
        self.assertNotIn(tiles["S"].numbers[0].sentence, shown)  # the sentence is the panel's
        self.assertIn("No prompt.csv in", " ".join(texts(row.children[4])))

    def test_a_tiles_panel_shows_the_numbers_with_their_label_contrast_sentence_verdict_and_explanation(self):
        result = self.call("tile-S")
        shown = texts(result[OUT["panel"]])
        headline = self.board.tile("S").numbers[0]
        self.assertEqual(shown[:4], ["S", "Similarity", "S1", WHAT["S1"]])
        for part in (headline.shown, headline.label, f"against {headline.contrast_shown}", headline.sentence, headline.verdict, f": {headline.explanation}"):
            self.assertIn(part, shown, part)
        self.assertEqual(headline.contrast_shown, "random positions 30 %")
        blocks = [child for child in result[OUT["panel"]][2].children]
        self.assertEqual([block.className for block in blocks], ["number-block large", "number-block large"])
        result = self.call("tile-P")
        self.assertIn("No prompt.csv in", " ".join(texts(result[OUT["panel"]])))


class StateFileTest(unittest.TestCase):
    """The dashboard's state file: the closed start, a restart with the explored cards and the shown card, the
    trail, another version refused, unknown cards dropped, written at every change."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.path = self.folder / "state" / STATE_NAME

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_fresh_start_without_a_file_and_the_file_written_at_a_change(self):
        state = DashboardState.load(self.path, {"a", "b"})
        self.assertTrue(state.fresh)
        self.assertFalse(self.path.exists())
        state.show("a")
        state.save()
        self.assertTrue(self.path.is_file())
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document, {"version": VERSION, "cards": {"a": {"read": False, "explored": True}}, "shown": "a", "trail": []})
        self.assertEqual(STATE_NAME, "dashboard_state.json")

    def test_a_restart_reads_the_explored_cards_the_shown_card_and_the_trail(self):
        state = DashboardState(self.path)
        state.show("a")
        state.mark_read("a")
        state.show("b")
        state.show("c")
        state.save()
        again = DashboardState.load(self.path, {"a", "b", "c", "d"})
        self.assertEqual((again.shown, again.trail, again.explored_ids), ("c", ["a", "b"], {"a", "b", "c"}))
        self.assertTrue(again.is_read("a"))
        self.assertFalse(again.is_read("b"))
        self.assertFalse(again.fresh)
        self.assertEqual(again.back(), "b")
        self.assertEqual(again.trail, ["a"])

    def test_unknown_cards_are_dropped_and_another_version_is_refused(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"version": VERSION, "cards": {"a": {"explored": True}, "x": {"read": True, "explored": True}},
                                         "shown": "x", "trail": ["a", "x"]}), encoding="utf-8")
        state = DashboardState.load(self.path, {"a", "b"})
        self.assertEqual((state.shown, state.trail, state.explored_ids), (None, ["a"], {"a"}))
        self.path.write_text(json.dumps({"version": 2, "cards": {}}), encoding="utf-8")
        with self.assertRaises(ValueError) as refused:
            DashboardState.load(self.path, {"a"})
        self.assertIn("backup question", str(refused.exception))
        self.assertIsNone(DashboardState.load(None, {"a"}).path)  # no file: in memory alone
        DashboardState(None).save()  # nothing written

    def test_main_loads_the_state_file_from_the_state_folder_and_stops_on_another_version(self):
        table_dir = write_test_table(self.folder / "course")
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"version": VERSION, "cards": {SLICING: {"read": True, "explored": True}}, "shown": SLICING, "trail": []}),
                             encoding="utf-8")
        served = {}

        def fake_create_app(board):
            served["board"] = board
            return SimpleNamespace(run=lambda **options: served.update(options))

        with mock.patch("app.dashboard.create_app", fake_create_app), mock.patch("builtins.print"):
            dashboard_main(table_dir, self.folder / "results", self.path.parent, port=8052)
        board = served["board"]
        self.assertEqual(board.state.path, self.path)
        self.assertEqual((board.state.shown, board.state.explored_ids), (SLICING, {SLICING}))
        self.assertEqual(served["port"], 8052)
        self.assertEqual(board.results.folder, self.folder / "results")
        self.path.write_text(json.dumps({"version": 7}), encoding="utf-8")
        with mock.patch("app.dashboard.create_app", fake_create_app), mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as stop:
                dashboard_main(table_dir, None, self.path.parent)
        self.assertIn("version 7", str(stop.exception.code))
        with self.assertRaises(SystemExit) as stop:
            dashboard_main(self.folder / "missing")
        self.assertIn("cards.csv", str(stop.exception.code))


class QuizRuleTest(unittest.TestCase):
    """The judgement as plain functions."""

    def test_right_only_with_exactly_the_correct_options(self):
        self.assertTrue(quiz.is_right((0,), [0]))
        self.assertFalse(quiz.is_right((0,), [0, 1]))  # an extra wrong one
        self.assertFalse(quiz.is_right((0, 2), [0]))  # a missing correct one
        self.assertTrue(quiz.is_right((0, 2), [2, 0]))
        self.assertFalse(quiz.is_right((0,), []))  # nothing checked is no answer
        self.assertTrue(quiz.confirm_disabled([]))
        self.assertTrue(quiz.confirm_disabled(None))
        self.assertFalse(quiz.confirm_disabled([1]))

    def test_the_checks_are_read_from_the_checklists_value_in_file_order(self):
        self.assertEqual(quiz.checked_indices(["2", 0, "x", 7, 2], 4), [0, 2])
        self.assertEqual(quiz.checked_indices(None, 4), [])

    def test_the_judgement_gives_the_checked_options_feedback_on_a_wrong_answer(self):
        card = model_support.CARDS[1]
        self.assertEqual(card.id, SLICING)
        self.assertEqual(quiz.judge(card, [0]), (True, []))
        right, blocks = quiz.judge(card, [3, 0])
        self.assertFalse(right)
        self.assertEqual([(block.option, block.correct) for block in blocks], [("list[start:stop]", True), ("list(start, stop)", False)])
        self.assertEqual(blocks[1].answer, "list(...) builds a list from an iterable.")
        self.assertTrue(quiz.has_options(card))
        self.assertFalse(quiz.has_options(model_support.CARDS[0]))
        self.assertEqual((quiz.CORRECT_LABEL, quiz.WRONG_LABEL), ("Correct answer", "Wrong answer"))


NO_VIEWER = """
import sys
sys.modules["project"] = None  # the viewer is not there: any import of it fails
sys.path.insert(0, {module_dir!r})
sys.path.insert(0, {tests_dir!r})
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import test_dashboard
from app import launch
from app.dashboard.app import INPUTS, OUT, Dashboard, register_callback
from app.course_table import load_table
with tempfile.TemporaryDirectory() as folder:
    board = Dashboard(load_table(test_dashboard.write_test_table(Path(folder))))
update = register_callback(test_dashboard.test_standalone.RecordingApp(), board)
view = None
for trigger, request in (("prompt-go", "how do I slice a list"), ("related-toggle", None), ("related-1", None), ("tile-D", None)):
    with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
        view = update(*([0] * INPUTS), [], request, view)[OUT["view"]]
assert len(board.state.explored_ids) == 2 and len(board.state.trail) == 1 and view["panel"] == "D", (board.state.shown, view)
assert not launch.viewer_available()
loaded = sorted(name for name in sys.modules if (name == "project" or name.startswith(("project.", "app.glue"))) and sys.modules[name] is not None)
print("LOADED", loaded)
"""


class NoViewerTest(unittest.TestCase):
    def test_the_dashboard_runs_with_the_viewer_blocked_and_imports_none_of_it_or_the_glue(self):
        tests_dir, module_dir = Path(__file__).resolve().parent, model_support.MODULE_DIR
        script = NO_VIEWER.format(module_dir=str(module_dir), tests_dir=str(tests_dir))
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, cwd=str(module_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LOADED []", result.stdout)


TABLE_DIR = model_support.COURSE_TABLE_DIR
RESULTS_DIR = model_support.MODULE_DIR / "results"


@unittest.skipUnless(not missing_files(TABLE_DIR), "the course table is not there (export_course.py writes it)")
class RealTableTest(unittest.TestCase):
    def test_the_real_course_table_serves_a_request_the_related_cards_and_the_tiles(self):
        board = Dashboard(load_table(TABLE_DIR), results_dir=RESULTS_DIR)
        update = register_callback(test_standalone.RecordingApp(), board)
        view = None
        for trigger, request in (("prompt-go", "how do I merge two data frames in pandas"), ("related-toggle", None)):
            with mock.patch("dash.ctx", SimpleNamespace(triggered_id=trigger)):
                result = update(*([0] * INPUTS), [], request, view)
            view = result[OUT["view"]]
        shown = board.state.shown
        if shown is not None:
            self.assertIn(shown, board.card_by_id)
            self.assertTrue(board.where(shown))
            self.assertEqual(len(entries(result, "related", RELATED_SLOTS)), len(board.neighbours(shown)) + (1 if board.neighbours(shown) else 0))
        else:
            self.assertEqual(result[OUT["answer_class"]], "prompt-answer")
        for tile in board.tiles():
            self.assertTrue(tile.numbers or tile.line, tile.letter)


if __name__ == "__main__":
    unittest.main()
