"""The dashboard (app.dashboard): the capstone's app without the viewer, one card at a time with what a learner
sees of the models, on the dashboard's own state (Dash; nothing of the viewer, nothing of the glue).

Dashboard fits the course model with T1, D1 and P1 on the course table once, as the models view's Page does, and
reads three things of them: the shown card's related cards from the index (CourseModel.related, the five nearest),
D1's order among the unread quizzes (the next card), and P1's answer to a request (PromptModel.find); nothing else
of the models shows on the card: no level, no margin, no score, no tag. Where a card lies is T1's membership, the
model's topics as the full app gives every deck from the start; the topics keep the scripted ids and names.

* The card (one at a time, as the viewer shows it): the front the question, a click on the card or Enter turns it
  to the answer; a quiz's front the question with its options and Confirm under it, Confirm disabled while nothing
  is checked (its own callback on the checks), Confirm or Enter turns it to its back, the judgement (quiz.judge),
  Enter on the back turns it to the front with the checks cleared.
  The turn marks the card read (a quiz once judged). With no card shown the card's place says how to begin.
* The navigation: Related lists the shown card's five related cards by the viewer's rule (neighbours.rank_by_confidence
  without the confidence: the unread first, the quizzes D1 predicted in D1's order among them, the read ones after
  them in the neighbours' order), the next card first and marked and again in its place, each with where it lies
  (this topic, this deck, another deck by name), every entry a click that shows that card; nothing is explored when
  the list is drawn. Back returns to the card shown before (the trail, state.DashboardState.back), disabled on an
  empty trail. The request box is the one way in: P1's answer shows the topic's card where it navigates (the
  selected card, else the topic's nearest), else "no match" with up to three entries, the nearest topics' cards,
  each a click that shows its card and closes the answer; an empty request does nothing.
* What explores a card (step 33's rule on the dashboard's parts): a selection in any way, Go's answer that shows a
  card, a click on a Related entry, on one of Go's entries or on a card in the diagram, Back; nothing else and
  nothing ahead.
* The diagram (layout.diagram): decks → topics → cards, each item a dot with its name: the explored decks, the open
  deck's topics that hold an explored card, the open topic's explored cards, nothing unexplored; the open deck and
  topic are the shown card's until a click in the diagram opens others; the shown card's deck, topic and card
  highlighted; empty at a fresh start, filling as the learner explores. A click on a card selects it, on a topic
  lists its cards in the cards row, on a deck lists its topics in the topics row with the cards row empty; a click on
  a deck or a topic leaves the card as it is. A selection in any way opens the shown card's deck and topic again.
* The slide row: five tiles, S, C, T, D and P, each the generation the app runs (S2 with the encoder, else S1; C1,
  T1, D1, P1) with its headline and its two numbers with their verdict words from the results files
  (standalone.headlines, the same numbers and words as the models view), one line where a results file is missing;
  nothing live. A click on a tile opens the model's panel over the two columns, the tiles staying visible; another
  tile shows that model in its place, a second click on the open tile closes it; Enter does nothing while it is open.
  The panel holds the numbers on its left and the model's figure on its right (panel_figure), the whole course's
  whatever card is shown: S where the related cards lie (this topic, this deck, another deck), C the decks on the
  sphere with the connections below the preset and the shown card's deck in gold, T the margins with the threshold
  and the most confident quarter, D D1's confusion matrix on the rated quizzes, P P1's judged requests on and off the
  course by where it took them (the counts evaluate.py writes).
* The state: the read and explored flags, the shown card and the trail in the dashboard's own state file
  (state.DashboardState, saved at every change); the flip, Related drawn, the last request, the open tile, the
  diagram's open deck and topic and the demo flag in the `view` store (HOME's keys), runtime only. The closed start:
  a fresh start shows nothing, a restart shows the explored cards and the shown card again. No episodes' log.
* The demo (docs/demo/): DemoBoard, a second board over deck 00's table (data/demo/dashboard/), fitted at the start
  and kept in memory: T1's membership as written, P1 on its own self set, no D1, its state without a file. Demo in
  the top bar sets the flag and shows deck 00 from card 001 with a fresh state; every region renders from the board
  the flag names (Boards), the slide row stays the course's. End demo clears the flag and drops the demo's state, and
  the course's card, trail and explored cards come back from its state, which the demo never touched; a reload ends
  the demo the same way. Without course data the dashboard runs on deck 00's board alone: outside the demo the box
  says "no data loaded", the box and Go are disabled and the diagram and the card's place are empty; the tiles read
  the results files' newest runs, and the panels' figures but P's say they need the course.

register_callback registers the page's callback: every button and the hidden inputs of Enter and of the diagram's
clicks are its inputs, the checks, the request box's text and the view its states; it steps the board the view names
and renders every region from the state and the view (Boards.step, Boards.render). Confirm's disabled state has a
callback of its own on the checks, since the checks are the page's callback's state, not its input.
"""

import json
import math
from pathlib import Path
from typing import Optional, Sequence

from dash import Dash, Input, Output, State

from ..course_table import CourseTable, DeckRow
from ..figures import ELSEWHERE, HERE, confusion_figure
from ..figures import THIS_DECK as SAME_DECK
from ..models import clustering, embeddings, similar
from ..models.course_model import CourseModel, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.prompt_model import PromptModel, variant_label
from ..neighbours import rank_by_confidence, shorten
from ..standalone.app import DECK_PART, d2_label, s2_label
from ..standalone.headlines import WHAT, headlines, outcomes
from ..standalone.layout import CUT_LABELS, DECKS_VIEW, empty_slot, graph, margin_histogram
from ..standalone.results import LEVEL_FILES, Results, course_note
from . import quiz
from .layout import (
    BEGIN_HINT, DEMO_BUTTON, DIAGRAM_INPUT, DIAGRAM_ROWS, GO_SLOTS, KEY_INPUT, NO_DATA_HINT, PANEL_LETTERS, PANEL_NAMES, RELATED_SLOTS, THIS_DECK,
    THIS_TOPIC, Dot, Entry, Tile, card_face, card_where, demo_controls,
    diagram, go_slots, model_panel, page_layout, prompt_answer, quiz_control_class, quiz_options, related_slots, related_toggle, tile_class,
)
from .panel_figures import deck_sphere, outcome_grid, places_figure
from .state import DashboardState

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"  # next to evaluate.py: the capstone's root
TITLE = "Guided learning: the dashboard"
# The view store: what is not saved. `request`: the last request as {"text", "match", "entries"}, the entries of
# a "no match" as [deck id, topic id, card id] each, None once one of them is clicked; `open`: the diagram's open
# deck and topic as [deck id, topic id or None] after a click on a deck or a topic, None for the shown card's;
# `demo`: the demo runs, every region from deck 00's board
HOME = {"back": False, "related": False, "request": None, "panel": None, "open": None, "demo": False}
TURN_TRIGGERS = ("card-face", KEY_INPUT)  # a click on the card, Enter
PROMPT_TRIGGERS = ("prompt-go", "prompt-text")  # Go, and Enter in the request box (its n_submit)
RELATED_PREFIX, GO_PREFIX, TILE_PREFIX = "related-", "go-", "tile-"
# The callback's outputs by name, in order (OUT gives each one's place in the result)
OUTPUT_NAMES = (
    ["view", "where", "face", "face_class", "quiz_class", "options", "checks", "related_class", "related_disabled"]
    + [name for index in range(RELATED_SLOTS) for name in (f"related_{index}", f"related_{index}_class")]
    + ["message", "answer_class"]
    + [name for index in range(GO_SLOTS) for name in (f"go_{index}", f"go_{index}_class")]
    + ["diagram", "back_disabled"]
    + [f"tile_{letter}" for letter in PANEL_LETTERS]
    + ["panel", "panel_class"]
    + ["demo_label", "demo_class", "demo_disabled", "placeholder", "box_disabled", "go_disabled"]
)
OUT = {name: index for index, name in enumerate(OUTPUT_NAMES)}
# The callback's inputs, in order: the card's click, Enter, Confirm, Related, its slots, Back, Demo, Go, Enter in the
# box, Go's slots, the tiles, a click in the diagram; then the states: the checks, the box's text, the view
INPUT_IDS = (
    ["card-face", KEY_INPUT, "quiz-confirm", "related-toggle"] + [f"{RELATED_PREFIX}{index}" for index in range(RELATED_SLOTS)]
    + ["back", DEMO_BUTTON, "prompt-go", "prompt-text"] + [f"{GO_PREFIX}{index}" for index in range(GO_SLOTS)] + [f"{TILE_PREFIX}{letter}" for letter in PANEL_LETTERS]
    + [DIAGRAM_INPUT]
)
INPUTS = len(INPUT_IDS)
VALUE_INPUTS = (KEY_INPUT, DIAGRAM_INPUT)  # the hidden inputs, read by their value
FOCUS_SHARE = 0.25  # T's figure: the most confident quarter, the share its tile's second number is over
NO_RELATED = "No related card above the minimum score."
NO_MARGINS = "No card with a margin."
NO_OUTCOMES = ("No outcome counts in the newest run of prompt.csv yet. The offline evaluation writes them, next to app/: "
               "uv run evaluate.py --part prompt")
NO_DATA_FIGURE = "No course data loaded: this figure is drawn from the course."
NO_D1 = "No D1 on deck 00: its quizzes carry their levels as written."


def slot_of(trigger, prefix: str) -> Optional[int]:
    """The slot's index of a trigger like "related-2"; None for another trigger."""
    if not isinstance(trigger, str) or not trigger.startswith(prefix):
        return None
    rest = trigger[len(prefix):]
    return int(rest) if rest.isdigit() else None


def tile_letter(trigger) -> Optional[str]:
    """The model letter of a tile's click ("tile-S"); None for another trigger."""
    if isinstance(trigger, str) and trigger.startswith(TILE_PREFIX) and trigger[len(TILE_PREFIX):] in PANEL_LETTERS:
        return trigger[len(TILE_PREFIX):]
    return None


def toggle_panel(view: dict, letter: str) -> None:
    """A tile's click: its model's panel opens, or closes when it is the open one."""
    view["panel"] = None if view["panel"] == letter else letter


def diagram_click(value) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """The row ("deck", "topic" or "card"), the id and a topic's deck of a click in the diagram, from the hidden
    input's JSON (assets/diagram.js); three Nones for anything else."""
    try:
        click = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        click = None
    if not isinstance(click, dict) or click.get("row") not in DIAGRAM_ROWS or not isinstance(click.get("id"), str):
        return None, None, None
    deck = click.get("deck")
    return click["row"], click["id"], deck if isinstance(deck, str) else None


class Dashboard:
    """The models fitted on the course table, the dashboard's state, and what the regions show of them."""

    def __init__(
        self, table: CourseTable, state: Optional[DashboardState] = None, model: Optional[CourseModel] = None,
        difficulty: Optional[DifficultyReport] = None, results_dir: Optional[Path] = None, vectoriser=None,
        min_score: Optional[float] = similar.MIN_SCORE,
    ) -> None:
        """`vectoriser` and `min_score` are the index's when the model is fitted here: the app's vectoriser at its own
        minimum from the launcher (dashboard.main), TF-IDF at the fixed minimum by default. `state` is the state
        file's (state.DashboardState.load); none keeps the state in memory alone."""
        self.table = table
        self.state = state if state is not None else DashboardState()
        self.changed = False  # whether the state changed in the last step, so the callback saves it
        self.model = model if model is not None else CourseModel(table.cards, table.deck_ids, table.seeds, vectoriser, min_score=min_score)
        self.difficulty = difficulty if difficulty is not None else fit_difficulty(table.cards, self.model.index)
        self.prompt = PromptModel(self.model.index, self.model.topic_model, table.cards)
        self.card_by_id = {card.id: card for card in table.cards}
        self.deck_by_id: dict[str, DeckRow] = {deck.deck_id: deck for deck in table.decks}
        self.topic_by_card: dict[str, TopicLayout] = {
            card_id: placed for placed in self.model.topics.values() for card_id in placed.ids
        }
        self._figures: dict = {}  # the panels' figures that never change (all but C's), made at the first opening
        index = self.model.index
        encoded = isinstance(index.vectoriser, (embeddings.HybridVectoriser, embeddings.SentenceEncoder))
        self.levels = {"S": "S2" if encoded else "S1", "C": "C1", "T": "T1", "D": "D1", "P": "P1"}  # the generation the app runs
        chosen = {"P1": variant_label(index.vectoriser), "D2": d2_label()}
        if s2_label() is not None:
            chosen["S2"] = s2_label()
        self.results = Results(results_dir, self.results_note(), None, chosen=chosen)

    def results_note(self) -> Optional[str]:
        """The note of the results rows this board's tiles read: its course's."""
        return course_note(self.table.deck_ids)

    # --- where a card lies -------------------------------------------------------------------------------------

    def deck_of(self, card_id: Optional[str]) -> Optional[DeckRow]:
        deck_id = self.model.deck_of.get(card_id) if card_id else None
        return self.deck_by_id.get(deck_id) if deck_id is not None else None

    def topic_of(self, card_id: Optional[str]) -> Optional[TopicLayout]:
        """The card's topic under T1's membership."""
        return self.topic_by_card.get(card_id) if card_id else None

    def where(self, card_id: Optional[str]) -> str:
        """The card's title row: the deck and the topic by name ("Alpha › Lists")."""
        deck, topic = self.deck_of(card_id), self.topic_of(card_id)
        if deck is None:
            return ""
        return f"{deck.name} › {topic.name}" if topic is not None else deck.name

    def place(self, card_id: str, other: str) -> str:
        """Where a related card lies, seen from the shown card: this topic, this deck, or another deck by name."""
        topic, other_topic = self.topic_of(card_id), self.topic_of(other)
        if topic is not None and other_topic is not None and (topic.deck_id, topic.topic_id) == (other_topic.deck_id, other_topic.topic_id):
            return THIS_TOPIC
        if self.model.deck_of.get(card_id) == self.model.deck_of.get(other):
            return THIS_DECK
        deck = self.deck_of(other)
        return deck.name if deck is not None else f"deck {self.model.deck_of.get(other, '–')}"

    # --- the related cards and the next card -------------------------------------------------------------------

    def neighbours(self, card_id: str) -> list[str]:
        """The card's related cards from the index, best first (the five nearest above the index's own minimum)."""
        return [other for other, _score in self.model.related(card_id) if other in self.card_by_id]

    def next_card(self, card_id: str) -> Optional[str]:
        """The neighbour to take next, by the viewer's rule without the confidence: the unread first, the quizzes
        D1 predicted in D1's order among them, the read ones after them in the neighbours' order."""
        neighbours = self.neighbours(card_id)
        if not neighbours:
            return None
        read = {other: self.state.is_read(other) for other in neighbours}
        return rank_by_confidence(neighbours, read, {}, self.difficulty.scores)[0]

    def related_entries(self, card_id: Optional[str]) -> list[Entry]:
        """Related's entries: the next card first, marked, then every neighbour in the index's order, each with
        its question shortened and where it lies; none without a card or without neighbours."""
        if card_id not in self.card_by_id:
            return []
        neighbours = self.neighbours(card_id)
        if not neighbours:
            return []
        entries = [Entry(other, shorten(self.card_by_id[other].question), self.place(card_id, other)) for other in neighbours]
        first = next(entry for entry in entries if entry.card_id == self.next_card(card_id))
        return [first._replace(marked=True)] + entries

    # --- the request box ---------------------------------------------------------------------------------------

    def find(self, text: str) -> tuple[Optional[str], list[list[str]]]:
        """P1's answer to a request: (the card to show, []) where P1 navigates (the selected card, else the entered
        topic's nearest card); (None, the "no match" entries) otherwise: the nearest topics' cards as [deck id,
        topic id, card id], up to GO_SLOTS, a topic without a card left out."""
        answer = self.prompt.find(text)
        if answer.navigate and answer.ranking:
            card = answer.card if answer.card is not None else answer.ranking[0].card
            if card in self.card_by_id:
                return card, []
        entries = []
        for scored in answer.targets:
            if scored.card is None or scored.card not in self.card_by_id:
                continue
            deck_id = self.model.deck_ids.get(scored.key[0], scored.key[0])
            entries.append([deck_id, scored.key[1], scored.card])
        return None, entries[:GO_SLOTS]

    def request_entries(self, request: Optional[dict]) -> list[Entry]:
        """The last request's "no match" entries as Go's entries: the deck's and the topic's names with the card's
        question shortened; none after a match or before a request."""
        if not isinstance(request, dict) or request.get("match"):
            return []
        entries = []
        for item in request.get("entries") or []:
            deck_id, topic_id, card_id = (list(item) + [None, None, None])[:3]
            card = self.card_by_id.get(card_id)
            if card is None:
                continue
            deck = self.deck_by_id.get(deck_id)
            topic = self.model.topics.get((deck_id, topic_id))
            label = f"{deck.name if deck else deck_id} › {topic.name if topic else topic_id} · {shorten(card.question)}"
            entries.append(Entry(card_id, label))
        return entries

    # --- the diagram -------------------------------------------------------------------------------------------

    def explored(self) -> set[str]:
        return {card_id for card_id in self.state.explored_ids if card_id in self.card_by_id}

    def explored_decks(self) -> list[str]:
        """The decks that hold an explored card, in deck id order."""
        return sorted({self.model.deck_of[card_id] for card_id in self.explored() if card_id in self.model.deck_of})

    def explored_topics(self, deck_id: Optional[str]) -> list[TopicLayout]:
        """The deck's topics that hold an explored card, in file order."""
        if deck_id is None:
            return []
        explored = self.explored()
        return [placed for placed in self.model.topics_of_deck(deck_id) if any(card_id in explored for card_id in placed.ids)]

    def opened(self, view_open) -> tuple[Optional[str], Optional[str]]:
        """The diagram's open deck and topic: the ones a click in the diagram opened (the view's `open`), else the
        shown card's."""
        if isinstance(view_open, (list, tuple)) and len(view_open) == 2:
            return view_open[0], view_open[1]
        shown = self.state.shown if self.state.shown in self.card_by_id else None
        topic = self.topic_of(shown)
        return (self.model.deck_of.get(shown) if shown else None), (topic.topic_id if topic is not None else None)

    def diagram(self, view_open=None) -> tuple[list[Dot], list[Dot], list[Dot]]:
        """The three rows: the explored decks in deck id order; the open deck's topics that hold an explored card, in
        file order; the open topic's explored cards, in the topic's order (opened: the shown card's deck and topic, or
        the ones a click in the diagram opened); the shown card's deck, topic and card highlighted, the open deck and
        topic marked."""
        explored = self.explored()
        shown = self.state.shown if self.state.shown in self.card_by_id else None
        shown_deck = self.model.deck_of.get(shown) if shown else None
        shown_topic = self.topic_of(shown)
        shown_key = (shown_topic.deck_id, shown_topic.topic_id) if shown_topic is not None else None
        open_deck, open_topic = self.opened(view_open)
        decks = [
            Dot(deck_id, self.deck_by_id[deck_id].name if deck_id in self.deck_by_id else deck_id, deck_id == shown_deck, deck_id == open_deck)
            for deck_id in self.explored_decks()
        ]
        topics = [
            Dot(placed.topic_id, placed.name, (placed.deck_id, placed.topic_id) == shown_key, placed.topic_id == open_topic, placed.deck_id)
            for placed in self.explored_topics(open_deck)
        ]
        placed = self.model.topics.get((open_deck, open_topic)) if open_topic is not None else None
        cards = [Dot(card_id, shorten(self.card_by_id[card_id].question), card_id == shown) for card_id in placed.ids if card_id in explored] if placed else []
        return decks, topics, cards

    # --- the slide row -----------------------------------------------------------------------------------------

    def tile(self, letter: str) -> Tile:
        """The model's tile: the generation the app runs, its headline and its two numbers from the results files, or
        the one line where the file or the rows are missing."""
        level = self.levels[letter]
        numbers = headlines(self.results, level) or []
        line = ""
        if not numbers:
            line = self.results.hint(LEVEL_FILES[level]) if self.results.run(level) is None else self.results.no_rows(level)
        return Tile(letter, PANEL_NAMES[letter], level, WHAT[level], tuple(numbers), line)

    def tiles(self) -> list[Tile]:
        return [self.tile(letter) for letter in PANEL_LETTERS]

    # --- the panels' figures -----------------------------------------------------------------------------------

    def place_counts(self) -> dict[str, int]:
        """Where the course's related cards lie, seen from their card (as Related places them): this topic, this deck,
        another deck, counted over every card's related cards."""
        counts = dict.fromkeys((HERE, SAME_DECK, ELSEWHERE), 0)
        kinds = {THIS_TOPIC: HERE, THIS_DECK: SAME_DECK}
        for card_id in self.card_by_id:
            for other in self.neighbours(card_id):
                counts[kinds.get(self.place(card_id, other), ELSEWHERE)] += 1
        return counts

    def deck_links(self) -> list[tuple[str, str]]:
        """The deck connections below the preset, as the models view draws them: the first merges, until a part would
        hold more than DECK_PART decks."""
        return list(self.model.deck_links[: clustering.preset_rank(self.model.deck_links, DECK_PART)])

    def deck_figure(self, shown: Optional[str] = None):
        """C's figure: the decks on the sphere with their connections, the shown card's deck gold and in front (`shown`,
        the board's own shown card by default; a card of another board marks none)."""
        names = {deck_id: deck.name for deck_id, deck in self.deck_by_id.items()}
        shown = shown if shown is not None else self.state.shown
        shown = shown if shown in self.card_by_id else None
        current = self.model.deck_of.get(shown) if shown else None
        figure = deck_sphere(self.model.deck_points, names, self.deck_links(), current, CUT_LABELS[DECKS_VIEW].format(size=DECK_PART))
        return graph(figure, "panel-graph")

    def static_figure(self, letter: str):
        """S's, T's, D's or P's figure, the same at every opening, or the line that says why there is none."""
        if letter == "S":
            counts = self.place_counts()
            if not sum(counts.values()):
                return empty_slot(NO_RELATED)
            return graph(places_figure(counts, len(self.card_by_id)), "panel-graph")
        if letter == "T":
            topics = self.model.topic_model
            margins = [margin for margin in topics.margin_of.values() if not math.isnan(margin)]
            if not margins:
                return empty_slot(NO_MARGINS)
            return graph(margin_histogram(margins, topics.margin, topics.threshold_for_share(FOCUS_SHARE), FOCUS_SHARE), "panel-graph")
        if letter == "D":
            if not self.difficulty.measured:
                return empty_slot(self.difficulty.note or "No difficulty model.")
            return graph(confusion_figure(self.difficulty), "panel-graph")
        counts = outcomes(self.results)
        return graph(outcome_grid(counts), "panel-graph") if counts is not None else empty_slot(NO_OUTCOMES)

    def panel_figure(self, letter: str, shown: Optional[str] = None):
        """The open model's figure, the whole course's whatever card is shown (C marks the shown card's deck, `shown`
        as deck_figure takes it): S where the related cards lie, C the decks on the sphere, T the margins with the
        threshold and the most confident quarter, D D1's confusion matrix, P P1's judged requests on and off the course;
        a line where there is none."""
        if letter == "C":
            return self.deck_figure(shown)
        if letter not in self._figures:
            self._figures[letter] = self.static_figure(letter)
        return self._figures[letter]

    # --- the moves ---------------------------------------------------------------------------------------------

    def step(self, view: Optional[dict], trigger, value=None, request: Optional[str] = None, clicked=None) -> tuple[dict, list]:
        """The view and the checks after a trigger: a click on the card or Enter turns it (a quiz's Confirm with
        checks, Enter on its back to the front with the checks cleared; nothing while a panel is open), Confirm
        judges, Related draws or hides the list, one of its entries shows that card, Back the card before, Go or
        Enter in the box runs the request, one of Go's entries shows that card and closes the answer, a tile opens its
        panel or closes it, a click in the diagram (`clicked`, the hidden input's value) selects its card or opens its
        deck or topic. A selection in any way explores the card, puts the front up, hides the list, clears the checks
        and opens the shown card's deck and topic in the diagram."""
        view = {**HOME, **(view if isinstance(view, dict) else {})}
        if view["panel"] not in (None, *PANEL_LETTERS):
            view["panel"] = None
        if not isinstance(view["request"], dict):
            view["request"] = None
        if not (isinstance(view["open"], (list, tuple)) and len(view["open"]) == 2):
            view["open"] = None
        checks = list(value or [])
        state = self.state
        shown = state.shown if state.shown in self.card_by_id else None
        card = self.card_by_id.get(shown)
        if card is None:
            view["back"], view["related"], checks = False, False, []

        def select(card_id: str) -> None:
            nonlocal checks
            state.show(card_id)
            self.changed = True
            view["back"], view["related"], view["open"], checks = False, False, None, []

        def judge() -> None:
            view["back"] = True
            if not state.is_read(shown):
                state.mark_read(shown)
                self.changed = True

        if trigger in TURN_TRIGGERS and card is not None:
            if trigger == KEY_INPUT and view["panel"] is not None:
                pass  # Enter does nothing while a model's panel is open
            elif quiz.has_options(card):
                if view["back"]:
                    view["back"], checks = False, []
                elif trigger == KEY_INPUT and not quiz.confirm_disabled(checks):
                    judge()
            else:
                view["back"] = not view["back"]
                if view["back"]:
                    judge()
        elif trigger == "quiz-confirm" and card is not None and quiz.has_options(card) and not view["back"] and not quiz.confirm_disabled(checks):
            judge()
        elif trigger == "related-toggle" and card is not None:
            view["related"] = not view["related"]
        elif slot_of(trigger, RELATED_PREFIX) is not None and card is not None and view["related"]:
            entries = self.related_entries(shown)
            index = slot_of(trigger, RELATED_PREFIX)
            if index < len(entries):
                select(entries[index].card_id)
        elif trigger == "back":
            if state.back() is not None:
                self.changed = True
                view["back"], view["related"], view["open"], checks = False, False, None, []
        elif trigger in PROMPT_TRIGGERS:
            text = (request or "").strip()
            if text:
                found, entries = self.find(text)
                view["request"] = {"text": text, "match": found is not None, "entries": entries}
                if found is not None:
                    select(found)
        elif slot_of(trigger, GO_PREFIX) is not None:
            entries = self.request_entries(view["request"])
            index = slot_of(trigger, GO_PREFIX)
            if index < len(entries):
                select(entries[index].card_id)
                view["request"] = None  # the answer closed
        elif tile_letter(trigger) is not None:
            toggle_panel(view, tile_letter(trigger))
        elif trigger == DIAGRAM_INPUT:
            row, dot_id, deck_id = diagram_click(clicked)
            if row == "card" and state.is_explored(dot_id) and dot_id in self.card_by_id:
                select(dot_id)
            elif row == "deck" and dot_id in self.explored_decks():
                view["open"] = [dot_id, None]  # the topics row lists the deck's topics, the cards row is empty
            elif row == "topic" and any(placed.topic_id == dot_id for placed in self.explored_topics(deck_id)):
                view["open"] = [deck_id, dot_id]  # the cards row lists the topic's cards
        return view, checks

    def render(self, view: dict, checks: Sequence, slide: Optional["Dashboard"] = None, hint: str = BEGIN_HINT) -> tuple:
        """Every region from the state and the view, in OUTPUT_NAMES' order up to the model's panel: the tiles' classes
        and the panel from `slide` (this board by default; the course's in the demo), `hint` with no card shown."""
        slide = slide if slide is not None else self
        shown = self.state.shown if self.state.shown in self.card_by_id else None
        card = self.card_by_id.get(shown)
        checks = list(checks or [])
        face, face_class = card_face(card, view["back"], checks, hint)
        entries = self.related_entries(shown) if card is not None and view["related"] else []
        go_entries = self.request_entries(view["request"])
        decks, topics, cards = self.diagram(view["open"])
        tile = slide.tile(view["panel"]) if view["panel"] in PANEL_LETTERS else None
        return (
            view,
            card_where(self.where(shown)),
            face,
            face_class,
            quiz_control_class(card, view["back"]),
            quiz_options(card),
            checks,
            *related_toggle(card is not None, view["related"]),
            *related_slots(entries, view["related"]),
            *prompt_answer(view["request"], go_entries),
            *go_slots(go_entries),
            diagram(decks, topics, cards),
            not self.state.can_go_back,
            *[tile_class(letter == view["panel"]) for letter in PANEL_LETTERS],
            *model_panel(tile, slide.panel_figure(tile.letter, shown) if tile is not None else None),
        )


class DemoBoard(Dashboard):
    """The board of deck 00 (docs/demo/architecture.md, "The dashboard"): a Dashboard over deck 00's table, fitted at
    the start and kept in memory: the course model over deck 00's cards with T1's membership as written (no card
    moved), its index for Related, P1 over that index and those topics with its floors from deck 00's own self set; no
    D1, so Related ranks the unread first in the neighbours' order. Its vectoriser is the app's kind without the
    embeddings' cache (embeddings.uncached_copy); its state a DashboardState without a file, fresh at every Demo.

    Without course data its slide row is the page's: the results files' newest runs, whatever course they name, and
    P's figure from them; the other figures are drawn from the course and say so."""

    def __init__(self, table: CourseTable, results_dir: Optional[Path] = None, vectoriser=None,
                 min_score: Optional[float] = similar.MIN_SCORE) -> None:
        model = CourseModel(table.cards, table.deck_ids, table.seeds, vectoriser, min_score=min_score, moves=False)
        no_d1 = DifficultyReport("", (0, 0, 0), 0, 0, 0, note=NO_D1)
        super().__init__(table, DashboardState(), model, no_d1, results_dir)
        self.start = table.cards[0].id  # card 001 of topic 01, the file's first card (the deck's rules)

    def results_note(self) -> Optional[str]:
        """None: every row, deck 00 being no course of the results files."""
        return None

    def restart(self) -> None:
        """Demo: a fresh state with card 001 shown, which explores it."""
        self.state = DashboardState()
        self.state.show(self.start)
        self.changed = False

    def end(self) -> None:
        """End demo (or a reload): the demo's state dropped."""
        self.state = DashboardState()
        self.changed = False

    def deck_figure(self, shown: Optional[str] = None):
        return empty_slot(NO_DATA_FIGURE)

    def static_figure(self, letter: str):
        return super().static_figure(letter) if letter == "P" else empty_slot(NO_DATA_FIGURE)


class Boards:
    """The course's board and deck 00's, and which one the regions show: the board the view's demo flag names (deck
    00's while the demo runs, the course's outside it); the slide row the course's, deck 00's board's without course
    data. At least one of the two is there."""

    def __init__(self, course: Optional[Dashboard], demo: Optional[DemoBoard] = None) -> None:
        if course is None and demo is None:
            raise ValueError("the dashboard needs the course's board or deck 00's")
        self.course = course
        self.demo = demo

    @property
    def slide(self) -> Dashboard:
        """The board whose tiles and panels show: the course's, else deck 00's."""
        return self.course if self.course is not None else self.demo

    def board(self, view: dict) -> Optional[Dashboard]:
        """The board the view names; None without course data outside the demo."""
        return self.demo if view.get("demo") and self.demo is not None else self.course

    def step(self, view: Optional[dict], trigger, value=None, request: Optional[str] = None, clicked=None) -> tuple[dict, list]:
        """Demo and End demo, else the step of the board the view names (Dashboard.step). Demo gives deck 00's board a
        fresh state with card 001 shown, End demo drops it; the view starts clean either way, the flag set or cleared.
        Outside the demo deck 00's state is dropped (a reload ended it); a flag with deck 00's state fresh starts the
        view clean (a restart ended it). Without course data, outside the demo, only the tiles answer."""
        view = {**HOME, **(view if isinstance(view, dict) else {})}
        view["demo"] = bool(view["demo"]) and self.demo is not None
        if view["demo"] and self.demo.state.fresh:
            view = dict(HOME)
        if trigger == DEMO_BUTTON and self.demo is not None:
            running = not view["demo"]
            if running:
                self.demo.restart()
            else:
                self.demo.end()
            return {**HOME, "demo": running}, []
        if not view["demo"] and self.demo is not None and not self.demo.state.fresh:
            self.demo.end()
        board = self.board(view)
        if board is None:
            if view["panel"] not in (None, *PANEL_LETTERS):
                view["panel"] = None
            if tile_letter(trigger) is not None:
                toggle_panel(view, tile_letter(trigger))
            return view, []
        return board.step(view, trigger, value, request, clicked)

    def save(self) -> None:
        """The state file of a board whose state changed in the last step (deck 00's has none)."""
        for board in (self.course, self.demo):
            if board is not None and board.changed:
                board.state.save()
                board.changed = False

    def render(self, view: dict, checks: Sequence) -> tuple:
        """Every region in OUTPUT_NAMES' order: the board the view names (deck 00's, fresh and empty, without course
        data outside the demo) with the slide row's tiles and panel, then Demo and the request box."""
        board = self.board(view)
        drawn = board if board is not None else self.demo
        regions = drawn.render(view, checks, self.slide, BEGIN_HINT if board is not None else NO_DATA_HINT)
        return regions + demo_controls(view["demo"], self.demo is not None, self.course is None)


def register_callback(app: Dash, board):
    """The dashboard's callbacks on `board` (Boards, or the course's Dashboard alone); returns the page's (for tests).
    Its inputs: the card's click, the hidden input Enter writes, Confirm, Related and its slots, Back, Demo, Go and
    Enter in the request box, Go's slots, the five tiles and the hidden input a click in the diagram writes
    (INPUT_IDS); its states the checks, the box's text and the view. It steps the state and the view (Boards.step),
    saves a state file when its state changed, and renders every region (Boards.render). The second callback sets
    Confirm's disabled state from the checks, at every check and whenever the page's callback sets them."""
    boards = board if isinstance(board, Boards) else Boards(board)

    @app.callback(
        Output("view", "data"),
        Output("card-where", "children"),
        Output("card-face", "children"),
        Output("card-face", "className"),
        Output("quiz-control", "className"),
        Output("quiz-options", "options"),
        Output("quiz-options", "value"),
        Output("related-toggle", "className"),
        Output("related-toggle", "disabled"),
        *[output for index in range(RELATED_SLOTS) for output in (Output(f"{RELATED_PREFIX}{index}", "children"), Output(f"{RELATED_PREFIX}{index}", "className"))],
        Output("prompt-message", "children"),
        Output("prompt-answer", "className"),
        *[output for index in range(GO_SLOTS) for output in (Output(f"{GO_PREFIX}{index}", "children"), Output(f"{GO_PREFIX}{index}", "className"))],
        Output("diagram", "children"),
        Output("back", "disabled"),
        *[Output(f"{TILE_PREFIX}{letter}", "className") for letter in PANEL_LETTERS],
        Output("model-panel", "children"),
        Output("model-panel", "className"),
        Output(DEMO_BUTTON, "children"),
        Output(DEMO_BUTTON, "className"),
        Output(DEMO_BUTTON, "disabled"),
        Output("prompt-text", "placeholder"),
        Output("prompt-text", "disabled"),
        Output("prompt-go", "disabled"),
        *[Input(component, "value" if component in VALUE_INPUTS else ("n_submit" if component == "prompt-text" else "n_clicks")) for component in INPUT_IDS],
        State("quiz-options", "value"),
        State("prompt-text", "value"),
        State("view", "data"),
    )
    def update(*values):
        from dash import ctx

        checks, request, view = values[-3:]
        view, checks = boards.step(view, ctx.triggered_id, checks, request, values[INPUT_IDS.index(DIAGRAM_INPUT)])
        boards.save()
        return boards.render(view, checks)

    @app.callback(Output("quiz-confirm", "disabled"), Input("quiz-options", "value"))
    def confirm(checks):
        return quiz.confirm_disabled(checks)

    return update


def create_app(board: Optional[Dashboard], demo: Optional[DemoBoard] = None) -> Dash:
    """The Dash app of the dashboard on the course's board and deck 00's (either may be missing, not both), in the one
    theme (assets/style.css), with its callback; the tiles' content is built once at the layout."""
    boards = Boards(board, demo)
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.index_string = app.index_string.replace("<html>", '<html lang="en">', 1)  # the stylesheet's hyphens need the language
    app.layout = page_layout(HOME, boards.slide.tiles(), has_demo=demo is not None, no_data=board is None)
    register_callback(app, boards)
    return app
