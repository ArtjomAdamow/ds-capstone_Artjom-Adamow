"""The dashboard (app.dashboard): the capstone's app without the viewer, one card at a time with what a learner
sees of the models, on the dashboard's own state (Dash; nothing of the viewer, nothing of the glue).

Dashboard fits the course model with T1, D1 and P1 on the course table once, as the models view's Page does, and
reads three things of them: the shown card's related cards from the index (CourseModel.related, the five nearest),
D1's order among the unread quizzes (the next card), and P1's answer to a request (PromptModel.find); nothing else
of the models shows on the card: no level, no margin, no score, no tag. Where a card lies is T1's membership, the
model's topics as the full app gives every deck from the start; the topics keep the scripted ids and names.

* The card (one at a time, as the viewer shows it): the front the question, a click on the card or Enter turns it
  to the answer; a quiz's front the question with its options and Confirm under it, Confirm disabled while nothing
  is checked, its back the judgement (quiz.judge), Enter on the back turns it to the front with the checks cleared.
  The turn marks the card read (a quiz once judged). With no card shown the card's place says how to begin.
* The navigation: Related lists the shown card's five related cards by the viewer's rule (neighbours.rank_by_confidence
  without the confidence: the unread first, the quizzes D1 predicted in D1's order among them, the read ones after
  them in the neighbours' order), the next card first and marked and again in its place, each with where it lies
  (this topic, this deck, another deck by name), every entry a click that shows that card; nothing is explored when
  the list is drawn. Back returns to the card shown before (the trail, state.DashboardState.back), disabled on an
  empty trail. The request box is the one way in: P1's answer shows the topic's card where it navigates (the
  selected card, else the topic's nearest), else "no match" with up to three entries, the nearest topics' cards,
  each a click; an empty request does nothing.
* What explores a card (step 33's rule on the dashboard's parts): a selection in any way, Go's answer that shows a
  card, a click on a Related entry or on one of Go's entries, Back; nothing else and nothing ahead.
* The diagram (layout.diagram): decks → topics → cards, the explored decks, the shown card's deck's topics that hold
  an explored card, that topic's explored cards, nothing unexplored; the shown card's deck, topic and card
  highlighted; empty at a fresh start, filling as the learner explores; no click.
* The slide row: five tiles, S, C, T, D and P, each the generation the app runs (S2 with the encoder, else S1; C1,
  T1, D1, P1) with its headline and its two numbers with their verdict words from the results files
  (standalone.headlines, the same numbers and words as the models view), one line where a results file is missing;
  nothing live. A click on a tile opens the model's panel over the two columns, the tiles staying visible; another
  tile shows that model in its place, a second click on the open tile closes it; Enter does nothing while it is open.
* The state: the read and explored flags, the shown card and the trail in the dashboard's own state file
  (state.DashboardState, saved at every change); the flip, Related drawn, the last request and the open tile in the
  `view` store (HOME's keys), runtime only. The closed start: a fresh start shows nothing, a restart shows the
  explored cards and the shown card again. No episodes' log.

register_callback is the one callback: every button and the hidden input Enter writes are its inputs, the checks,
the request box's text and the view its states; it renders every region from the state and the view (render).
"""

from pathlib import Path
from typing import Optional, Sequence

from dash import Dash, Input, Output, State

from ..course_table import CourseTable, DeckRow
from ..models import embeddings, similar
from ..models.course_model import CourseModel, TopicLayout
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.prompt_model import PromptModel, variant_label
from ..neighbours import rank_by_confidence, shorten
from ..standalone.app import d2_label, s2_label
from ..standalone.headlines import WHAT, headlines
from ..standalone.results import LEVEL_FILES, Results, course_note
from . import quiz
from .layout import (
    GO_SLOTS, KEY_INPUT, PANEL_LETTERS, PANEL_NAMES, RELATED_SLOTS, THIS_DECK, THIS_TOPIC, Dot, Entry, Tile, card_face, card_where,
    diagram, go_slots, model_panel, page_layout, prompt_answer, quiz_control_class, quiz_options, related_slots, related_toggle, tile_class,
)
from .state import DashboardState

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"  # next to evaluate.py: the capstone's root
TITLE = "Guided learning: the dashboard"
# The view store: what is not saved. `request`: the last request as {"text", "match", "entries"}, the entries of
# a "no match" as [deck id, topic id, card id] each
HOME = {"back": False, "related": False, "request": None, "panel": None}
TURN_TRIGGERS = ("card-face", KEY_INPUT)  # a click on the card, Enter
PROMPT_TRIGGERS = ("prompt-go", "prompt-text")  # Go, and Enter in the request box (its n_submit)
RELATED_PREFIX, GO_PREFIX, TILE_PREFIX = "related-", "go-", "tile-"
# The callback's outputs by name, in order (OUT gives each one's place in the result)
OUTPUT_NAMES = (
    ["view", "where", "face", "face_class", "quiz_class", "options", "checks", "confirm_disabled", "related_class", "related_disabled"]
    + [name for index in range(RELATED_SLOTS) for name in (f"related_{index}", f"related_{index}_class")]
    + ["message", "answer_class"]
    + [name for index in range(GO_SLOTS) for name in (f"go_{index}", f"go_{index}_class")]
    + ["diagram", "back_disabled"]
    + [f"tile_{letter}" for letter in PANEL_LETTERS]
    + ["panel", "panel_class"]
)
OUT = {name: index for index, name in enumerate(OUTPUT_NAMES)}
# The callback's inputs, in order: the card's click, Enter, Confirm, Related, its slots, Back, Go, Enter in the box,
# Go's slots, the tiles; then the states: the checks, the box's text, the view
INPUT_IDS = (
    ["card-face", KEY_INPUT, "quiz-confirm", "related-toggle"] + [f"{RELATED_PREFIX}{index}" for index in range(RELATED_SLOTS)]
    + ["back", "prompt-go", "prompt-text"] + [f"{GO_PREFIX}{index}" for index in range(GO_SLOTS)] + [f"{TILE_PREFIX}{letter}" for letter in PANEL_LETTERS]
)
INPUTS = len(INPUT_IDS)


def slot_of(trigger, prefix: str) -> Optional[int]:
    """The slot's index of a trigger like "related-2"; None for another trigger."""
    if not isinstance(trigger, str) or not trigger.startswith(prefix):
        return None
    rest = trigger[len(prefix):]
    return int(rest) if rest.isdigit() else None


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
        index = self.model.index
        encoded = isinstance(index.vectoriser, (embeddings.HybridVectoriser, embeddings.SentenceEncoder))
        self.levels = {"S": "S2" if encoded else "S1", "C": "C1", "T": "T1", "D": "D1", "P": "P1"}  # the generation the app runs
        chosen = {"P1": variant_label(index.vectoriser), "D2": d2_label()}
        if s2_label() is not None:
            chosen["S2"] = s2_label()
        self.results = Results(results_dir, course_note(table.deck_ids), None, chosen=chosen)

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

    def diagram(self) -> tuple[list[Dot], list[Dot], list[Dot]]:
        """The three rows: the explored decks in deck id order; the shown card's deck's topics that hold an explored
        card, in file order; that topic's explored cards, in the topic's order; the shown card's highlighted."""
        explored = {card_id for card_id in self.state.explored_ids if card_id in self.card_by_id}
        shown = self.state.shown if self.state.shown in self.card_by_id else None
        shown_deck = self.model.deck_of.get(shown) if shown else None
        shown_topic = self.topic_of(shown)
        deck_ids = sorted({self.model.deck_of[card_id] for card_id in explored if card_id in self.model.deck_of})
        decks = [Dot(deck_id, self.deck_by_id[deck_id].name if deck_id in self.deck_by_id else deck_id, deck_id == shown_deck) for deck_id in deck_ids]
        topics: list[Dot] = []
        cards: list[Dot] = []
        if shown_deck is not None:
            for placed in self.model.topics_of_deck(shown_deck):
                if any(card_id in explored for card_id in placed.ids):
                    topics.append(Dot(placed.topic_id, placed.name, shown_topic is not None and placed.topic_id == shown_topic.topic_id))
        if shown_topic is not None:
            cards = [Dot(card_id, shorten(self.card_by_id[card_id].question), card_id == shown) for card_id in shown_topic.ids if card_id in explored]
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

    # --- the moves ---------------------------------------------------------------------------------------------

    def step(self, view: Optional[dict], trigger, value=None, request: Optional[str] = None) -> tuple[dict, list]:
        """The view and the checks after a trigger: a click on the card or Enter turns it (a quiz's Confirm with
        checks, Enter on its back to the front with the checks cleared; nothing while a panel is open), Confirm
        judges, Related draws or hides the list, one of its entries shows that card, Back the card before, Go or
        Enter in the box runs the request, one of Go's entries shows that card, a tile opens its panel or closes it.
        A selection in any way explores the card, puts the front up, hides the list and clears the checks."""
        view = {**HOME, **(view if isinstance(view, dict) else {})}
        if view["panel"] not in (None, *PANEL_LETTERS):
            view["panel"] = None
        if not isinstance(view["request"], dict):
            view["request"] = None
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
            view["back"], view["related"], checks = False, False, []

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
                view["back"], view["related"], checks = False, False, []
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
        elif isinstance(trigger, str) and trigger.startswith(TILE_PREFIX) and trigger[len(TILE_PREFIX):] in PANEL_LETTERS:
            letter = trigger[len(TILE_PREFIX):]
            view["panel"] = None if view["panel"] == letter else letter
        return view, checks

    def render(self, view: dict, checks: Sequence) -> tuple:
        """Every region from the state and the view, in OUTPUT_NAMES' order."""
        shown = self.state.shown if self.state.shown in self.card_by_id else None
        card = self.card_by_id.get(shown)
        checks = list(checks or [])
        face, face_class = card_face(card, view["back"], checks)
        entries = self.related_entries(shown) if card is not None and view["related"] else []
        go_entries = self.request_entries(view["request"])
        decks, topics, cards = self.diagram()
        tile = self.tile(view["panel"]) if view["panel"] in PANEL_LETTERS else None
        return (
            view,
            card_where(self.where(shown)),
            face,
            face_class,
            quiz_control_class(card, view["back"]),
            quiz_options(card),
            checks,
            quiz.confirm_disabled(checks),
            *related_toggle(card is not None, view["related"]),
            *related_slots(entries, view["related"]),
            *prompt_answer(view["request"], go_entries),
            *go_slots(go_entries),
            diagram(decks, topics, cards),
            not self.state.can_go_back,
            *[tile_class(letter == view["panel"]) for letter in PANEL_LETTERS],
            *model_panel(tile),
        )


def register_callback(app: Dash, board: Dashboard):
    """The dashboard's one callback; returns the function (for tests). Its inputs: the card's click, the hidden
    input Enter writes, Confirm, Related and its slots, Back, Go and Enter in the request box, Go's slots and the
    five tiles (INPUT_IDS); its states the checks, the box's text and the view. It steps the state and the view
    (Dashboard.step), saves the state file when the state changed, and renders every region (Dashboard.render)."""

    @app.callback(
        Output("view", "data"),
        Output("card-where", "children"),
        Output("card-face", "children"),
        Output("card-face", "className"),
        Output("quiz-control", "className"),
        Output("quiz-options", "options"),
        Output("quiz-options", "value"),
        Output("quiz-confirm", "disabled"),
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
        *[Input(component, "value" if component == KEY_INPUT else ("n_submit" if component == "prompt-text" else "n_clicks")) for component in INPUT_IDS],
        State("quiz-options", "value"),
        State("prompt-text", "value"),
        State("view", "data"),
    )
    def update(*values):
        from dash import ctx

        checks, request, view = values[-3:]
        view, checks = board.step(view, ctx.triggered_id, checks, request)
        if board.changed:
            board.state.save()
            board.changed = False
        return board.render(view, checks)

    return update


def create_app(board: Dashboard) -> Dash:
    """The Dash app of the dashboard, in the one theme (assets/style.css), with its callback; the tiles' content is
    built once at the layout."""
    app = Dash(__name__, assets_folder=str(ASSETS_DIR), title=TITLE)
    app.layout = page_layout(HOME, board.tiles())
    register_callback(app, board)
    return app
