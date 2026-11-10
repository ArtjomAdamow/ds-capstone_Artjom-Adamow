"""The dashboard's regions (Dash), in the one theme (assets/style.css holds the
tokens and the type scale). Landscape only: the top bar, under it two columns at the screen's height less the slide
row, the slide row along the bottom at the full width, a model's panel over the two columns while a tile is open.

* the top bar: the title; the request box (`prompt-text`, Enter submits, and Go `prompt-go`), the one way in, with
  its answer hanging under it (`prompt-answer`: the message `prompt-message` and up to GO_SLOTS entries `go-<n>`,
  each a click to its card; hidden while there is nothing to say); Back (`back`), disabled on an empty trail; Demo
  (`demo`), "End demo" while the demo runs, disabled without deck 00's table; without course data, outside the demo,
  the box says NO_DATA and the box and Go are disabled;
* the diagram (`diagram`), the left column: three rows, decks → topics → cards, each a row of items in plain HTML,
  a dot with its name, the shown card's deck, topic and card highlighted, the open deck and topic marked; a click on
  an item goes to the callback through the hidden input `diagram-click` (assets/diagram.js);
* the card, the right column: with no card how to begin (BEGIN_HINT; NO_DATA_HINT without course data); its title
  row (`card-where`, the deck and the topic by name, and the Related button `related-toggle`), the card's frame with
  the card's face (`card-face`: the front or the back; a click on it turns a flashcard) and, under a quiz's question
  in the same frame, the options (`quiz-options`, the checks live in its value) with Confirm (`quiz-confirm`), the
  one scroll bar of the page inside the frame; under the frame the Related list, RELATED_SLOTS fixed buttons
  (`related-<n>`), the next card first and marked, hidden while unused;
* the slide row (`tile-S` to `tile-P`): one tile per model with its headline and its two numbers with their verdict
  words, or one line where the results file is missing; a click opens the model's panel (`model-panel`) over the two
  columns, the tiles staying visible under it: the numbers on its left (`panel-text`), the model's figure on its
  right (`panel-figure`);
* the hidden inputs `key-enter`, written by assets/keys.js on Enter outside the request box and the buttons, and
  `diagram-click`, written by assets/diagram.js on a click in the diagram; the `view` store (app.HOME's keys) with
  what is not saved: the flip, Related drawn, the last request, the open tile, the diagram's open deck and topic,
  the demo flag.

What the regions show comes from app.Dashboard; the state is its own (state.DashboardState).
"""

from typing import NamedTuple, Optional, Sequence

from dash import dcc, html

from ..headlines import Headline
from ..panels import PANEL_LETTERS, PANEL_NAMES
from . import quiz

KEY_INPUT = "key-enter"  # the hidden input Enter writes (assets/keys.js)
DIAGRAM_INPUT = "diagram-click"  # the hidden input a click in the diagram writes (assets/diagram.js)
DEMO_BUTTON = "demo"  # Demo and End demo, in the top bar after Back
DIAGRAM_ROWS = ("deck", "topic", "card")  # the rows' names in a click, by the rows' order
RELATED_SLOTS = 6  # the next card, marked, then the five neighbours
GO_SLOTS = 3  # the nearest three topics of a "no match"
EYEBROW = "GUIDED LEARNING · THE DASHBOARD"
TITLE = "One card at a time."
PLACEHOLDER = "A question, a concept or a code name"
BEGIN_HINT = "Type a request in the box above and press Go: the card it finds is shown here."
NO_DATA = "No data loaded"  # the box's line without course data, outside the demo
NO_DATA_HINT = "No course data loaded. Demo, in the top bar, shows how the dashboard works on a deck of its own."
DEMO_LABEL = "Demo"
END_DEMO_LABEL = "End demo"
NO_MATCH = "No match in the course."
NO_MATCH_NEAREST = "No match in the course; the nearest topics, each a click:"
RELATED_LABEL = "Related"
NEXT_MARK = "next"
THIS_TOPIC = "this topic"
THIS_DECK = "this deck"
FLASHCARD_HINT = "Click the card or press Enter to turn it."
QUIZ_FRONT_HINT = "Check one or more options, then Confirm or press Enter."
BACK_HINT = "Press Enter to turn the card back."
ROW_LABELS = ("Decks", "Topics", "Cards")
PANEL_HINT = "Click the open tile to close this panel; another tile shows its model here."


class Entry(NamedTuple):
    """A Related entry or one of Go's: the card, its label (the question shortened, or the deck and the topic with
    it), where it lies, and whether it is the marked next card."""

    card_id: str
    label: str
    place: str = ""
    marked: bool = False


class Dot(NamedTuple):
    """An item of the diagram: its id, its name, whether it is the shown card's, whether it is the open deck or topic
    (the one whose topics or cards the next row lists), and a topic's deck."""

    dot_id: str
    name: str
    current: bool = False
    opened: bool = False
    deck: str = ""


class Tile(NamedTuple):
    """A tile of the slide row, and its panel: the model letter, its name, the generation the numbers are of, the
    headline (what the model does), its two numbers, or the one line where there are none."""

    letter: str
    name: str
    level: str
    what: str
    numbers: tuple[Headline, ...] = ()
    line: str = ""


# --- the card ----------------------------------------------------------------------------------------------------


def card_where(text: str) -> list:
    """The card's title row's left part: the deck and the topic by name; nothing with no card."""
    return [html.Span(text, className="card-where-text")] if text else []


def card_face(card, back: bool, value, hint: str = BEGIN_HINT) -> tuple[list, str]:
    """The card's face and its class: with no card how to begin (`hint`); a flashcard's question or its answer; a
    quiz's question (the options stand under it, quiz_control) or its judgement (quiz.judge): the card's feedback
    under "Correct answer", or every checked option with its own feedback under "Wrong answer"."""
    if card is None:
        return [html.P(hint, className="empty-state")], "card-face empty"
    if quiz.has_options(card):
        if not back:
            return [html.Div(quiz.QUESTION_LABEL, className="card-label"), dcc.Markdown(card.question, className="card-content")], "card-face quiz front"
        right, blocks = quiz.judge(card, value)
        if right:
            body = [dcc.Markdown(card.answer, className="card-content")]
        else:
            body = [html.Div(
                [
                    html.Div(
                        [dcc.Markdown(block.option, className="quiz-feedback-option"), dcc.Markdown(block.answer, className="quiz-feedback-answer")],
                        className="quiz-feedback-block " + ("quiz-feedback-correct" if block.correct else "quiz-feedback-wrong"),
                    )
                    for block in blocks
                ],
                className="quiz-feedback",
            )]
        label = quiz.CORRECT_LABEL if right else quiz.WRONG_LABEL
        return [html.Div(label, className="card-label")] + body + [html.Div(BACK_HINT, className="card-hint")], "card-face quiz back"
    if back:
        return [html.Div(quiz.ANSWER_LABEL, className="card-label"), dcc.Markdown(card.answer, className="card-content"),
                html.Div(BACK_HINT, className="card-hint")], "card-face back"
    return [html.Div(quiz.QUESTION_LABEL, className="card-label"), dcc.Markdown(card.question, className="card-content"),
            html.Div(FLASHCARD_HINT, className="card-hint")], "card-face front"


def quiz_control_class(card, back: bool) -> str:
    """The options and Confirm under a quiz's question; hidden on the back and for anything else."""
    return "quiz-control" if card is not None and quiz.has_options(card) and not back else "quiz-control hidden"


def quiz_options(card) -> list[dict]:
    """The checklist's options: one per answer option in file order, the text as Markdown, valued by index."""
    if card is None or not quiz.has_options(card):
        return []
    return [{"label": dcc.Markdown(text, className="quiz-option-text"), "value": index} for index, text in enumerate(card.options)]


def quiz_control() -> html.Div:
    """The quiz's control in the layout from the start (the checklist is a callback state, Confirm an input): the
    dividing line, the checklist, Confirm and the hint."""
    return html.Div(
        [
            html.Div(className="quiz-divider"),
            dcc.Checklist(id="quiz-options", options=[], value=[], className="quiz-options", inputClassName="quiz-checkbox",
                          labelClassName="quiz-option"),
            html.Div(html.Button("Confirm", id="quiz-confirm", n_clicks=0, className="top-button quiz-confirm", disabled=True),
                     className="quiz-confirm-row"),
            html.Div(QUIZ_FRONT_HINT, className="card-hint"),
        ],
        id="quiz-control", className=quiz_control_class(None, False),
    )


# --- Related and Go's entries ------------------------------------------------------------------------------------


def related_toggle(has_card: bool, drawn: bool) -> tuple[str, bool]:
    """The Related button's class and disabled state: open while the list is drawn, disabled with no card."""
    return ("related-toggle open" if drawn else "related-toggle"), not has_card


def entry_children(entry: Entry) -> list:
    """An entry's content: the mark "next" where it is the next card, its label, where it lies."""
    parts = []
    if entry.marked:
        parts.append(html.Span(NEXT_MARK, className="entry-mark"))
    parts.append(html.Span(entry.label, className="entry-label"))
    if entry.place:
        parts.append(html.Span(entry.place, className="entry-place"))
    return parts


def related_slots(entries: Sequence[Entry], drawn: bool) -> list:
    """The RELATED_SLOTS buttons' children and classes, flat in slot order: an entry each while the list is drawn,
    the marked one first; the slots beyond the entries hidden."""
    flat: list = []
    for index in range(RELATED_SLOTS):
        if drawn and index < len(entries):
            entry = entries[index]
            flat += [entry_children(entry), "related-entry marked" if entry.marked else "related-entry"]
        else:
            flat += [[], "related-entry hidden"]
    return flat


def prompt_answer(request: Optional[dict], entries: Sequence[Entry]) -> tuple[list, str]:
    """The message under the request box and the answer's class: "no match" with its entries' line; nothing after
    a match (the card shows) or before a request."""
    if not request or request.get("match"):
        return [], "prompt-answer hidden"
    return [html.Div(NO_MATCH_NEAREST if entries else NO_MATCH, className="prompt-message-text")], "prompt-answer"


def go_slots(entries: Sequence[Entry]) -> list:
    """The GO_SLOTS buttons' children and classes, flat in slot order; the slots beyond the entries hidden."""
    flat: list = []
    for index in range(GO_SLOTS):
        if index < len(entries):
            flat += [entry_children(entries[index]), "go-entry"]
        else:
            flat += [[], "go-entry hidden"]
    return flat


# --- the diagram -------------------------------------------------------------------------------------------------


def diagram_item(row: str, dot: Dot) -> html.Span:
    """An item of a row: the dot and its name (the full name on hover too), its row, its id and a topic's deck as
    data for assets/diagram.js; a span, so the focus stays where Enter turns the card."""
    data = {"data-row": row, "data-dot": dot.dot_id}
    if dot.deck:
        data["data-deck"] = dot.deck
    class_name = "diagram-item" + (" current" if dot.current else "") + (" open" if dot.opened else "")
    return html.Span([html.Span(className="dot"), html.Span(dot.name, className="dot-name")], className=class_name, title=dot.name, **data)


def diagram(decks: Sequence[Dot], topics: Sequence[Dot], cards: Sequence[Dot]) -> list:
    """The three rows, decks → topics → cards: each a label and its items, a dot with its name, the shown card's
    deck, topic and card highlighted, the open deck and topic marked."""
    rows = []
    for label, row, dots in zip(ROW_LABELS, DIAGRAM_ROWS, (decks, topics, cards)):
        rows.append(html.Div(
            [html.Div(label, className="diagram-label"), html.Div([diagram_item(row, dot) for dot in dots], className="diagram-dots")],
            className="diagram-row",
        ))
    return rows


# --- the top bar's demo ------------------------------------------------------------------------------------------


def demo_controls(running: bool, has_demo: bool, no_data: bool) -> tuple:
    """Demo's label, class and disabled state, then the request box's placeholder and disabled state and Go's: "End
    demo" while the demo runs, Demo disabled without deck 00's table; without course data, outside the demo, the box
    says NO_DATA, and the box and Go are disabled."""
    closed = no_data and not running
    return (
        END_DEMO_LABEL if running else DEMO_LABEL,
        "top-button demo-toggle open" if running else "top-button demo-toggle",
        not has_demo,
        NO_DATA if closed else PLACEHOLDER,
        closed,
        closed,
    )


# --- the slide row and its panel ---------------------------------------------------------------------------------


def number_block(headline: Headline, large: bool) -> html.Div:
    """One of a model's two numbers: on a tile the value at the heading size, its label and its verdict word; on the
    panel the value at the number size with its label, its contrast, its sentence, its verdict word and its
    explanation."""
    parts = [html.Div(headline.shown, className="number-value"), html.Div(headline.label, className="number-label")]
    if large:
        contrast = []
        if headline.of:
            contrast.append(html.Span(headline.of, className="number-of"))
        if headline.contrast is not None:
            contrast.append(html.Span(f"against {headline.contrast_shown}", className="number-contrast"))
        if headline.gain:
            contrast.append(html.Span(headline.gain, className="number-gain"))
        parts.append(html.Div(contrast, className="number-contrast-line"))
        parts.append(html.P(headline.sentence, className="number-sentence"))
    verdict = [html.Span(headline.verdict, className="verdict-word")]
    if large and headline.explanation:
        verdict.append(html.Span(f": {headline.explanation}", className="verdict-words"))
    parts.append(html.Div(verdict, className="number-verdict"))
    return html.Div(parts, className="number-block large" if large else "number-block")


def tile_numbers(tile: Tile, large: bool) -> html.Div:
    """The tile's or the panel's numbers, or the one line without a results file."""
    if not tile.numbers:
        return html.Div(tile.line, className="tile-line")
    return html.Div([number_block(headline, large) for headline in tile.numbers], className="numbers")


def tile_button(tile: Tile) -> html.Button:
    """A tile of the slide row: the letter with the model's name, the headline, the two numbers with their verdict
    words; its content stands from the start, nothing live."""
    return html.Button(
        [
            html.Div([html.Span(tile.letter, className="tile-letter"), html.Span(tile.name, className="tile-name")], className="tile-head"),
            html.Div(tile.what, className="tile-what"),
            tile_numbers(tile, large=False),
        ],
        id=f"tile-{tile.letter}", n_clicks=0, className=tile_class(False), title=f"{tile.name}: open or close its panel",
    )


def tile_class(is_open: bool) -> str:
    return "tile open" if is_open else "tile"


def model_panel(tile: Optional[Tile], figure=None) -> tuple[list, str]:
    """The open model's panel over the two columns, or hidden; its body on the left the letter and the name, the
    headline at the heading size, the two numbers one under the other at the number size with their parts,
    and how to close it; on the right the model's figure (app.Dashboard.panel_figure) at the panel's full height."""
    if tile is None:
        return [], "model-panel hidden"
    text = html.Div(
        [
            html.Div([html.Span(tile.letter, className="tile-letter"), html.Span(tile.name, className="panel-name"),
                      html.Span(tile.level, className="panel-level")], className="panel-head"),
            html.Div(tile.what, className="panel-what"),
            tile_numbers(tile, large=True),
            html.Div(PANEL_HINT, className="card-hint panel-hint"),
        ],
        className="panel-text",
    )
    figure_part = html.Div(figure if figure is not None else [], className="panel-figure")
    return [html.Div([text, figure_part], className="panel-body")], "model-panel"


# --- the page ----------------------------------------------------------------------------------------------------


def page_layout(home: dict, tiles: Sequence[Tile], has_demo: bool = False, no_data: bool = False) -> html.Main:
    """The page: the top bar with the request box, its answer, Back and Demo; the diagram and the card with its
    Related list; the slide row's tiles; the model's panel; the hidden inputs of Enter and of the diagram's clicks and
    the `view` store. `has_demo`: deck 00's table is there; `no_data`: the course table is not."""
    label, demo_class, demo_disabled, placeholder, box_disabled, go_disabled = demo_controls(False, has_demo, no_data)
    return html.Main(
        [
            html.Header(
                [
                    html.Div([html.P(EYEBROW, className="eyebrow"), html.H1(TITLE)], className="topbar-title"),
                    html.Div(
                        [
                            dcc.Input(id="prompt-text", type="text", value="", n_submit=0, placeholder=placeholder, className="prompt-input",
                                      autoComplete="off", disabled=box_disabled),
                            html.Button("Go", id="prompt-go", n_clicks=0, className="top-button prompt-go", disabled=go_disabled),
                            html.Div(
                                [html.Div(id="prompt-message", className="prompt-message")]
                                + [html.Button([], id=f"go-{index}", n_clicks=0, className="go-entry hidden") for index in range(GO_SLOTS)],
                                id="prompt-answer", className="prompt-answer hidden",
                            ),
                        ],
                        className="prompt-box",
                    ),
                    html.Div(
                        [html.Button("Back", id="back", n_clicks=0, className="top-button", disabled=True),
                         html.Button(label, id=DEMO_BUTTON, n_clicks=0, className=demo_class, disabled=demo_disabled)],
                        className="topbar-actions",
                    ),
                ],
                className="topbar",
            ),
            html.Section(
                [
                    html.Div(id="diagram", className="diagram"),
                    html.Div(
                        [
                            html.Div(
                                [html.Div(id="card-where", className="card-where"),
                                 html.Button(RELATED_LABEL, id="related-toggle", n_clicks=0, className="related-toggle", disabled=True)],
                                className="card-head",
                            ),
                            html.Div([html.Div(id="card-face", className="card-face empty", n_clicks=0), quiz_control()], className="card-frame"),
                            html.Div([html.Button([], id=f"related-{index}", n_clicks=0, className="related-entry hidden") for index in range(RELATED_SLOTS)],
                                     className="related-list"),
                        ],
                        className="card-column",
                    ),
                    html.Div(id="model-panel", className="model-panel hidden"),
                ],
                className="workspace",
            ),
            html.Section([tile_button(tile) for tile in tiles], className="slide-row"),
            dcc.Store(id="view", data=dict(home)),
            dcc.Input(id=KEY_INPUT, value="0", type="text", className="hidden-input"),
            dcc.Input(id=DIAGRAM_INPUT, value="", type="text", className="hidden-input"),
        ],
        className="app-shell",
    )


PANEL_LETTERS = PANEL_LETTERS
PANEL_NAMES = PANEL_NAMES
