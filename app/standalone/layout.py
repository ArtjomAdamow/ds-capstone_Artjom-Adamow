"""The standalone page's layout, map and panels (Dash and plotly; nothing of the viewer), written anew after the
viewer's in its theme (assets/style.css holds the tokens):

* the top bar: the title, Back and Open;
* the map (`map`), a square of 12.5 % of the window's area (assets/map_side.js): the view's dots on the sphere, and
  what the expanded model generation adds (app.Page decides, map_figure draws): with S1 or S2 the generation's
  neighbours (every dot coloured by the share of its cards' neighbours in its own deck or topic, a card by the share
  in the entered topic, on the figures' blue ramp, a dot without a neighbour an open grey circle; in the cards view
  the neighbour pairs inside the topic as lines along the surface, the selected card's bright); with C the
  connections below the preset gold and dashed (straight between decks, along the surface between cards), the
  centroid of a deck and the axes with their ticks, the map's own coordinates, the unit sphere's; with T the
  centroid, the axes and T1's markers (a card's margin on a teal scale, a cross below the threshold, the diamonds of
  the moved cards, a topic's dot grown with its share of cards between two topics); with D1 or D2 the cards coloured
  as the viewer colours them after Helper: a rated quiz by its level, a quiz the generation predicted in the predicted
  colour as a cross (its level on the hover), a flashcard grey, with the legend inside the map (plotly's, at the top
  left); with every panel collapsed grey dots at random positions, nothing else;
* window A (`dot-panel`), under the map (beside it in portrait): the selected dot's ids and counts, a click on a
  deck's or a topic's enters it; with nothing selected the dendrogram of the view's membership with the preset's
  cut-off line, its leaves by id (figures.dendrogram_figure);
* the difficulty window (`difficulty-window`), under window A at its width in both orientations, shown with D
  expanded only: the expanded generation's confusion matrix and the shares of the levels of the view's part of the
  course (metrics_window);
* the model panels (`box-S` to `box-D`), a column right of all three (under them in portrait): one per model letter, a
  row with its letter and name (`panel-S`, a click expands it and collapses any other) and a close box (`close-S`);
  expanded, a row per generation in a fixed order, the oldest on top (S1 above S2, D1 above D2; C1 and T1 alone:
  GENERATIONS), the newest expanded by default; a click on another generation's row expands it in place and collapses
  the expanded one, the order unchanged (generation_rows). An expanded generation holds
  what it is, its live measures and the tables of the newest run of its rows, one row per measure and one column per
  variant with the chosen one marked. A click on a measure's row opens its description under it (what the measure
  is, its range, from app/docs/columns.md), one row at a time: a click on another row moves it, a click on the same
  row or the description's close box closes it. Both clicks go through the hidden input `row-click`
  (assets/panel_rows.js sets it); the expanded generation and the one open row live in the `view` store. The panels'
  text is at double size (style.css, `--panel-font`).

No learning material shows: no card's question, answer or explanation, no deck's description; a card is its
comb_id. The map's hover labels and the line under the top bar are ids too: a deck's id, a topic's label (its topic
id and a hash of the name, app.Page.topic_label), a card's comb_id.

What the views show comes from app.Page; the view itself lives in the `view` store (app.HOME's keys).
"""

from typing import Mapping, Optional, Sequence

import plotly.graph_objects as go
from dash import dcc, html

from ..cards import STACK_DIFFICULTY, STACK_PLAIN, STACK_QUIZ, STACK_SOURCE
from ..figures import (
    CUT_COLOR, FONT, GOLD, GRID, INK, MUTED, PREDICTED_COLOR, SEQUENTIAL, confusion_figure, metrics_subtitle, score_header,
    score_rows, shares_figure, sphere_surface, surface_arc,
)
from .results import Table

CARDS_VIEW = "cards"
TOPICS_VIEW = "topics"
DECKS_VIEW = "decks"
BACKGROUND = "#102a43"
GREY = "#7c8b99"  # the viewer's colour of a dot in no group: a flashcard, the cards' sphere
LEVEL_COLORS = {"easy": "#4f7fcf", "medium": "#b8956a", "hard": "#ff5349"}  # blue, beige-brown, red-orange
# The quizzes D1 predicted: the viewer's "Quiz difficulty: predicted" group (glue.difficulty_groups, the oldest group,
# so its colour decides there too) and the predicted series of the shares figure share this amber
PREDICTED_SYMBOL = "cross"
PREDICTED = "predicted"
FLASHCARD = "flashcard"
UNRATED = "unrated quiz"  # an unrated quiz without a prediction (D1 without metrics)
DECK_COLOR = MUTED  # the decks in one colour
TOPIC_COLOR = MUTED
LINK_WIDTH = 4
DOT_SIZE = {DECKS_VIEW: 8, TOPICS_VIEW: 10, CARDS_VIEW: 9}
TOPIC_GROWTH = 8  # a topic's dot grows by this many px at a share of 1 of its cards below the threshold (T)
SELECTED_SIZE = 14
SELECTED_RING = "rgba(255, 255, 255, 0.55)"
EMPTIED_SYMBOL = "circle-open"  # a topic the model left without cards
CROSS_SYMBOL = "cross"  # a card below the threshold, between two topics (T)
MOVED_IN_SYMBOL, MOVED_IN_COLOR = "diamond", "#c084fc"  # a card T1 moved into the topic from another scripted topic
MOVED_OUT_SYMBOL, MOVED_OUT_COLOR = "diamond-open", "#f472b6"  # a card scripted here that T1 moved elsewhere
MARGIN_PALE, MARGIN_FULL = (204, 251, 241), (20, 184, 166)  # the teal scale of a card's margin: #ccfbf1 to #14b8a6
# S: a dot's share of neighbours in its own part on the figures' blue ramp (the confusion matrix's), from its middle
# step at 0 to its light step at 1, so every dot stays clear of the dark surface; the neighbour pairs' lines on the
# same ramp, the selected card's bright, the others faint
SHARE_LOW, SHARE_HIGH = SEQUENTIAL[1][1], SEQUENTIAL[2][1]
NO_NEIGHBOUR_SYMBOL = "circle-open"  # a dot without a neighbour above the generation's minimum score
NEIGHBOUR_LINE = {"color": SHARE_LOW, "width": 3, "opacity": 0.5}
SELECTED_LINE = {"color": SHARE_HIGH, "width": 5, "opacity": 1.0}
CENTROID_GLOW = (16, 10, 4)  # the sizes of the centroid's stacked markers, outside in
CENTROID_ALPHA = 0.4  # the centroid's opacity at its core, where all its markers overlap
AXIS_RANGE = [-1.15, 1.15]
AXIS_TICKS = [-1, -0.5, 0, 0.5, 1]  # the unit sphere's coordinates
AXIS_FONT = {"color": MUTED, "family": "DM Mono, monospace", "size": 10}
DEFAULT_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}
# The classes of the cards view with D, in legend order: (name, colour, symbol)
LEGEND = (
    ("easy", LEVEL_COLORS["easy"], "circle"), ("medium", LEVEL_COLORS["medium"], "circle"), ("hard", LEVEL_COLORS["hard"], "circle"),
    (PREDICTED, PREDICTED_COLOR, PREDICTED_SYMBOL), (FLASHCARD, GREY, "circle"), (UNRATED, GREY, "circle"),
)
LEGEND_STYLE = {
    "title": {"text": "Cards by difficulty", "font": {"color": MUTED, "family": "DM Mono, monospace", "size": 11}},
    "x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(7, 24, 39, 0.65)", "bordercolor": "rgba(247, 242, 232, 0.16)",
    "borderwidth": 1, "font": {"color": INK, "family": "DM Mono, monospace", "size": 11}, "itemclick": False, "itemdoubleclick": False,
}
QUIZ_STACKS = (STACK_QUIZ, STACK_DIFFICULTY)
STACK_NAMES = {STACK_PLAIN: "lecture flashcard", STACK_QUIZ: "unrated quiz", STACK_DIFFICULTY: "rated quiz", STACK_SOURCE: "quiz-file flashcard"}
PANEL_NAMES = {"S": "Similarity", "C": "Course model", "T": "Topics", "D": "Difficulty"}
PANEL_LETTERS = tuple(PANEL_NAMES)
# The model generations of each panel, in their rows' fixed order, the oldest on top; the newest is expanded when its
# panel opens
GENERATIONS = {"S": ("S1", "S2"), "C": ("C1",), "T": ("T1",), "D": ("D1", "D2")}
GENERATION_NAMES = {
    "S1": "S1 · similarity, TF-IDF", "S2": "S2 · similarity, sentence encoder", "C1": "C1 · course model",
    "T1": "T1 · topics", "D1": "D1 · difficulty, text statistics", "D2": "D2 · difficulty, zero-shot tags",
}
NO_LINKS = {
    DECKS_VIEW: "No connections between the decks.",
    TOPICS_VIEW: "No connections between topics: the model has no topic vectors yet.",
    CARDS_VIEW: "No connections: the model connects a topic of at least two cards.",
}
OVERVIEW_HINT = {
    DECKS_VIEW: "Select a deck to see its ids; Open enters it.",
    TOPICS_VIEW: "Select a topic to see its ids and its cards' counts; Open enters it. Back goes up to the decks.",
    CARDS_VIEW: "Select a card to see its ids and its level. Back goes up to the topics.",
}
TOPICS_NOTE = "The model's connections are not drawn in this view: connections visualization is not yet decided in this view."
WINDOW_NOTE = "The confusion matrix and the shares of the levels are in the window under the dendrogram's window."
ROW_HINT = "Click a row for what the measure is."
LEVEL_HINT = "Click to expand this generation; the map follows it."
NO_DESCRIPTION = "No description of this column yet."


# --- the map ---------------------------------------------------------------------------------------------------


def _axis(title: str, visible: bool) -> dict:
    """An axis of the scene: the fixed range, its ticks and a faint grid, no spike lines on hover; hidden without
    the model's positions (the range stays, so the sphere keeps its size)."""
    axis = {"visible": visible, "range": AXIS_RANGE, "showspikes": False}
    if visible:
        axis.update({
            "tickvals": AXIS_TICKS, "tickfont": AXIS_FONT, "title": {"text": title, "font": AXIS_FONT}, "gridcolor": GRID,
            "zeroline": False, "showbackground": False,
        })
    return axis


def _anchor() -> go.Scatter3d:
    """Invisible corners that pin the scene to the fixed range, however few dots are shown."""
    return go.Scatter3d(
        x=AXIS_RANGE, y=AXIS_RANGE, z=AXIS_RANGE, mode="markers", marker={"size": 0.001, "opacity": 0},
        hoverinfo="skip", showlegend=False, name="AnchorBounds",
    )


def links_trace(points: Mapping, pairs, along_surface: bool, name: str = "Connections", color: str = GOLD,
                dash: str = "dash", width: float = LINK_WIDTH, opacity: float = 1.0) -> Optional[go.Scatter3d]:
    """Pairs of drawn dots as one trace (None between two lines), straight or along the sphere's surface: by default
    the model's connections, gold and dashed; S's neighbour pairs solid on the blue ramp. None without a pair of two
    drawn dots."""
    xs: list = []
    ys: list = []
    zs: list = []
    for source, target in pairs:
        if source not in points or target not in points:
            continue
        if along_surface:
            line = surface_arc(points[source], points[target])
        else:
            line = tuple([points[source][axis], points[target][axis]] for axis in range(3))
        xs += list(line[0]) + [None]
        ys += list(line[1]) + [None]
        zs += list(line[2]) + [None]
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line={"color": color, "width": width, "dash": dash}, opacity=opacity,
        hoverinfo="skip", showlegend=False, name=name,
    )


def dots_trace(points: Mapping, labels: Mapping, colors: Mapping, view: str, symbols: Optional[Mapping] = None,
               sizes: Optional[Mapping] = None) -> go.Scatter3d:
    """The view's dots, clickable: their ids are the customdata."""
    ids = list(points)
    marker: dict = {"size": [sizes.get(dot, DOT_SIZE[view]) for dot in ids] if sizes else DOT_SIZE[view],
                    "color": [colors[dot] for dot in ids], "line": {"width": 0}}
    if symbols:
        marker["symbol"] = [symbols.get(dot, "circle") for dot in ids]
    return go.Scatter3d(
        x=[points[dot][0] for dot in ids], y=[points[dot][1] for dot in ids], z=[points[dot][2] for dot in ids],
        mode="markers", customdata=ids, text=[labels[dot] for dot in ids], hovertemplate="%{text}<extra></extra>",
        marker=marker, showlegend=False, name={DECKS_VIEW: "Decks", TOPICS_VIEW: "Topics"}.get(view, "Cards"),
    )


def legend_traces(points: Mapping, classes: Mapping[str, str]) -> list[go.Scatter3d]:
    """The legend inside the map with D: one entry per class the view holds, in LEGEND's order, each a marker in the
    class's colour and symbol on one of its cards (under the dots trace, neither hoverable nor clickable), so plotly
    draws the legend from real markers."""
    traces = []
    for name, color, symbol in LEGEND:
        card_id = next((dot for dot in points if classes.get(dot) == name), None)
        if card_id is None:
            continue
        x, y, z = points[card_id]
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="markers", marker={"size": DOT_SIZE[CARDS_VIEW], "color": color, "symbol": symbol, "line": {"width": 0}},
            hoverinfo="skip", showlegend=True, name=name,
        ))
    return traces


def selected_trace(point, color: str) -> list[go.Scatter3d]:
    """The selected dot: a white ring behind it, the dot larger."""
    x, y, z = ([value] for value in point)
    return [
        go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={"size": SELECTED_SIZE * 1.4, "color": SELECTED_RING},
                     hoverinfo="skip", showlegend=False, name="Selected ring"),
        go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={"size": SELECTED_SIZE, "color": color},
                     hoverinfo="skip", showlegend=False, name="Selected"),
    ]


def centroid_traces() -> list[go.Scatter3d]:
    """The deck's place among its topics, at the origin, faint, neither selectable nor hoverable: stacked markers of
    one opacity each (a 3D marker ignores its colour's alpha), so that the core, where all overlap, has
    CENTROID_ALPHA."""
    layer = 1 - (1 - CENTROID_ALPHA) ** (1 / len(CENTROID_GLOW))
    return [
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker={"size": size, "color": GREY, "opacity": layer, "line": {"width": 0}},
                     hoverinfo="skip", showlegend=False, name="Centroid")
        for size in CENTROID_GLOW
    ]


def margin_color(margin: float, scale: float) -> str:
    """A card's margin on the teal scale: pale at 0, full at `scale` (the course's 95th percentile) and above."""
    share = 0.0 if scale <= 0 else max(0.0, min(1.0, margin / scale))
    red, green, blue = (round(pale + (full - pale) * share) for pale, full in zip(MARGIN_PALE, MARGIN_FULL))
    return f"#{red:02x}{green:02x}{blue:02x}"


def share_color(share: float) -> str:
    """A dot's share of neighbours in its own part on the blue ramp: SHARE_LOW at 0, SHARE_HIGH at 1."""
    share = max(0.0, min(1.0, share))
    low, high = (tuple(int(color[index:index + 2], 16) for index in (1, 3, 5)) for color in (SHARE_LOW, SHARE_HIGH))
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(low, high))
    return f"#{red:02x}{green:02x}{blue:02x}"


def map_figure(page, state: dict) -> go.Figure:
    """The map of the view in `state` (app.HOME's keys) under the expanded generation. The hover labels: a deck's id,
    a topic's label (its id and a hash of its name, app.Page.topic_label), a card's comb_id; with S the dot's share of
    neighbours in its part, with T its margin, with D a predicted quiz's level."""
    view, selected, panel = state["view"], state.get("selected"), state.get("panel")
    model = page.uses_model(state)
    figure = go.Figure()
    figure.add_trace(_anchor())
    symbols = None
    sizes = None
    links = None
    legend = False
    if view == DECKS_VIEW:
        points = page.deck_points(state)
        labels = {deck_id: deck_id for deck_id in points}
        colors = {deck_id: DECK_COLOR for deck_id in points}
        if panel == "C":
            links = links_trace(points, page.deck_links(), along_surface=False)
    elif view == TOPICS_VIEW:
        if model:
            figure.add_traces(centroid_traces())
        topics = page.topics(state["deck"], state)
        points = page.topic_points(state["deck"], state)
        labels = {page.topic_dot(topic): page.topic_label(topic) for topic in topics}
        colors = {dot: TOPIC_COLOR for dot in points}
        symbols = {page.topic_dot(topic): EMPTIED_SYMBOL for topic in topics if not topic.ids}
        if panel == "T":
            sizes = {page.topic_dot(topic): DOT_SIZE[TOPICS_VIEW] + TOPIC_GROWTH * page.between_share(topic) for topic in topics}
    else:
        topic = page.entered(state)
        figure.add_trace(sphere_surface(GREY))
        points = page.card_points(topic, state) if topic is not None else {}
        labels = {card_id: card_id for card_id in points}
        report = page.difficulty_of(state) if panel == "D" else None
        if report is not None:
            colors, symbols, classes, labels = page.d_markers(points, report)
            figure.add_traces(legend_traces(points, classes))
            legend = bool(points)
        elif panel == "T" and topic is not None:
            colors, symbols, labels = page.t1_markers(topic, points)
        else:
            colors = {card_id: GREY for card_id in points}
        if panel == "C" and topic is not None:
            links = links_trace(points, page.card_links(topic, state), along_surface=True)
        if panel == "S":
            pairs, own = page.neighbour_pairs(points, state)
            figure.add_traces([trace for trace in (
                links_trace(points, pairs, True, "Neighbours", dash="solid", **NEIGHBOUR_LINE),
                links_trace(points, own, True, "Selected neighbours", dash="solid", **SELECTED_LINE),
            ) if trace is not None])
    if panel == "S":
        marked = page.s_markers(state, points)
        if marked is not None:
            colors, symbols, labels = marked
    if links is not None:
        figure.add_trace(links)
    if points:
        figure.add_trace(dots_trace(points, labels, colors, view, symbols, sizes))
    if selected in points:
        figure.add_traces(selected_trace(points[selected], colors[selected]))
    figure.update_layout(
        uirevision=f"standalone:{view}:{state.get('deck') or ''}:{state.get('topic') or ''}:{'model' if model else 'random'}",
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font={"color": INK, "family": "Georgia"},
        margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=legend, legend=LEGEND_STYLE,
        scene={
            "aspectmode": "cube", "bgcolor": BACKGROUND, "camera": DEFAULT_CAMERA,
            "xaxis": _axis("x", model), "yaxis": _axis("y", model), "zaxis": _axis("z", model),
        },
        meta={"view": view, "panel": panel},
    )
    return figure


def card_class(card, predicted: bool) -> str:
    """A card's class on the map with D, as the viewer colours it after Helper: its authored level; a quiz D1
    predicted "predicted" (the oldest group there, so its colour decides, not the predicted level's); a flashcard
    "flashcard"; an unrated quiz without a prediction "unrated quiz"."""
    if card.difficulty in LEVEL_COLORS:
        return card.difficulty
    if predicted:
        return PREDICTED
    return UNRATED if card.stack in QUIZ_STACKS else FLASHCARD


def class_style(name: str) -> tuple[str, str]:
    """A class's (colour, symbol) from LEGEND."""
    return next((color, symbol) for class_name, color, symbol in LEGEND if class_name == name)


# --- the readout tables (C1's measures, the dot's ids, S1's and T1's live measures) ---------------------------------


def model_readout(scope: str, sections) -> list:
    """Live measures (app.Page.measures and the like): what they cover, then one table of label and value per
    section under its heading; a row's hint is its tooltip."""
    rows = []
    for heading, measures in sections:
        rows.append(html.Tr(html.Th(heading, colSpan=2)))
        rows += [
            html.Tr([html.Td(label), html.Td(value, className="readout-value")], title=hint)
            for label, value, hint in measures
        ]
    return [html.Div(scope, className="readout-scope"), html.Table(html.Tbody(rows), className="readout-table")]


def id_table(rows) -> html.Table:
    """A selected dot's ids and counts, one row of label and value each, in the readout's table."""
    return html.Table(
        html.Tbody([html.Tr([html.Td(label), html.Td(value, className="readout-value")]) for label, value in rows]),
        className="readout-table id-table",
    )


# --- window A: the selected dot's ids, or the dendrogram ---------------------------------------------------------


def deck_panel(deck_id: str, topics: int, cards: int) -> html.Div:
    """A selected deck: its id, its topics and their cards under the membership; Open or a click on the panel
    enters it."""
    return html.Div(
        [
            html.Div("Deck", className="card-label"),
            id_table([("deck_id", deck_id), ("topics", topics), ("cards", cards)]),
            html.Div("Open, or a click on this frame, enters the deck. Back unselects it.", className="card-hint"),
        ],
        className="card-panel dot-panel-enter",
    )


def topic_counts(topic, card_by_id) -> list[tuple[str, int]]:
    """A topic's size and mix of cards under the membership."""
    cards = [card_by_id[card_id] for card_id in topic.ids if card_id in card_by_id]
    quizzes = sum(card.stack in QUIZ_STACKS for card in cards)
    rated = sum(card.difficulty is not None for card in cards)
    return [("cards", len(cards)), ("flashcards", len(cards) - quizzes), ("quizzes", quizzes), ("rated", rated)]


def topic_panel(topic, card_by_id, rest: bool = False) -> html.Div:
    """A selected topic: its ids, whether it is the deck's rest topic (the cards that fit no other) and its counts;
    Open or a click on the panel enters it."""
    rows = [("deck_id", topic.deck_id), ("topic_id", topic.topic_id), ("rest topic", "yes" if rest else "no")]
    emptied = [] if topic.ids else [html.Div("No cards: the model moved every card of this topic to another one.", className="deck-description")]
    return html.Div(
        [html.Div("Topic", className="card-label"), id_table(rows + topic_counts(topic, card_by_id))]
        + emptied
        + [html.Div("Open, or a click on this frame, enters the topic. Back unselects it.", className="card-hint")],
        className="card-panel dot-panel-enter",
    )


def card_panel(card, deck_id: str, topic_id: str, predicted: Optional[str] = None, between: bool = False,
               extra: Sequence[tuple[str, str]] = (), by: str = "D1") -> html.Div:
    """A selected card: its ids, its stack, its difficulty, authored or predicted (`by` D1, or D2 with D2
    expanded), whether T1 placed it between two topics, and `extra` rows (with S where its neighbours lie; with T its
    margin, its runner-up topic, a moved card's other topic); not its text."""
    if card.difficulty:
        level = card.difficulty
    elif predicted:
        level = f"{predicted} (predicted by {by})"
    else:
        level = "–"
    rows = [
        ("comb_id", card.id), ("deck_id", deck_id), ("topic_id", topic_id), ("stack", f"{card.stack} · {STACK_NAMES[card.stack]}"),
        ("difficulty", level), ("between two topics", "yes" if between else "no"),
    ] + list(extra)
    return html.Div([html.Div("Card", className="card-label"), id_table(rows)], className="card-panel")


def overview_panel(title: str, figure: Optional[go.Figure], view: str, subtitle: Optional[str] = None) -> html.Div:
    """Nothing selected: the dendrogram of the view's membership with the preset's line, or why there is none."""
    body = (
        dcc.Graph(figure=figure, config={"displayModeBar": False}, className="dendrogram-graph")
        if figure is not None else html.P(NO_LINKS[view], className="empty-state")
    )
    return html.Div(
        [html.Div(title, className="card-label")]
        + ([html.Div(subtitle, className="deck-description")] if subtitle else [])
        + [body, html.Div(OVERVIEW_HINT[view], className="card-hint")],
        className="card-panel overview-panel",
    )


# --- D1's and D2's metrics: the scores in the D panel, the matrix and the shares in the difficulty window -----------


def score_table(report) -> html.Table:
    """The key scores: the generation (mean ± sd) and the baselines (their means)."""
    header = html.Tr([html.Th("")] + [html.Th(name) for name in score_header(report)])
    rows = [
        html.Tr([html.Td(label, className="score-label"), html.Td(d1, className="score-model")] + [html.Td(value) for value in baselines])
        for label, d1, baselines in score_rows(report)
    ]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores")


def _graph(figure: go.Figure, class_name: str) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, className=class_name)


def metrics_panel(report, scope: str) -> html.Div:
    """What the generation (`report.model`, D1 or D2) measured for the view's part of the course (`report`, the
    course's, a deck's or a topic's), in the D panel: the title, what was measured, the score table and where the
    matrix and the shares are; or why there is nothing to show."""
    head = [html.Div(f"Difficulty model {report.model} · {scope}", className="card-label")]
    if not report.measured:
        return html.Div(head + [html.P(report.note or "No difficulty model.", className="empty-state")], className="tool-panel difficulty-panel")
    return html.Div(
        head
        + [
            html.Div(metrics_subtitle(report), className="deck-description"),
            score_table(report),
            html.Div("With the deck shares every level keeps its true count, so macro-F1 equals the balanced accuracy.",
                     className="card-hint"),
            html.Div(WINDOW_NOTE, className="card-hint"),
        ],
        className="tool-panel difficulty-panel",
    )


def metrics_window(report, scope: str) -> html.Div:
    """The difficulty window, under the dendrogram's window with D expanded: the generation's confusion matrix (the
    rated quizzes' true and predicted level) and the shares of the levels of the view's part; a part without metrics
    its predicted shares only (a deck without rated quizzes), a part without quizzes its note."""
    children = [html.Div(f"Difficulty model {report.model} · {scope}", className="card-label")]
    if report.measured:
        children += [
            html.Div("Rated quizzes: true and predicted level", className="card-label metrics-heading"),
            _graph(confusion_figure(report), "difficulty-confusion"),
            html.Div("Shares of the levels", className="card-label metrics-heading"),
            _graph(shares_figure(report), "difficulty-shares"),
        ]
    elif report.predictions:
        children += [html.Div("Shares of the levels", className="card-label metrics-heading"), _graph(shares_figure(report), "difficulty-shares")]
    else:
        children.append(html.P(report.note or "No difficulty model.", className="empty-state"))
    return html.Div(children, className="card-panel difficulty-window")


def window_class(expanded: bool) -> str:
    return f"window-d {'expanded' if expanded else 'collapsed'}"


# --- S's and T1's markers: their legends, T1's histogram -----------------------------------------------------------


def s_legend(level: str) -> html.Div:
    """The markers of the map with S1 or S2 expanded: the share scale, the two kinds of lines, the open circle."""
    items = [
        html.Div([html.Span(className="share-scale"), html.Span("share of a dot's neighbours in its own deck or topic (a card: in the entered topic), 0 to 1")], className="legend-item"),
        html.Div([html.Span("━", className="legend-glyph", style={"color": SHARE_HIGH}), html.Span("the selected card's neighbour pairs inside the topic")], className="legend-item"),
        html.Div([html.Span("━", className="legend-glyph", style={"color": SHARE_LOW}), html.Span("the other neighbour pairs inside the topic")], className="legend-item"),
        html.Div([html.Span("○", className="legend-glyph", style={"color": GREY}), html.Span("no neighbour above the minimum score")], className="legend-item"),
    ]
    return html.Div([html.Div(f"The map with {level}", className="card-label")] + items, className="color-legend t1-legend")



def t1_legend(scale: float, threshold: float) -> html.Div:
    """The markers of the map with T expanded: the margin scale, the cross, the two diamonds and the open circle."""
    items = [
        html.Div([html.Span(className="margin-scale"), html.Span(f"margin 0 to {scale:.2f} (the course's 95th percentile)")], className="legend-item"),
        html.Div([html.Span("✚", className="legend-glyph", style={"color": margin_color(0.0, scale)}), html.Span(f"below {threshold:.2f}: between two topics")], className="legend-item"),
        html.Div([html.Span("◆", className="legend-glyph", style={"color": MOVED_IN_COLOR}), html.Span("moved in by T1 from another scripted topic")], className="legend-item"),
        html.Div([html.Span("◇", className="legend-glyph", style={"color": MOVED_OUT_COLOR}), html.Span("scripted here, moved out by T1")], className="legend-item"),
        html.Div([html.Span("○", className="legend-glyph", style={"color": TOPIC_COLOR}), html.Span("an emptied topic; a topic's dot grows with its share of cards between two topics")], className="legend-item"),
    ]
    return html.Div([html.Div("The map with T", className="card-label")] + items, className="color-legend t1-legend")


def margin_histogram(margins: Sequence[float], threshold: float) -> go.Figure:
    """The view's margins in bins, the threshold as a dashed line."""
    figure = go.Figure(go.Histogram(x=list(margins), nbinsx=20, marker={"color": margin_color(1.0, 1.0)}, hoverinfo="skip"))
    figure.add_vline(x=threshold, line={"color": CUT_COLOR, "dash": "dash", "width": 1})
    figure.update_layout(
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font=FONT, height=150, margin={"l": 36, "r": 10, "t": 10, "b": 30},
        xaxis={"title": {"text": "margin"}, "gridcolor": GRID, "zeroline": False},
        yaxis={"title": {"text": "cards"}, "gridcolor": GRID, "zeroline": False}, bargap=0.05,
    )
    return figure


# --- the model panels ---------------------------------------------------------------------------------------------


def measure_key(prefix: str, columns: Sequence[str]) -> str:
    """A row's key in the `view` store: the generation and the table's place, then the row's first column."""
    return f"{prefix}:{columns[0]}"


def description_row(entry: Optional[tuple[str, str, str]], width: int) -> html.Tr:
    """The open description under a row: what the measure is and its range (app/docs/columns.md), with the close
    box; not the column's name again, which the row above shows; one line when the glossary has no entry."""
    if entry is None:
        text = [html.Div(NO_DESCRIPTION)]
    else:
        _names, meaning, span = entry
        text = [html.Div(meaning), html.Div(f"Range: {span}", className="measure-range")]
    return html.Tr(
        html.Td([html.Button("×", className="measure-close", title="Close"), html.Div(text, className="measure-text")], colSpan=width),
        className="measure-description",
    )


def results_table(table: Table, prefix: str, expanded: Optional[str], glossary: Mapping[str, tuple[str, str, str]]) -> html.Div:
    """A level's results of the newest run: the heads (the chosen variant marked), one row per measure, clickable
    (its key in `data-measure`; assets/panel_rows.js sends it), the open row's description under it; the
    bookkeeping as the subtitle."""
    header = html.Tr([html.Th("")] + [html.Th(head, className="chosen" if chosen else None) for head, chosen in zip(table.heads, table.chosen)])
    rows = []
    for (label, values), columns in zip(table.rows, table.row_columns):
        key = measure_key(prefix, columns)
        rows.append(html.Tr(
            [html.Td(label, className="score-label")] + [html.Td(value, className="chosen" if chosen else None) for value, chosen in zip(values, table.chosen)],
            className="measure-row open" if key == expanded else "measure-row", title=ROW_HINT, **{"data-measure": key},
        ))
        if key == expanded:
            rows.append(description_row(glossary.get(columns[0]), len(table.heads) + 1))
    return html.Div(
        [
            html.Div(table.title, className="card-label metrics-heading"),
            html.Div(table.subtitle, className="deck-description"),
            html.Div(html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores results-table"), className="results-scroll"),
        ],
        className="results-block",
    )


def level_section(description: str, live: Sequence, tables: Optional[Sequence[Table]], hint: str, level: str,
                  expanded: Optional[str], glossary: Mapping[str, tuple[str, str, str]]) -> html.Div:
    """An expanded generation's section (its row names it): what it is, its live measures, its results tables with
    the one open row's description, or why there is none (`hint`: how to make the file, or that it holds no rows of
    the generation)."""
    body = [html.Div(description, className="deck-description")] + list(live)
    if not tables:
        body.append(html.P(hint, className="empty-state"))
    else:
        body += [results_table(table, f"{level}:{index}", expanded, glossary) for index, table in enumerate(tables)]
    return html.Div(body, className="level-section")


def generation_rows(letter: str, expanded: str, section) -> list:
    """An expanded panel's content: a row per generation in GENERATIONS' fixed order, the oldest on top, each
    clickable (its generation in `data-level`; assets/panel_rows.js sends it), the expanded one open with its section
    under it wherever it stands: expanding another moves no row."""
    rows = []
    for level in GENERATIONS[letter]:
        is_open = level == expanded
        rows.append(html.Div(
            [html.Span("▾" if is_open else "▸", className="level-mark"), html.Span(GENERATION_NAMES[level])],
            className="level-row open" if is_open else "level-row", title=None if is_open else LEVEL_HINT, **{"data-level": level},
        ))
        if is_open:
            rows.append(section)
    return rows


def model_panel_boxes() -> list[html.Div]:
    """The four panels, collapsed: the row with the letter and the name (a click expands), the close box, and the
    content the callback fills (`content-S`); the callback sets each box's class."""
    boxes = []
    for letter, name in PANEL_NAMES.items():
        head = html.Div(
            [
                html.Button(f"{letter} – {name}", id=f"panel-{letter}", n_clicks=0, className="panel-row"),
                html.Button("×", id=f"close-{letter}", n_clicks=0, className="panel-close", title="Collapse"),
            ],
            className="panel-head",
        )
        boxes.append(html.Div([head, html.Div(id=f"content-{letter}", className="panel-content")], id=f"box-{letter}", className=box_class(False)))
    return boxes


def box_class(expanded: bool) -> str:
    return f"model-panel {'expanded' if expanded else 'collapsed'}"


# --- the page ---------------------------------------------------------------------------------------------------


def page_layout(home: dict) -> html.Main:
    """The page; the callback fills the map, window A, the difficulty window and the panels at load. The hidden
    input `row-click` carries a click on a generation's or a measure's row (assets/panel_rows.js) to the callback."""
    return html.Main(
        [
            html.Header(
                [
                    html.Div([html.P("GUIDED LEARNING · THE MODELS", className="eyebrow"), html.H1("What the models see.")]),
                    html.Div(
                        [
                            html.Button("Back", id="back", className="top-button", n_clicks=0, disabled=True),
                            html.Button("Open", id="open", className="top-button", n_clicks=0, disabled=True),
                        ],
                        className="topbar-actions",
                    ),
                ],
                className="topbar",
            ),
            html.Section([html.Div(id="where", className="card-count")], className="toolbar"),
            html.Section(
                [
                    html.Div(
                        dcc.Graph(id="map", config={"displayModeBar": False, "scrollZoom": True}, responsive=True),
                        className="map-frame",
                    ),
                    html.Div(id="dot-panel", n_clicks=0, className="window-a"),
                    html.Div(id="difficulty-window", className=window_class(False)),
                    html.Div(model_panel_boxes(), id="model-panels", className="model-panels"),
                ],
                className="workspace",
            ),
            dcc.Store(id="view", data=dict(home)),
            dcc.Input(id="row-click", value="", type="text", className="row-click"),
        ],
        className="app-shell",
    )
