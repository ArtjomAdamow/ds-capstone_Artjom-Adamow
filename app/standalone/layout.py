"""The standalone page's layout, map and panels (Dash and plotly; nothing of the viewer), written anew after the
viewer's in its theme (assets/style.css holds the tokens):

* the top bar: the title, Back and Open;
* the map (`map`): the decks on the sphere, a deck's topics on a ring around its centroid, a topic's cards on the
  surface of its sphere; the model's connections below the preset gold and dashed (straight between decks, along
  the surface between cards); the cards coloured by their authored difficulty, predicted and unrated ones grey;
* window A (`dot-panel`): the selected dot's panel, a click on a deck's or a topic's enters it; with nothing
  selected the dendrogram of the view's connections with the preset's cut-off line (figures.dendrogram_figure);
* window B (`metrics-panel`): the difficulty model D1's metrics of the view (the course, the deck, the topic).

What the views show comes from app.Page; the view itself lives in the `view` store (app.HOME's keys).
"""

from typing import Optional

import plotly.graph_objects as go
from dash import dcc, html

from ..cards import STACK_DIFFICULTY, STACK_QUIZ
from ..figures import (
    GOLD, INK, MUTED, confusion_figure, metrics_subtitle, score_header, score_rows, shares_figure, sphere_surface,
    surface_arc,
)

CARDS_VIEW = "cards"
TOPICS_VIEW = "topics"
DECKS_VIEW = "decks"
BACKGROUND = "#102a43"
GREY = "#7c8b99"  # the viewer's colour of a dot in no group: a predicted or unrated card, the cards' sphere
LEVEL_COLORS = {"easy": "#4f7fcf", "medium": "#b8956a", "hard": "#ff5349"}  # blue, beige-brown, red-orange
DECK_COLOR = MUTED  # the decks in one colour
TOPIC_COLOR = MUTED
LINK_WIDTH = 4
DOT_SIZE = {DECKS_VIEW: 8, TOPICS_VIEW: 10, CARDS_VIEW: 9}
SELECTED_SIZE = 14
SELECTED_RING = "rgba(255, 255, 255, 0.55)"
EMPTIED_SYMBOL = "circle-open"  # a topic the model left without cards
CENTROID_GLOW = ((64, 0.08), (40, 0.14), (16, 0.3))  # (size, opacity) of the centroid's stacked markers
AXIS_RANGE = [-1.15, 1.15]
DEFAULT_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}
LEGEND = (("easy", LEVEL_COLORS["easy"]), ("medium", LEVEL_COLORS["medium"]), ("hard", LEVEL_COLORS["hard"]), ("predicted or unrated", GREY))
QUIZ_STACKS = (STACK_QUIZ, STACK_DIFFICULTY)
NO_LINKS = {
    DECKS_VIEW: "No connections between the decks.",
    TOPICS_VIEW: "No connections between topics: the model has no topic vectors yet.",
    CARDS_VIEW: "No connections: the model connects a topic of at least two cards.",
}
OVERVIEW_HINT = {
    DECKS_VIEW: "Select a deck to see its description; Open enters it.",
    TOPICS_VIEW: "Select a topic to see its cards' counts; Open enters it. Back goes up to the decks.",
    CARDS_VIEW: "Select a card to read it. Back goes up to the topics.",
}


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


# --- the map ---------------------------------------------------------------------------------------------------


def _anchor() -> go.Scatter3d:
    """Invisible corners that pin the scene to the fixed range, however few dots are shown."""
    return go.Scatter3d(
        x=AXIS_RANGE, y=AXIS_RANGE, z=AXIS_RANGE, mode="markers", marker={"size": 0.001, "opacity": 0},
        hoverinfo="skip", showlegend=False, name="AnchorBounds",
    )


def links_trace(points: dict, pairs, along_surface: bool) -> Optional[go.Scatter3d]:
    """The model's connections as one gold dashed trace (None between two lines): straight, or along the sphere's
    surface. None without a connection between two drawn dots."""
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
        x=xs, y=ys, z=zs, mode="lines", line={"color": GOLD, "width": LINK_WIDTH, "dash": "dash"},
        hoverinfo="skip", showlegend=False, name="Connections",
    )


def dots_trace(points: dict, labels: dict, colors: dict, view: str, symbols: Optional[dict] = None) -> go.Scatter3d:
    """The view's dots, clickable: their ids are the customdata."""
    ids = list(points)
    marker = {"size": DOT_SIZE[view], "color": [colors[dot] for dot in ids], "line": {"width": 0}}
    if symbols:
        marker["symbol"] = [symbols.get(dot, "circle") for dot in ids]
    return go.Scatter3d(
        x=[points[dot][0] for dot in ids], y=[points[dot][1] for dot in ids], z=[points[dot][2] for dot in ids],
        mode="markers", customdata=ids, text=[labels[dot] for dot in ids], hovertemplate="%{text}<extra></extra>",
        marker=marker, showlegend=False, name={DECKS_VIEW: "Decks", TOPICS_VIEW: "Topics"}.get(view, "Cards"),
    )


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
    """The deck's place among its topics, at the origin, neither selectable nor hoverable."""
    return [
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker={"size": size, "color": _rgba(GREY, alpha), "line": {"width": 0}},
                     hoverinfo="skip", showlegend=False, name="Centroid")
        for size, alpha in CENTROID_GLOW
    ]


def map_figure(page, state: dict) -> go.Figure:
    """The map of the view in `state` (app.HOME's keys)."""
    view, selected = state["view"], state.get("selected")
    figure = go.Figure()
    figure.add_trace(_anchor())
    symbols = None
    if view == DECKS_VIEW:
        points = page.deck_points()
        labels = {deck_id: page.deck_by_id[deck_id].name for deck_id in points}
        colors = {deck_id: DECK_COLOR for deck_id in points}
        links = links_trace(points, page.deck_links(), along_surface=False)
    elif view == TOPICS_VIEW:
        figure.add_traces(centroid_traces())
        topics = page.topics(state["deck"])
        points = page.topic_points(state["deck"])
        labels = {page.topic_dot(topic): topic.name for topic in topics}
        colors = {dot: TOPIC_COLOR for dot in points}
        symbols = {page.topic_dot(topic): EMPTIED_SYMBOL for topic in topics if not topic.ids}
        links = None  # the model has no topic connections (step 19)
    else:
        topic = page.topic(state["deck"], state["topic"])
        figure.add_trace(sphere_surface(GREY))
        points = page.card_points(topic) if topic is not None else {}
        labels = {card_id: page.card_by_id[card_id].question for card_id in points}
        colors = {card_id: card_color(page.card_by_id[card_id]) for card_id in points}
        links = links_trace(points, page.card_links(topic), along_surface=True) if topic is not None else None
    if links is not None:
        figure.add_trace(links)
    if points:
        figure.add_trace(dots_trace(points, labels, colors, view, symbols))
    if selected in points:
        figure.add_traces(selected_trace(points[selected], colors[selected]))
    figure.update_layout(
        uirevision=f"standalone:{view}:{state.get('deck') or ''}:{state.get('topic') or ''}",
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font={"color": INK, "family": "Georgia"},
        margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=False,
        scene={
            "aspectmode": "cube", "bgcolor": BACKGROUND, "camera": DEFAULT_CAMERA,
            "xaxis": {"visible": False, "range": AXIS_RANGE}, "yaxis": {"visible": False, "range": AXIS_RANGE},
            "zaxis": {"visible": False, "range": AXIS_RANGE},
        },
        meta={"view": view},
    )
    return figure


def card_color(card) -> str:
    """A card's colour as the viewer shows it after Helper: its authored difficulty's, else grey (predicted or
    unrated)."""
    return LEVEL_COLORS.get(card.difficulty, GREY)


def color_legend() -> html.Div:
    return html.Div(
        [html.Div("Cards by difficulty", className="card-label")]
        + [
            html.Div([html.Span(className="legend-swatch", style={"backgroundColor": color}), html.Span(name)], className="legend-item")
            for name, color in LEGEND
        ],
        className="color-legend",
    )


# --- window A: the selected dot, or the dendrogram -------------------------------------------------------------


def deck_panel(deck, topics: int) -> html.Div:
    """A selected deck: its name and description; Open or a click on the panel enters it."""
    return html.Div(
        [
            html.Div("Deck", className="card-label"),
            html.Div(deck.name, className="card-content"),
            dcc.Markdown(deck.description, className="deck-description"),
            html.Div(f"{topics} topics of the model.", className="deck-description"),
            html.Div("Open, or a click on this frame, enters the deck. Back unselects it.", className="card-hint"),
        ],
        className="card-panel dot-panel-enter",
    )


def topic_counts(topic, card_by_id) -> str:
    """A topic's size and mix of cards under the model's membership."""
    cards = [card_by_id[card_id] for card_id in topic.ids if card_id in card_by_id]
    if not cards:
        return "No cards: the model moved every card of this topic to another one."
    quizzes = sum(card.stack in QUIZ_STACKS for card in cards)
    rated = sum(card.difficulty is not None for card in cards)
    return f"{len(cards)} cards: {len(cards) - quizzes} flashcards, {quizzes} quiz cards, {rated} of them with a difficulty."


def topic_panel(topic, card_by_id, rest: bool = False) -> html.Div:
    """A selected topic: its name and its counts; Open or a click on the panel enters it."""
    return html.Div(
        [html.Div("Topic", className="card-label"), html.Div(topic.name, className="card-content")]
        + ([html.Div("The deck's cards that fit no other topic.", className="deck-description")] if rest else [])
        + [
            html.Div(topic_counts(topic, card_by_id), className="deck-description"),
            html.Div("Open, or a click on this frame, enters the topic. Back unselects it.", className="card-hint"),
        ],
        className="card-panel dot-panel-enter",
    )


def card_panel(card, predicted: Optional[str] = None) -> html.Div:
    """A selected card: its question with its answer, and its difficulty, authored or predicted by D1."""
    if card.difficulty:
        level = [html.Div(f"Difficulty: {card.difficulty}", className="card-level")]
    elif predicted:
        level = [html.Div(f"Predicted by D1: {predicted}", className="card-level")]
    else:
        level = []
    return html.Div(
        [
            html.Div("Question", className="card-label"),
            dcc.Markdown(card.question, className="card-content"),
            html.Div("Answer", className="card-label"),
            dcc.Markdown(card.answer, className="card-answer"),
        ]
        + level,
        className="card-panel",
    )


def overview_panel(title: str, figure: Optional[go.Figure], view: str, subtitle: Optional[str] = None) -> html.Div:
    """Nothing selected: the dendrogram of the view's connections with the preset's line, or why there is none."""
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


# --- window B: D1's metrics ------------------------------------------------------------------------------------


def score_table(report) -> html.Table:
    """The key scores: D1 (mean ± sd) and the baselines (their means)."""
    header = html.Tr([html.Th("")] + [html.Th(name) for name in score_header(report)])
    rows = [
        html.Tr([html.Td(label, className="score-label"), html.Td(d1, className="score-model")] + [html.Td(value) for value in baselines])
        for label, d1, baselines in score_rows(report)
    ]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores")


def _graph(figure: go.Figure, class_name: str) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, className=class_name)


def metrics_panel(report, scope: str) -> html.Div:
    """What D1 measured and predicted for the view's part of the course (`report`, the course's, a deck's or a
    topic's), as the viewer's back of the card panel shows it; or why there is nothing to show."""
    title = html.Div(f"Difficulty model D1 · {scope}", className="card-label")
    if not report.measured:
        children = [title, html.P(report.note or "No difficulty model.", className="empty-state")]
        if report.predictions:  # a deck without rated quizzes still has its predicted ones
            children.append(_graph(shares_figure(report), "difficulty-shares"))
        return html.Div(children, className="tool-panel difficulty-panel")
    return html.Div(
        [
            title,
            html.Div(metrics_subtitle(report), className="deck-description"),
            score_table(report),
            html.Div("With the deck shares every level keeps its true count, so macro-F1 equals the balanced accuracy.",
                     className="card-hint"),
            html.Div("Rated quizzes: true and predicted level", className="card-label metrics-heading"),
            _graph(confusion_figure(report), "difficulty-confusion"),
            html.Div("Shares of the levels", className="card-label metrics-heading"),
            _graph(shares_figure(report), "difficulty-shares"),
        ],
        className="tool-panel difficulty-panel",
    )


# --- the page ---------------------------------------------------------------------------------------------------


def page_layout(home: dict) -> html.Main:
    """The page; the callback fills the map and the panels at load."""
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
                        [
                            html.Div(dcc.Graph(id="map", config={"displayModeBar": False, "scrollZoom": True}), className="map-frame"),
                            color_legend(),
                        ],
                        className="map-column",
                    ),
                    html.Div(
                        [
                            html.Div(id="dot-panel", n_clicks=0, className="window-a"),
                            html.Div(id="metrics-panel", className="window-b"),
                        ],
                        className="side-column",
                    ),
                ],
                className="workspace",
            ),
            dcc.Store(id="view", data=dict(home)),
        ],
        className="app-shell",
    )
