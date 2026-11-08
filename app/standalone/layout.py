"""The standalone page's layout, map and regions (Dash and plotly; nothing of the viewer), written anew after the
viewer's in its theme (assets/style.css holds the tokens and the type scale). The slides look: two columns under
the top bar, large type, the most relevant on top, everything else under Details.

* the top bar: the title, Back and Open, and the Sputnik link (`viewer-link`) where the launcher serves the viewer
  next to the page (a URL; nothing of the viewer is imported), none without it; under it the breadcrumb (`where`:
  "Deck 17 › Topic 02", without a topic's hash);
* the map (`map`), the left column, a square at the full height under the top bar (assets/map_side.js): the view's
  dots on the sphere, drawn by layers: the check boxes inside the map (`layers`, one per model letter) say which
  models draw, each on a channel of its own, whatever set is checked (app.Page decides their states, map_figure
  draws): the positions and the connections from C (gold and dashed, straight between decks, along the surface
  between cards; the centroid of a deck; C on the scripted topics where T is off), the membership with its fill by
  margin and its symbols from T (a cross below the threshold, the diamonds of the moved cards, a topic's dot grown
  with its share of cards between two topics, an emptied topic an open circle), the ring colours from D (a rated
  quiz's by its level, a predicted quiz's amber, a ring behind a quiz the two generations put at different levels),
  the selected card's neighbour pairs as lines along the surface from S (the selected card's bright), P1's answer
  as the highlighted target from P; nothing checked, the bare course: the scripted topics' grey dots at random. The
  legend stands in a strip of its own under the sphere, one group per layer in force with its plain title
  (LEGEND_TITLES), the symbols at a constant size; T's margin as a colour bar along the full height at the right
  edge; no axes; the hover names a dot by id (a topic by its label with its hash);
* the right column, from top to bottom: the tab bar (`tab-S` to `tab-P`, the model letters with their names; a
  click opens one, a second click closes it), the open model's generations as small tabs (`levels`, their clicks
  through assets/panel_rows.js as the rows' were; the "app" mark on the one the app runs), the section
  (`section-head`: the headline, the two numbers with their contrast, sentence, verdict and explanation, the chosen
  variant in plain words; `section-live`: what stays on top for the live demo, T's line of the view next to the
  course and P's answer; P's request box `prompt-box` and the Details button `details-toggle` stand in the layout
  from the start, hidden where they do not apply), the selected dot's tag strip (`tag-strip`: a large title and a
  few tags, its Open `tag-open` for a deck or a topic; hidden with nothing selected) and the figure slot
  (`figure-slot`, the rest of the height): one figure by the open tab, filling the slot, or Details opened in its
  place, scrolling inside it, the page's one scroll bar (today's generation section with its live measures and its
  results tables, whose rows open their description on a click, and what moved in).

No learning material shows: no card's question, answer or explanation, no deck's description; a card is its
comb_id. The map's hover labels and the breadcrumb are ids too: a deck's id, a topic's id (the hover with the hash,
app.Page.topic_label), a card's comb_id.

What the views show comes from app.Page; the view itself lives in the `view` store (app.HOME's keys).
"""

from typing import Mapping, Optional, Sequence

import plotly.graph_objects as go
from dash import dcc, html

from ..cards import STACK_DIFFICULTY, STACK_PLAIN, STACK_QUIZ, STACK_SOURCE
from ..figures import (
    CUT_COLOR, FONT, GOLD, GRID, INK, LABEL_FONT, MUTED, PREDICTED_COLOR, SEQUENTIAL, TRANSPARENT, confusion_figure, metrics_subtitle,
    plain_layout, score_header, score_rows, shares_figure, sphere_surface, surface_arc,
)
from .headlines import NO_TAB, WHAT, Headline, plain_words
from .results import Table, glossary_entry

CARDS_VIEW = "cards"
TOPICS_VIEW = "topics"
DECKS_VIEW = "decks"
BACKGROUND = "#102a43"
GREY = "#7c8b99"  # the viewer's colour of a dot in no group: a flashcard, the cards' sphere
LEVEL_COLORS = {"easy": "#4f7fcf", "medium": "#b8956a", "hard": "#ff5349"}  # blue, beige-brown, red-orange
# The quizzes D1 predicted: the viewer's "Quiz difficulty: predicted" group (glue.difficulty_groups, the oldest group,
# so its colour decides there too) and the predicted series of the shares figure share this amber
DISAGREE = "D1 and D2 disagree"  # the ring behind a predicted quiz the two generations put at different levels
DISAGREE_SYMBOL = "circle-open"
DISAGREE_SIZE = 24
PREDICTED = "predicted"
NOT_RATED = "not rated"  # a flashcard, or an unrated quiz without a prediction: grey, one entry
DECK_COLOR = MUTED  # the decks in one colour
TOPIC_COLOR = MUTED
LINK_WIDTH = 4
DOT_SIZE = 13  # every view's dots
TOPIC_GROWTH = 8  # a topic's dot grows by this many px at a share of 1 of its cards below the threshold (T)
SELECTED_SIZE = 20
SELECTED_RING = "rgba(255, 255, 255, 0.55)"
RING_WIDTH = 3  # D's ring around a card's dot
TARGET_SIZE = 28  # P1's target: a gold ring behind the dot
TARGET = "P1's target"
EMPTIED_SYMBOL = "circle-open"  # a topic the model left without cards
CROSS_SYMBOL = "cross"  # a card below the threshold, between two topics (T)
MOVED_IN_SYMBOL, MOVED_IN_COLOR = "diamond", "#c084fc"  # a card T1 moved into the topic from another scripted topic
MOVED_OUT_SYMBOL, MOVED_OUT_COLOR = "diamond-open", "#f472b6"  # a card scripted here that T1 moved elsewhere
MARGIN_PALE, MARGIN_FULL = (204, 251, 241), (20, 184, 166)  # the teal scale of a card's margin: #ccfbf1 to #14b8a6
# S: the neighbour pairs' lines on the figures' blue ramp, the selected card's bright, the others faint
SHARE_LOW, SHARE_HIGH = SEQUENTIAL[1][1], SEQUENTIAL[2][1]
NEIGHBOUR_LINE = {"color": SHARE_LOW, "width": 3, "opacity": 0.5}
SELECTED_LINE = {"color": SHARE_HIGH, "width": 5, "opacity": 1.0}
NEIGHBOUR_PAIRS, SELECTED_PAIRS = "neighbour pairs", "the selected card's pairs"  # the lines' traces, by their legend names
CENTROID_GLOW = (16, 10, 4)  # the sizes of the centroid's stacked markers, outside in
CENTROID_ALPHA = 0.4  # the centroid's opacity at its core, where all its markers overlap
AXIS_RANGE = [-1.15, 1.15]
DEFAULT_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}
MAP_FONT = {"color": INK, "family": "DM Mono, monospace", "size": 18}  # the legend and the colour bar
HOVER_FONT = {"color": INK, "family": "DM Mono, monospace", "size": 16}
LEGEND_STRIP = 0.16  # the share of the map's height under the sphere for the legend's strip
BAR_WIDTH = 0.1  # the share of the map's width at the right edge for the colour bar
LABEL_BACKING = "rgba(7, 24, 39, 0.95)"  # a label on the plot over a line: the line hidden behind it
# The classes of the cards view with D, in legend order: (name, colour, symbol); a ring in the class's colour around
# a circle, as the dots are drawn (a predicted quiz too: its amber ring, no symbol of its own)
LEGEND = (
    ("easy", LEVEL_COLORS["easy"], "circle"), ("medium", LEVEL_COLORS["medium"], "circle"), ("hard", LEVEL_COLORS["hard"], "circle"),
    (PREDICTED, PREDICTED_COLOR, "circle"), (NOT_RATED, GREY, "circle"),
)
BETWEEN, MOVED_IN, MOVED_OUT = "between two topics", "moved in", "moved out"
EMPTIED, GROWN = "emptied topic", "larger = more<br>uncertain cards"  # two lines in the strip
T_TOPIC_LEGEND = ((EMPTIED, TOPIC_COLOR, EMPTIED_SYMBOL), (GROWN, TOPIC_COLOR, "circle"))
LEGEND_TITLES = {"S": "Similar cards", "T": "Topic fit", "D": "Difficulty", "P": "Prompt"}  # plain, per layer
QUIZ_STACKS = (STACK_QUIZ, STACK_DIFFICULTY)
STACK_NAMES = {STACK_PLAIN: "Q&A flashcard", STACK_QUIZ: "unrated quiz", STACK_DIFFICULTY: "rated quiz", STACK_SOURCE: "quiz-file flashcard"}
PANEL_NAMES = {"S": "Similarity", "C": "Course model", "T": "Topics", "D": "Difficulty", "P": "Prompt"}
PANEL_LETTERS = tuple(PANEL_NAMES)
# The model generations of each tab, in their tabs' fixed order, the oldest first; the newest is open when its tab opens
GENERATIONS = {"S": ("S1", "S2"), "C": ("C1",), "T": ("T1",), "D": ("D1", "D2"), "P": ("P1",)}
GENERATION_NAMES = {
    "S1": "S1 · similarity, TF-IDF", "S2": "S2 · similarity, sentence encoder", "C1": "C1 · course model",
    "T1": "T1 · topics", "D1": "D1 · difficulty, text statistics", "D2": "D2 · difficulty, zero-shot tags",
    "P1": "P1 · navigation prompt",
}
APP_MARK = "app"  # on the generation tab the app runs
PROMPT_PLACEHOLDER = "A question, a concept or a code name"
PROMPT_HINT = "Type a request and press Go or Enter: P1 answers with the three nearest topics, and the page goes where it points."
PROMPT_OFF = "P1 needs S: check S's layer to ask."
TARGET_HINT = "Click to go to this target."
NO_LINKS = {
    DECKS_VIEW: "No connections between the decks.",
    TOPICS_VIEW: "No tree of the topics: the model clusters a deck of at least two topics with cards.",
    CARDS_VIEW: "No connections: the model connects a topic of at least two cards.",
}
VIEWER_LINK = "Sputnik"  # the top bar's link to the viewer served next to the page
VIEWER_LINK_TITLE = "The viewer Sputnik, served next to this page: the map with the models on from the start, in a new tab"
OVERVIEW_HINT = {
    DECKS_VIEW: "Select a deck to see its tags; Open enters it.",
    TOPICS_VIEW: "Select a topic to see its tags and its cards' counts; Open enters it. Back goes up to the decks.",
    CARDS_VIEW: "Select a card to see its tags and its level. Back goes up to the topics.",
}
TOPICS_NOTE = "The topic connections are not drawn on the map: the topics stand at random, so a line would say nothing; the dendrogram shows the deck's tree."
MACRO_F1_NOTE = "With the deck shares every level keeps its true count, so macro-F1 equals the balanced accuracy."
CUT_LABELS = {DECKS_VIEW: "groups of ≤ {size} decks", CARDS_VIEW: "groups of ≤ {size} cards", TOPICS_VIEW: "the full tree"}
SELECT_A_CARD = "Select a card to see its neighbours; until then the dendrogram."
NO_CARDS = "No cards in this view."
ROW_HINT = "Click a row for what the measure is."
LEVEL_HINT = "Click to open this generation."
NO_DESCRIPTION = "No description of this column yet."
DETAILS = "Details"
LAYER_NOTES = {"fallback": "scripted topics", "off": "needs S"}  # what a check box says of its layer's state


# --- the map ---------------------------------------------------------------------------------------------------


def _axis() -> dict:
    """An axis of the scene: hidden, the fixed range (so the sphere keeps its size), no spike lines on hover."""
    return {"visible": False, "range": AXIS_RANGE, "showspikes": False}


def _anchor() -> go.Scatter3d:
    """Invisible corners that pin the scene to the fixed range, however few dots are shown."""
    return go.Scatter3d(
        x=AXIS_RANGE, y=AXIS_RANGE, z=AXIS_RANGE, mode="markers", marker={"size": 0.001, "opacity": 0},
        hoverinfo="skip", showlegend=False, name="AnchorBounds",
    )


def links_trace(points: Mapping, pairs, along_surface: bool, name: str = "Connections", color: str = GOLD,
                dash: str = "dash", width: float = LINK_WIDTH, opacity: float = 1.0, legend: bool = False,
                group: Optional[str] = None) -> Optional[go.Scatter3d]:
    """Pairs of drawn dots as one trace (None between two lines), straight or along the sphere's surface: by default
    the model's connections, gold and dashed; S's neighbour pairs solid on the blue ramp, an entry of the legend by
    their name in the layer's group. None without a pair of two drawn dots."""
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
        hoverinfo="skip", showlegend=legend, name=name, **_group(group),
    )


def _group(letter: Optional[str]) -> dict:
    """A legend entry's group: the layer's letter and its plain title."""
    if letter is None:
        return {}
    return {"legendgroup": letter, "legendgrouptitle": {"text": LEGEND_TITLES.get(letter, letter), "font": {**MAP_FONT, "color": MUTED}}}


def _marker(color: str, symbol: str, size: float, ring: Optional[str] = None) -> dict:
    """A dot's marker: its fill, symbol and size, and D's ring where it has one (a transparent line without)."""
    return {"size": size, "color": color, "symbol": symbol, "line": {"width": RING_WIDTH, "color": ring or TRANSPARENT}}


def dots_trace(points: Mapping, labels: Mapping, colors: Mapping, view: str, symbols: Optional[Mapping] = None,
               sizes: Optional[Mapping] = None, rings: Optional[Mapping] = None) -> go.Scatter3d:
    """The view's dots, clickable: their ids are the customdata; D's rings as the markers' lines."""
    ids = list(points)
    marker: dict = {
        "size": [sizes.get(dot, DOT_SIZE) for dot in ids] if sizes else DOT_SIZE,
        "color": [colors[dot] for dot in ids],
        "line": {"width": RING_WIDTH, "color": [rings.get(dot) or TRANSPARENT for dot in ids] if rings else TRANSPARENT},
    }
    if symbols:
        marker["symbol"] = [symbols.get(dot, "circle") for dot in ids]
    return go.Scatter3d(
        x=[points[dot][0] for dot in ids], y=[points[dot][1] for dot in ids], z=[points[dot][2] for dot in ids],
        mode="markers", customdata=ids, text=[labels[dot] for dot in ids], hovertemplate="%{text}<extra></extra>",
        marker=marker, showlegend=False, name={DECKS_VIEW: "Decks", TOPICS_VIEW: "Topics"}.get(view, "Cards"),
    )


def legend_traces(points: Mapping, classes: Mapping[str, str], entries, group: str, colors: Optional[Mapping] = None,
                  symbols: Optional[Mapping] = None, sizes: Optional[Mapping] = None, rings: Optional[Mapping] = None) -> list[go.Scatter3d]:
    """The legend's entries of one layer: one per class the view holds, in `entries`' order ((name, colour,
    symbol): D's LEGEND, T's), each a marker on one of its dots as that dot is drawn (its fill and symbol, D's ring
    in the class's colour; under the dots trace, neither hoverable nor clickable), so plotly draws the legend from
    real markers, in the layer's group with its plain title."""
    traces = []
    for name, color, symbol in entries:
        dot = next((dot for dot in points if classes.get(dot) == name), None)
        if dot is None:
            continue
        x, y, z = points[dot]
        fill = colors.get(dot, color) if colors else color
        drawn = symbols.get(dot, symbol) if symbols else symbol
        ring = rings.get(dot) if rings else None
        size = sizes.get(dot, DOT_SIZE) if sizes else DOT_SIZE
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="markers", marker=_marker(fill, drawn, size, ring), hoverinfo="skip", showlegend=True,
            name=name, **_group(group),
        ))
    return traces


def by_symbol(symbols: Optional[Mapping[str, str]], entries) -> dict[str, str]:
    """The dots' classes told by their symbols: a dot whose symbol is an entry's is of that entry's class."""
    names = {symbol: name for name, _color, symbol in entries}
    return {dot: names[symbol] for dot, symbol in (symbols or {}).items() if symbol in names}


def t1_entries(scale: float) -> tuple:
    """T's classes of the cards view: the cross in the pale end of the margin scale, the two diamonds."""
    return ((BETWEEN, margin_color(0.0, scale), CROSS_SYMBOL), (MOVED_IN, MOVED_IN_COLOR, MOVED_IN_SYMBOL),
            (MOVED_OUT, MOVED_OUT_COLOR, MOVED_OUT_SYMBOL))


def legend_style() -> dict:
    """The legend's strip under the sphere: horizontal along the map's full width, its symbols at a constant size,
    its clicks off (a click would hide the dummy marker alone)."""
    return {
        "orientation": "h", "x": 0.0, "y": LEGEND_STRIP, "xanchor": "left", "yanchor": "top", "font": MAP_FONT,
        "itemsizing": "constant", "itemclick": False, "itemdoubleclick": False, "bgcolor": TRANSPARENT,
    }


def scale_trace(colorscale, top: float, title: str, tickformat: str) -> go.Scatter3d:
    """A colour scale inside the map: a colour bar along the full height at the right edge, from 0 to `top`, drawn
    by an invisible marker in the scene's corner (neither hoverable nor clickable)."""
    colorbar = {
        "x": 1.0, "y": 0.5, "xanchor": "right", "yanchor": "middle", "len": 1.0, "thickness": 16, "outlinewidth": 0,
        "tickformat": tickformat, "tickfont": MAP_FONT, "title": {"text": title, "side": "right", "font": {**MAP_FONT, "color": MUTED}},
    }
    return go.Scatter3d(
        x=AXIS_RANGE[:1], y=AXIS_RANGE[:1], z=AXIS_RANGE[:1], mode="markers",
        marker={"size": 0.001, "color": [0.0], "colorscale": colorscale, "cmin": 0.0, "cmax": top, "showscale": True, "colorbar": colorbar},
        hoverinfo="skip", showlegend=False, name="Scale",
    )


def disagreement_trace(points: Mapping, ids: Sequence[str]) -> Optional[go.Scatter3d]:
    """With D: an open ring behind every predicted quiz the two generations put at different levels,
    the legend's entry DISAGREE, neither hoverable nor clickable; None without one. It says "look here"
    and names no winner: the hand labels say who is right."""
    ids = [card_id for card_id in ids if card_id in points]
    if not ids:
        return None
    return go.Scatter3d(
        x=[points[card_id][0] for card_id in ids], y=[points[card_id][1] for card_id in ids], z=[points[card_id][2] for card_id in ids],
        mode="markers", marker={"size": DISAGREE_SIZE, "color": INK, "symbol": DISAGREE_SYMBOL, "line": {"width": 2, "color": INK}},
        hoverinfo="skip", showlegend=True, name=DISAGREE, **_group("D"),
    )


def target_trace(point) -> go.Scatter3d:
    """With P: a gold ring behind P1's target in the view, the legend's entry TARGET."""
    x, y, z = ([value] for value in point)
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker={"size": TARGET_SIZE, "color": GOLD, "symbol": "circle-open", "line": {"width": 3, "color": GOLD}},
        hoverinfo="skip", showlegend=True, name=TARGET, **_group("P"),
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


def margin_scale() -> list:
    """The teal scale as a colour scale: pale at 0, full at 1."""
    return [[0.0, margin_color(0.0, 1.0)], [1.0, margin_color(1.0, 1.0)]]


def faded(color: str, alpha: float = 0.7) -> str:
    """A hex colour at `alpha`, as rgba."""
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def share_color(share: float) -> str:
    """A share on the blue ramp: SHARE_LOW at 0, SHARE_HIGH at 1."""
    share = max(0.0, min(1.0, share))
    low, high = (tuple(int(color[index:index + 2], 16) for index in (1, 3, 5)) for color in (SHARE_LOW, SHARE_HIGH))
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(low, high))
    return f"#{red:02x}{green:02x}{blue:02x}"


def map_figure(page, state: dict) -> go.Figure:
    """The map of the view in `state` (app.HOME's keys) under the layers in force (app.Page.layer_states). The hover
    labels: a deck's id, a topic's label (its id and a hash of its name, app.Page.topic_label), a card's comb_id,
    with T its margin, with D a predicted quiz's level. The legend's strip names the classes the view holds, one
    group per layer (S's two kinds of lines; T's cross, diamonds, emptied and grown topics; D's classes and the
    disagreement ring; P's target), the colour bar T's margin."""
    view, selected = state["view"], state.get("selected")
    states = page.layer_states(state.get("layers"))
    s_on, t_on, d_on = (states.get(letter) == page.ON for letter in "STD")
    c_draws = states.get("C") in (page.ON, page.FALLBACK)
    p_draws = states.get("P") in (page.ON, page.FALLBACK)
    figure = go.Figure()
    figure.add_trace(_anchor())
    symbols = None
    sizes = None
    rings = None
    links = None
    extra: list = []  # the rings and the lines of the layers, under the dots
    scale = None
    if view == DECKS_VIEW:
        points = page.deck_points(state)
        labels = {deck_id: deck_id for deck_id in points}
        colors = {deck_id: DECK_COLOR for deck_id in points}
        if c_draws:
            links = links_trace(points, page.deck_links(), along_surface=False)
    elif view == TOPICS_VIEW:
        if c_draws:
            figure.add_traces(centroid_traces())
        topics = page.topics(state["deck"], state)
        points = page.topic_points(state["deck"], state)
        labels = {page.topic_dot(topic): page.topic_label(topic) for topic in topics}
        colors = {dot: TOPIC_COLOR for dot in points}
        symbols = {page.topic_dot(topic): EMPTIED_SYMBOL for topic in topics if not topic.ids}
        if t_on:
            shares = {page.topic_dot(topic): page.between_share(topic) for topic in topics}
            sizes = {dot: DOT_SIZE + TOPIC_GROWTH * share for dot, share in shares.items()}
            classes = by_symbol(symbols, T_TOPIC_LEGEND)
            grown = max((dot for dot in shares if dot not in classes), key=shares.get, default=None)
            if grown is not None and shares[grown] > 0:
                classes[grown] = GROWN  # the topic grown most stands for the size
            figure.add_traces(legend_traces(points, classes, T_TOPIC_LEGEND, "T", colors, symbols, sizes))
    else:
        topic = page.entered(state)
        figure.add_trace(sphere_surface(GREY))
        points = page.card_points(topic, state) if topic is not None else {}
        labels = {card_id: card_id for card_id in points}
        colors = {card_id: GREY for card_id in points}
        if t_on and topic is not None:
            colors, symbols, labels = page.t1_markers(topic, points)
            entries = t1_entries(page.margin_scale)
            figure.add_traces(legend_traces(points, by_symbol(symbols, entries), entries, "T", colors, symbols))
            scale = scale_trace(margin_scale(), page.margin_scale, "margin", ".2f")
        if d_on:
            report = page.difficulty_of(state)
            if report is not None:
                rings, classes, levels = page.d_rings(points, report)
                labels = {card_id: f"{labels[card_id]} · {levels[card_id]}" if card_id in levels else labels[card_id] for card_id in points}
                figure.add_traces(legend_traces(points, classes, LEGEND, "D", colors, symbols, None, rings))
                ring = disagreement_trace(points, page.disagreements(points, report))
                if ring is not None:
                    extra.append(ring)
        if c_draws and topic is not None:
            links = links_trace(points, page.card_links(topic, state), along_surface=True)
        if s_on:
            pairs, own = page.neighbour_pairs(points, state)
            extra += [trace for trace in (
                links_trace(points, own, True, SELECTED_PAIRS, dash="solid", legend=True, group="S", **SELECTED_LINE),
                links_trace(points, pairs, True, NEIGHBOUR_PAIRS, dash="solid", legend=True, group="S", **NEIGHBOUR_LINE),
            ) if trace is not None]
    if p_draws:
        target = page.target_dot(state)
        if target in points:
            extra.append(target_trace(points[target]))
    if links is not None:
        figure.add_trace(links)
    figure.add_traces(extra)
    if points:
        figure.add_trace(dots_trace(points, labels, colors, view, symbols, sizes, rings))
    if selected in points:
        figure.add_traces(selected_trace(points[selected], colors[selected]))
    legend = any(trace.showlegend for trace in figure.data)
    if scale is not None and points:
        if legend:  # the bar along the sphere's full height, above the legend's strip
            scale.marker.colorbar.update(len=1.0 - LEGEND_STRIP, y=1.0, yanchor="top")
        figure.add_trace(scale)
    figure.update_layout(
        uirevision=f"standalone:{view}:{state.get('deck') or ''}:{state.get('topic') or ''}:{page.positions_of(state)}",
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font={"color": INK, "family": "Georgia"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=legend, legend=legend_style(), hoverlabel={"font": HOVER_FONT},
        scene={
            "aspectmode": "cube", "bgcolor": BACKGROUND, "camera": DEFAULT_CAMERA,
            "xaxis": _axis(), "yaxis": _axis(), "zaxis": _axis(),
            "domain": {"x": [0.0, 1.0 - BAR_WIDTH if scale is not None else 1.0], "y": [LEGEND_STRIP if legend else 0.0, 1.0]},
        },
        meta={"view": view, "layers": dict(states)},
    )
    return figure


def card_class(card, predicted: bool) -> str:
    """A card's class on the map with D, as the viewer colours it after Helper: its authored level; a quiz the
    generation predicted "predicted" (the oldest group there, so its colour decides, not the predicted level's); a
    flashcard or an unrated quiz without a prediction "not rated"."""
    if card.difficulty in LEVEL_COLORS:
        return card.difficulty
    return PREDICTED if predicted else NOT_RATED


def class_style(name: str) -> tuple[str, str]:
    """A class's (colour, symbol) from LEGEND."""
    return next((color, symbol) for class_name, color, symbol in LEGEND if class_name == name)


def layer_options(states: Mapping[str, str], notes: Mapping[str, str] = LAYER_NOTES) -> list[dict]:
    """The check boxes' options: every model letter, its label saying its layer's state where it is not plainly on
    ("C · scripted topics", "T · needs S"; app.Page.layer_states)."""
    options = []
    for letter in PANEL_LETTERS:
        note = notes.get(states.get(letter, ""), "")
        options.append({"label": f"{letter} · {note}" if note else letter, "value": letter})
    return options


# --- the readout tables (C1's measures, S1's and T1's live measures), in Details -----------------------------


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


# --- the figures of the slot and of Details -------------------------------------------------------------------


def graph(figure: go.Figure, class_name: str) -> dcc.Graph:
    """A graph of the page, without plotly's bar, following its slot's size."""
    return dcc.Graph(figure=figure, config={"displayModeBar": False, "responsive": True}, responsive=True, className=class_name,
                     style={"height": "100%", "width": "100%"})


def empty_slot(text: str) -> html.Div:
    """The figure slot, or a part of Details, with a line instead of a figure."""
    return html.Div(html.P(text, className="empty-state"), className="slot-note")


def margin_histogram(margins: Sequence[float], threshold: float, focus: Optional[float] = None, focus_share: float = 0.25) -> go.Figure:
    """The view's margins in bins, the threshold as a dashed line labelled on the plot: below it a card is
    between two topics; with `focus` (the dashboard's) a second line where the `focus_share` most confident begin,
    labelled lower on the plot."""
    figure = go.Figure(go.Histogram(x=list(margins), nbinsx=20, marker={"color": margin_color(1.0, 1.0)}, hoverinfo="skip"))
    figure.add_vline(x=threshold, line={"color": CUT_COLOR, "dash": "dash", "width": 2},
                     annotation={"text": f"between two topics below {threshold:.2f}", "font": {**FONT, "color": CUT_COLOR}},
                     annotation_position="top right")
    if focus is not None:  # its label lower down, both on a dark backing where the lines cross a label
        figure.add_vline(x=focus, line={"color": GOLD, "dash": "dot", "width": 2},
                         annotation={"text": f"the {focus_share * 100:.0f} % most confident above {focus:.2f}", "font": {**FONT, "color": GOLD},
                                     "y": 0.75, "yanchor": "top", "bgcolor": LABEL_BACKING},
                         annotation_position="top right")
        figure.layout.annotations[0].bgcolor = LABEL_BACKING
    figure.update_layout(plain_layout(
        margin={"l": 70, "r": 10, "t": 30, "b": 60},
        xaxis={"title": {"text": "margin: best centroid's cosine − second best's", "font": LABEL_FONT}, "tickfont": FONT,
               "gridcolor": GRID, "zeroline": False, "fixedrange": True},
        yaxis={"title": {"text": "cards", "font": LABEL_FONT}, "tickfont": FONT, "gridcolor": GRID, "zeroline": False, "fixedrange": True},
        bargap=0.05,
    ))
    return figure


def score_table(report) -> html.Table:
    """The key scores: the generation (mean ± sd) and the baselines (their means)."""
    header = html.Tr([html.Th("")] + [html.Th(name) for name in score_header(report)])
    rows = [
        html.Tr([html.Td(label, className="score-label"), html.Td(d1, className="score-model")] + [html.Td(value) for value in baselines])
        for label, d1, baselines in score_rows(report)
    ]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores")


def metrics_panel(report, scope: str) -> html.Div:
    """What the generation (`report.model`, D1 or D2) measured for the view's part of the course (`report`, the
    course's, a deck's or a topic's), in Details: what was measured, the score table and the macro-F1 remark, the
    shares of the levels; or why there is nothing to show (a part without quizzes: the note)."""
    head = [html.Div(f"Difficulty model {report.model}", className="card-label")]
    if not report.measured:
        body = [html.P(report.note or "No difficulty model.", className="empty-state")]
        if report.predictions:
            body += [html.Div("Shares of the levels", className="card-label metrics-heading"), graph(shares_figure(report), "difficulty-shares")]
        return html.Div(head + body, className="tool-panel difficulty-panel")
    return html.Div(
        head
        + [
            html.Div(metrics_subtitle(report), className="deck-description"),
            score_table(report),
            html.Div(MACRO_F1_NOTE, className="card-hint"),
            html.Div("Shares of the levels", className="card-label metrics-heading"),
            graph(shares_figure(report), "difficulty-shares"),
        ],
        className="tool-panel difficulty-panel",
    )


# --- Details: the generation's section ----------------------------------------------------------------------


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
    """A level's results of the newest run: the heads (the chosen variant and the app's marked), one row per
    measure, clickable (its key in `data-measure`; assets/panel_rows.js sends it), the open row's description under
    it; the bookkeeping as the subtitle."""
    header = html.Tr([html.Th("")] + [html.Th(head, className="chosen" if chosen else None) for head, chosen in zip(table.heads, table.chosen)])
    rows = []
    for (label, values), columns in zip(table.rows, table.row_columns):
        key = measure_key(prefix, columns)
        rows.append(html.Tr(
            [html.Td(label, className="score-label")] + [html.Td(value, className="chosen" if chosen else None) for value, chosen in zip(values, table.chosen)],
            className="measure-row open" if key == expanded else "measure-row", title=ROW_HINT, **{"data-measure": key},
        ))
        if key == expanded:
            rows.append(description_row(glossary_entry(glossary, table.file, columns[0]), len(table.heads) + 1))
    return html.Div(
        [
            html.Div(table.title, className="card-label metrics-heading"),
            html.Div(table.subtitle, className="deck-description"),
            html.Div(html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores results-table"), className="results-scroll"),
        ],
        className="results-block",
    )


def level_section(description: str, live: Sequence, tables: Optional[Sequence[Table]], hint: str, level: str,
                  expanded: Optional[str], glossary: Mapping[str, tuple[str, str, str]], moved: Sequence = ()) -> html.Div:
    """Details' content: the open generation's section (what it is, its live measures, its results tables with
    the one open row's description, or why there is none: `hint`, how to make the file, or that it holds no rows of
    the generation), then what moved in (`moved`: the dendrogram while the slot shows another figure, the hints)."""
    body = [html.Div(description, className="deck-description")] + list(live)
    if not tables:
        body.append(html.P(hint, className="empty-state"))
    else:
        body += [results_table(table, f"{level}:{index}", expanded, glossary) for index, table in enumerate(tables)]
    return html.Div(body + list(moved), className="level-section details")


# --- the right column's regions ------------------------------------------------------------------------------


def tab_bar() -> html.Div:
    """The tab bar: one button per model letter with its name; the callback sets each tab's class."""
    return html.Div(
        [
            html.Button([html.Span(letter, className="tab-letter"), html.Span(name, className="tab-name")],
                        id=f"tab-{letter}", n_clicks=0, className=tab_class(False), title=f"{name}: open or close")
            for letter, name in PANEL_NAMES.items()
        ],
        id="tabs", className="tab-bar",
    )


def tab_class(is_open: bool) -> str:
    return "tab open" if is_open else "tab"


def level_tabs(letter: Optional[str], expanded: Optional[str], app_level: Optional[str]) -> list:
    """The open model's generations as small tabs in GENERATIONS' fixed order, the oldest first, each clickable
    (its generation in `data-level`; assets/panel_rows.js sends it), the open one marked, the one the app runs
    with the "app" mark; none without an open tab."""
    tabs = []
    for level in GENERATIONS.get(letter, ()):
        is_open = level == expanded
        children = [html.Span(GENERATION_NAMES[level], className="level-name")]
        if level == app_level:
            children.append(html.Span(APP_MARK, className="app-mark", title="The generation the app runs"))
        tabs.append(html.Div(children, className="level-tab open" if is_open else "level-tab", title=None if is_open else LEVEL_HINT,
                             **{"data-level": level}))
    return tabs


def number_block(headline: Headline) -> html.Div:
    """One of the section's two numbers: the value large, its label, the contrast (the floor's value, or the gain
    over the generation before), the sentence, the verdict and its explanation in words."""
    contrast = []
    if headline.contrast is not None:
        contrast.append(html.Span(f"against {headline.contrast_shown}", className="number-contrast"))
    if headline.gain:
        contrast.append(html.Span(headline.gain, className="number-gain"))
    if headline.of:
        contrast.insert(0, html.Span(headline.of, className="number-of"))
    verdict = [html.Span(headline.verdict, className="verdict-word")]
    if headline.explanation:
        verdict.append(html.Span(f": {headline.explanation}", className="verdict-words"))
    return html.Div(
        [
            html.Div(headline.shown, className="number-value"),
            html.Div(headline.label, className="number-label"),
            html.Div(contrast, className="number-contrast-line"),
            html.P(headline.sentence, className="number-sentence"),
            html.Div(verdict, className="number-verdict"),
        ],
        className="number-block",
    )


def section_head(level: str, numbers: Optional[Sequence[Headline]], variant: str, hint: str, scope: str) -> list:
    """The section's head: the headline (what the model does, and the scope), the two numbers, the chosen variant
    in plain words; without numbers the hint (no file, or no rows of the generation)."""
    head = [html.Div([html.Span(WHAT[level], className="headline-text"), html.Span(scope, className="headline-scope")], className="headline")]
    if not numbers:
        return head + [html.P(hint, className="empty-state")]
    head.append(html.Div([number_block(headline) for headline in numbers], className="numbers"))
    if variant:
        head.append(html.Div([html.Span("Variant: ", className="variant-label"), html.Span(plain_words(variant))], className="variant"))
    return head


def no_tab_line() -> list:
    """The section with no tab open: one line on what the five models are."""
    return [html.Div(NO_TAB, className="section-none")]


def live_line(text: str) -> list:
    """What stays on top for the live demo: T's line of the view next to the course."""
    return [html.Div(text, className="live-line")]


def details_class(is_open: bool, shown: bool) -> str:
    """The Details button's class: open, closed, or hidden with no tab open."""
    if not shown:
        return "details-toggle hidden"
    return "details-toggle open" if is_open else "details-toggle"


def prompt_box() -> html.Div:
    """P's request box: the text input (Enter submits) and Go, in the layout from the start; shown with P's tab open
    (its class), disabled where P1 does not function."""
    return html.Div(
        [
            dcc.Input(id="prompt-text", type="text", value="", n_submit=0, placeholder=PROMPT_PLACEHOLDER, className="prompt-input",
                      autoComplete="off"),
            html.Button("Go", id="prompt-go", n_clicks=0, className="top-button prompt-go"),
        ],
        id="prompt-box", className=prompt_class(False),
    )


def prompt_class(shown: bool) -> str:
    return "prompt-box" if shown else "prompt-box hidden"


def prompt_answer(request: str, decision: str, entries, scores: bool = False) -> list:
    """P1's answer by ids: the request, the decision line, then the three nearest topics, each a row with its label
    (and its nearest card) and, in Details, every route's score, a click going to it (`data-target`: the deck id,
    the topic id and the card where one is shown)."""
    rows = [html.Div(f"“{request}”", className="prompt-request"), html.Div(decision, className="prompt-decision")]
    for target, label, route_scores in entries:
        parts = [html.Span(label, className="prompt-label")]
        if scores:
            parts.append(html.Span(route_scores, className="prompt-scores"))
        rows.append(html.Div(parts, className="prompt-target", title=TARGET_HINT, **{"data-target": target}))
    return [html.Div(rows, className="prompt-answer")]


def topic_counts_of(topic, card_by_id) -> list[tuple[str, int]]:
    """A topic's size and mix of cards under the membership: (what, how many)."""
    cards = [card_by_id[card_id] for card_id in topic.ids if card_id in card_by_id]
    quizzes = sum(card.stack in QUIZ_STACKS for card in cards)
    rated = sum(card.difficulty is not None for card in cards)
    return [("cards", len(cards)), ("flashcards", len(cards) - quizzes), ("quizzes", quizzes), ("rated", rated)]


def tag_strip_body(title: str, tags: Sequence[tuple[str, str]]) -> list:
    """The tag strip's content: the large title ("Card 418001") and a few tags, each (text, class): a plain tag,
    a level's coloured tag ("easy", "predicted"), a layer's ("s", "t")."""
    return [
        html.Div(title, className="tag-title"),
        html.Div([html.Span(text, className=f"tag {kind}".strip()) for text, kind in tags], className="tags"),
    ]


def tag_class(shown: bool, can_open: bool) -> str:
    """The tag strip's class: hidden with nothing selected; without Open for a card."""
    if not shown:
        return "tag-strip hidden"
    return "tag-strip" if can_open else "tag-strip no-open"


# --- the page ---------------------------------------------------------------------------------------------------


def page_layout(home: dict, viewer_url: Optional[str] = None) -> html.Main:
    """The page; the callback fills the map, the check boxes' labels, the breadcrumb, the tabs, the generation
    tabs, the section, the tag strip and the figure slot at load. The hidden input `row-click` carries a click on a
    generation's tab, a measure's row or one of P1's targets (assets/panel_rows.js) to the callback. With
    `viewer_url` the top bar holds the Sputnik link next to Back and Open, opening the viewer in a new tab."""
    actions = [
        html.Button("Back", id="back", className="top-button", n_clicks=0, disabled=True),
        html.Button("Open", id="open", className="top-button", n_clicks=0, disabled=True),
    ]
    if viewer_url:
        actions.append(html.A(VIEWER_LINK, id="viewer-link", href=viewer_url, target="_blank", rel="noopener", title=VIEWER_LINK_TITLE,
                              className="top-button top-link"))
    return html.Main(
        [
            html.Header(
                [
                    html.Div([html.P("GUIDED LEARNING · THE MODELS", className="eyebrow"), html.H1("What the models see.")]),
                    html.Div(actions, className="topbar-actions"),
                ],
                className="topbar",
            ),
            html.Section([html.Div(id="where", className="breadcrumb")], className="toolbar"),
            html.Section(
                [
                    html.Div(
                        [
                            dcc.Graph(id="map", config={"displayModeBar": False, "scrollZoom": True}, responsive=True),
                            dcc.Checklist(id="layers", options=layer_options({}), value=[], className="layers", inputClassName="layer-box",
                                          labelClassName="layer-label"),
                        ],
                        className="map-frame",
                    ),
                    html.Div(
                        [
                            tab_bar(),
                            html.Div(id="levels", className="level-tabs"),
                            html.Div(
                                [
                                    html.Div(id="section-head", className="section-head"),
                                    prompt_box(),
                                    html.Div(id="section-live", className="section-live"),
                                    html.Button(DETAILS, id="details-toggle", n_clicks=0, className=details_class(False, False)),
                                ],
                                id="section", className="section",
                            ),
                            html.Div(
                                [html.Div(id="tag-body", className="tag-body"), html.Button("Open", id="tag-open", n_clicks=0, className="top-button tag-open")],
                                id="tag-strip", className=tag_class(False, False),
                            ),
                            html.Div(id="figure-slot", className="figure-slot"),
                        ],
                        className="right-column",
                    ),
                ],
                className="workspace",
            ),
            dcc.Store(id="view", data=dict(home)),
            dcc.Input(id="row-click", value="", type="text", className="row-click"),
        ],
        className="app-shell",
    )
