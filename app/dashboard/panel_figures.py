"""The figures of the dashboard's model panels (plotly only; nothing of Dash): each the whole
course's, whatever card is shown, but C's mark of the shown card's deck. T's histogram and D's confusion matrix are
app.figures' (figures.margin_histogram, figures.confusion_figure); the three drawn here:

* S, where the related cards lie (places_figure): one bar split into this topic, this deck and another deck, seen
  from each card, in the places' colours (figures.PLACE_COLORS), each part with its share, a bracket over
  the two parts outside the card's topic with their share together, the counts under the bar;
* C, the decks on the sphere (deck_sphere): the faint surface, the decks muted at C1's positions, the connections
  below the preset gold and dashed, straight between decks, the shown
  card's deck gold with a soft gold ring and turned to the front;
* P, P1's outcomes (outcome_grid): the judged requests in a grid, on and off the course by where P1 took them (into
  an acceptable topic, into another topic, no match), each cell its count and its outcome (right, wrong, missed) in
  the card's judgement colours; an off-course request has no acceptable topic, so that cell stays empty.

Every figure takes the size of its place (no fixed height) and no font is below 16 px (figures.FONT_SIZE).
"""

import math
from typing import Mapping, Optional, Sequence

import plotly.graph_objects as go

from ..figures import (
    DARK_INK, DECK_COLOR, DEFAULT_CAMERA, DOT_SIZE, ELSEWHERE, FONT, GOLD, GREY, HERE, HOVER_FONT, INK, LABEL_FONT, MUTED, PLACE_COLORS,
    SELECTED_SIZE, THIS_DECK, TRANSPARENT, Point, links_trace, plain_layout, sphere_surface,
)

PLACES = (HERE, THIS_DECK, ELSEWHERE)  # near to far, the bar's parts from the left
SURFACE = "#071827"  # the panel's background: the gap between the bar's parts
BAR = 0.9  # the bar's thickness on its axis of -1 to 1
CURRENT_DECK = "this card's deck"
A_DECK = "a deck"
EYE = 1.3  # the camera's distance from the centre: nearer than the default camera's (2.17), so the sphere fills its place
SCENE_RANGE = [-1.05, 1.05]  # the scene's box just around the sphere
RING_ALPHA = 0.3  # the soft gold ring behind the shown card's deck, as the diagram's current dot
LEGEND_SHARE = 0.12  # the share of the sphere's height under it for the legend
# P1's outcomes: right, wrong, missed in the card's judgement colours (style.css: --right, --coral, --muted)
RIGHT, WRONG, MISSED = "right", "wrong", "missed"
OUTCOME_COLORS = {RIGHT: "#69c6a5", WRONG: "#e76f51", MISSED: MUTED}
OUTCOME_CODES = {RIGHT: 0.0, WRONG: 0.5, MISSED: 1.0}
LANDINGS = ("into an acceptable topic", "into another topic", "no match")
WRAPPED = {"into an acceptable topic": "into an<br>acceptable<br>topic", "into another topic": "into<br>another<br>topic"}  # the column heads


def percent(share: float) -> str:
    """A share as the page shows it: "56 %"."""
    return f"{share * 100:.0f} %"


def places_figure(counts: Mapping[str, int], cards: int) -> go.Figure:
    """Where the course's related cards lie, seen from their card: one bar of three parts (PLACES), each its name and
    share inside it, a 2 px gap between them; a bracket over the parts outside the card's topic with their share
    together; under the bar the count of the related cards and of the cards they belong to."""
    total = sum(counts.get(place, 0) for place in PLACES)
    figure = go.Figure()
    for place in PLACES:
        count = counts.get(place, 0)
        share = count / total if total else 0.0
        figure.add_trace(go.Bar(
            x=[share], y=[0], orientation="h", width=BAR, name=place,
            marker={"color": PLACE_COLORS[place], "line": {"color": SURFACE, "width": 2}},
            text=[f"{place}<br>{percent(share)}"], textposition="inside", insidetextanchor="middle", constraintext="none",
            textfont={**LABEL_FONT, "color": INK if place == ELSEWHERE else DARK_INK},
            hovertemplate=f"{place}: {count:,} of {total:,} related cards ({percent(share)})<extra></extra>",
        ))
    here = counts.get(HERE, 0) / total if total else 0.0
    top = BAR / 2 + 0.12
    for x0, y0, x1, y1 in ((here, top, 1.0, top), (here, top, here, top - 0.08), (1.0, top, 1.0, top - 0.08)):
        figure.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line={"color": INK, "width": 2})
    figure.add_annotation(x=1.0, y=top + 0.04, xanchor="right", yanchor="bottom", showarrow=False,
                          text=f"outside the card's topic: {percent(1.0 - here)}", font=LABEL_FONT)
    figure.add_annotation(x=0.5, y=-BAR / 2 - 0.1, yanchor="top", showarrow=False, font={**FONT, "color": MUTED},
                          text=f"{total:,} related cards of {cards:,} cards, the five nearest of each")
    figure.update_layout(plain_layout(
        margin={"l": 10, "r": 10, "t": 40, "b": 10}, barmode="stack", hoverlabel={"font": FONT},
        legend={"orientation": "h", "x": 0, "y": 1.0, "yanchor": "bottom", "font": {**FONT, "color": MUTED}, "traceorder": "normal"},
        xaxis={"visible": False, "range": [0, 1], "fixedrange": True},
        yaxis={"visible": False, "range": [-1, 1], "fixedrange": True},
    ))
    return figure


def _axis() -> dict:
    """A hidden axis of the scene at a fixed range just around the sphere."""
    return {"visible": False, "range": SCENE_RANGE, "showspikes": False}


def _marker_trace(ids: Sequence[str], points: Mapping[str, Point], names: Mapping[str, str], name: str, marker: dict,
                  legend: bool = True, opacity: float = 1.0) -> go.Scatter3d:
    """Decks as one trace of markers, each named on hover by its id and its name."""
    return go.Scatter3d(
        x=[points[dot][0] for dot in ids], y=[points[dot][1] for dot in ids], z=[points[dot][2] for dot in ids],
        mode="markers", marker=marker, opacity=opacity, name=name, showlegend=legend,
        text=[f"{dot} · {names.get(dot, dot)}" for dot in ids], hovertemplate="%{text}<extra></extra>",
    )


def camera_on(point: Optional[Point]) -> dict:
    """The camera at EYE from the centre, turned so that the point faces the front; without one along the default
    camera's direction. Near a pole the up direction turns to y, so the view keeps its roll."""
    if point is None:
        point = tuple(DEFAULT_CAMERA["eye"].values())
    length = math.dist((0, 0, 0), point) or 1.0
    eye = {axis: value / length * EYE for axis, value in zip("xyz", point)}
    up = {"x": 0, "y": 1, "z": 0} if abs(point[2]) / length > 0.95 else DEFAULT_CAMERA["up"]
    return {**DEFAULT_CAMERA, "eye": eye, "up": up}


def deck_sphere(points: Mapping[str, Point], names: Mapping[str, str], links: Sequence[tuple[str, str]], current: Optional[str],
                cut: str) -> go.Figure:
    """The decks on the sphere at C1's positions: the faint surface, the connections below the preset (`cut` names
    it), the decks muted, the shown card's deck (`current`) gold with a soft gold ring and turned to the front; the
    legend under the sphere."""
    figure = go.Figure()
    figure.add_trace(sphere_surface(GREY))
    others = [dot for dot in points if dot != current]
    figure.add_trace(_marker_trace(others, points, names, A_DECK, {"size": DOT_SIZE, "color": DECK_COLOR}))
    if current in points:
        figure.add_trace(_marker_trace([current], points, names, "ring", {"size": SELECTED_SIZE * 1.6, "color": GOLD}, False, RING_ALPHA))
        figure.add_trace(_marker_trace([current], points, names, CURRENT_DECK, {"size": SELECTED_SIZE, "color": GOLD}))
    line = links_trace(points, links, name=f"connections: {cut}", legend=True)
    if line is not None:
        figure.add_trace(line)
    figure.update_layout(plain_layout(
        uirevision=f"deck-sphere:{current or ''}", margin={"l": 0, "r": 0, "t": 0, "b": 0}, hoverlabel={"font": HOVER_FONT},
        legend={"orientation": "h", "x": 0, "y": 0, "yanchor": "bottom", "font": {**FONT, "color": MUTED}, "itemclick": False,
                "itemdoubleclick": False},
        scene={
            "aspectmode": "cube", "bgcolor": TRANSPARENT, "camera": camera_on(points.get(current) if current else None),
            "xaxis": _axis(), "yaxis": _axis(), "zaxis": _axis(), "domain": {"x": [0.0, 1.0], "y": [LEGEND_SHARE, 1.0]},
        },
    ))
    return figure


def outcome_grid(counts: Mapping[str, int]) -> go.Figure:
    """P1's judged requests by where it took them (evaluate.py's counts, headlines.OUTCOME_COLUMNS): the rows on and
    off the course with their sizes, the columns into an acceptable topic, into another topic and no match; each cell
    its count and its outcome in its colour, the off-course request's acceptable topic an empty cell."""
    on = counts["hand_course_right"] + counts["hand_course_wrong"] + counts["hand_course_missed"]
    off = counts["hand_off_right"] + counts["hand_off_wrong"]
    rows = (f"on the course<br>({on})", f"off the course<br>({off})")
    cells = (
        ((counts["hand_course_right"], RIGHT), (counts["hand_course_wrong"], WRONG), (counts["hand_course_missed"], MISSED)),
        (None, (counts["hand_off_wrong"], WRONG), (counts["hand_off_right"], RIGHT)),
    )
    meaning = {
        (0, 0): "landed in an acceptable topic", (0, 1): "landed in another topic", (0, 2): "no match though the course teaches it",
        (1, 1): "landed in a topic though no deck teaches it", (1, 2): "no match, as no deck teaches it",
    }
    scale = [[0.0, OUTCOME_COLORS[RIGHT]], [0.25, OUTCOME_COLORS[RIGHT]], [0.25, OUTCOME_COLORS[WRONG]], [0.75, OUTCOME_COLORS[WRONG]],
             [0.75, OUTCOME_COLORS[MISSED]], [1.0, OUTCOME_COLORS[MISSED]]]
    z = [[None if cell is None else OUTCOME_CODES[cell[1]] for cell in row] for row in cells]
    hover = [[
        "" if cell is None else f"{cell[0]} of {(on, off)[row]} {('on', 'off')[row]}-course requests: {meaning[(row, column)]}, {cell[1]}"
        for column, cell in enumerate(cells[row])
    ] for row in range(2)]
    figure = go.Figure(go.Heatmap(
        z=z, x=list(LANDINGS), y=list(rows), zmin=0, zmax=1, colorscale=scale, showscale=False, xgap=2, ygap=2,
        text=hover, hovertemplate="%{text}<extra></extra>", hoverongaps=False,
    ))
    for row, row_label in enumerate(rows):
        for column, landing in enumerate(LANDINGS):
            cell = cells[row][column]
            if cell is None:
                figure.add_annotation(x=landing, y=row_label, text="–", showarrow=False, font={**LABEL_FONT, "color": MUTED})
                continue
            count, outcome = cell
            figure.add_annotation(x=landing, y=row_label, text=str(count), showarrow=False, yshift=14,
                                  font={**LABEL_FONT, "size": 32, "color": DARK_INK})
            figure.add_annotation(x=landing, y=row_label, text=outcome, showarrow=False, yshift=-18, font={**LABEL_FONT, "color": DARK_INK})
    figure.update_layout(plain_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10}, hoverlabel={"font": FONT},
        xaxis={"side": "top", "tickfont": LABEL_FONT, "tickangle": 0, "tickvals": list(LANDINGS), "fixedrange": True,
               "ticktext": [WRAPPED.get(landing, landing) for landing in LANDINGS], "showgrid": False, "zeroline": False, "automargin": True},
        yaxis={"autorange": "reversed", "tickfont": LABEL_FONT, "fixedrange": True, "showgrid": False, "zeroline": False,
               "automargin": True},
    ))
    return figure
