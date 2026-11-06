"""The models' plotly figures of the dashboard's panels (plotly only: no Dash), with the theme's tokens every figure
of the page is drawn in:

* D1's confusion matrix (confusion_figure) of the difficulty report (models.difficulty_model.DifficultyReport);
* the sphere: its surface (sphere_surface) and the connections as one trace of straight lines (links_trace); the
  colours and sizes of the dashboard's decks sphere (GREY, DECK_COLOR, DOT_SIZE, SELECTED_SIZE, DEFAULT_CAMERA,
  HOVER_FONT);
* T1's margins in bins with the threshold, and the dashboard's focus line (margin_histogram), on the margins' teal
  scale (margin_color);
* where a related card lies and the colour of each place (HERE, THIS_DECK, ELSEWHERE, PLACE_COLORS).

Every figure takes the size of its slot (no fixed height: the page's figure slot gives it), and no font is below
FONT_SIZE (16 px): a page's type scale (its assets/style.css) reaches no plotly figure, so the sizes are
set here. Colours: the sequential ramp is the dataviz reference steps for a dark surface, validated against the
panel's background; the levels are named on the axes, so no colour carries them alone. The theme's tokens (INK,
MUTED, GOLD) are the stylesheet's values, written here anew.
"""

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from models.difficulty_model import LEVELS, DifficultyReport

INK = "#f7f2e8"
MUTED = "#9fb3c8"
GOLD = "#f5b942"
GRID = "rgba(247, 242, 232, 0.12)"
DARK_INK = "#071827"  # the numbers on the light cells of the matrix
SEQUENTIAL = [[0.0, "#0d366b"], [0.5, "#3987e5"], [1.0, "#cde2fb"]]  # blue, from near the surface to light
CUT_COLOR = "#e06c75"  # the threshold's line
FONT_SIZE = 16  # the least font of every figure
LABEL_SIZE = 18  # the axes' titles, the matrix's cells, the labels on the plots
FONT = {"color": INK, "family": "DM Mono, monospace", "size": FONT_SIZE}
LABEL_FONT = {**FONT, "size": LABEL_SIZE}
SURFACE_ALPHA = 0.1  # faint, so the cards and their connections stand out
SURFACE_STEPS = 24  # the sphere's mesh: this many steps around and half as many from pole to pole
TRANSPARENT = "rgba(0, 0, 0, 0)"
# Where a related card lies, seen from its card, and the place's colour: its topic, its deck, another deck
HERE, THIS_DECK, ELSEWHERE = "this topic", "this deck", "another deck"
PLACE_COLORS = {HERE: "#cde2fb", THIS_DECK: "#3987e5", ELSEWHERE: "#0d366b"}
GREY = "#7c8b99"  # the colour of a dot in no group: a flashcard, the cards' sphere
DECK_COLOR = MUTED  # the decks in one colour
LINK_WIDTH = 4
DOT_SIZE = 13  # every view's dots
SELECTED_SIZE = 20
DEFAULT_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}
HOVER_FONT = {"color": INK, "family": "DM Mono, monospace", "size": 16}
MARGIN_PALE, MARGIN_FULL = (204, 251, 241), (20, 184, 166)  # the teal scale of a card's margin: #ccfbf1 to #14b8a6
LABEL_BACKING = "rgba(7, 24, 39, 0.95)"  # a label on the plot over a line: the line hidden behind it

Point = tuple[float, float, float]


def plain_layout(**kwargs) -> dict:
    """A figure's layout on the page: transparent, the page's font, no drag, the size of its slot."""
    return {"paper_bgcolor": TRANSPARENT, "plot_bgcolor": TRANSPARENT, "font": FONT, "dragmode": False, "autosize": True, **kwargs}


# --- D1's confusion matrix -------------------------------------------------------------------------------------


def confusion_figure(report: DifficultyReport) -> go.Figure:
    """Rows: the true level, columns: the predicted level, each row's shares (the diagonal is the recall); the cells
    at LABEL_SIZE."""
    shares = report.confusion
    figure = go.Figure(
        go.Heatmap(
            z=shares, x=list(LEVELS), y=list(LEVELS), zmin=0, zmax=1, colorscale=SEQUENTIAL, showscale=False, xgap=2, ygap=2,
            hovertemplate="true %{y}, predicted %{x}: %{z:.0%}<extra></extra>",
        )
    )
    for row, true_level in enumerate(LEVELS):
        for column, predicted_level in enumerate(LEVELS):
            value = float(shares[row][column])
            figure.add_annotation(
                x=predicted_level, y=true_level, text=f"{value:.0%}", showarrow=False,
                font={**LABEL_FONT, "color": DARK_INK if value > 0.55 else INK},
            )
    figure.update_layout(plain_layout(
        margin={"l": 80, "r": 10, "t": 10, "b": 60}, hoverlabel={"font": FONT},
        xaxis={"title": {"text": "predicted", "font": {**LABEL_FONT, "color": MUTED}}, "side": "bottom", "fixedrange": True},
        yaxis={"title": {"text": "true", "font": {**LABEL_FONT, "color": MUTED}}, "autorange": "reversed", "fixedrange": True},
    ))
    return figure


# --- the sphere -------------------------------------------------------------------------------------------------


def sphere_surface(color: str, radius: float = 1.0, steps: int = SURFACE_STEPS, opacity: float = SURFACE_ALPHA) -> go.Surface:
    """The surface of the sphere the cards sit on, faint in one colour, neither selectable nor hoverable."""
    rings = max(2, steps // 2)
    polar = np.linspace(0.0, math.pi, rings + 1)[:, None]
    around = np.linspace(0.0, 2 * math.pi, steps + 1)[None, :]
    return go.Surface(
        x=(radius * np.sin(polar) * np.cos(around)).tolist(),
        y=(radius * np.sin(polar) * np.sin(around)).tolist(),
        z=(radius * np.cos(polar) * np.ones_like(around)).tolist(),
        surfacecolor=np.zeros((rings + 1, steps + 1)).tolist(),
        colorscale=[[0, color], [1, color]], cmin=0, cmax=1, showscale=False,
        opacity=opacity, hoverinfo="skip", showlegend=False, name="Surface",
        contours={"x": {"highlight": False}, "y": {"highlight": False}, "z": {"highlight": False}},
    )


def links_trace(points: Mapping, pairs, name: str = "Connections", legend: bool = False) -> Optional[go.Scatter3d]:
    """Pairs of drawn dots as one trace of straight lines (None between two lines), gold and dashed: the model's
    connections, an entry of the legend by their name where `legend` says so. None without a pair of two drawn
    dots."""
    xs: list = []
    ys: list = []
    zs: list = []
    for source, target in pairs:
        if source not in points or target not in points:
            continue
        xs += [points[source][0], points[target][0], None]
        ys += [points[source][1], points[target][1], None]
        zs += [points[source][2], points[target][2], None]
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line={"color": GOLD, "width": LINK_WIDTH, "dash": "dash"}, hoverinfo="skip",
        showlegend=legend, name=name,
    )


# --- T1's margins -----------------------------------------------------------------------------------------------


def margin_color(margin: float, scale: float) -> str:
    """A card's margin on the teal scale: pale at 0, full at `scale` (the course's 95th percentile) and above."""
    share = 0.0 if scale <= 0 else max(0.0, min(1.0, margin / scale))
    red, green, blue = (round(pale + (full - pale) * share) for pale, full in zip(MARGIN_PALE, MARGIN_FULL))
    return f"#{red:02x}{green:02x}{blue:02x}"


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
