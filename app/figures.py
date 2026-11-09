"""The helper's plotly figures (plotly only: no Dash, nothing of the viewer), drawn by the dashboard and the standalone
page:

* the difficulty models D1 and D2 (models.difficulty_model.DifficultyReport): the line that says what was measured
  (metrics_subtitle), the score table's rows (score_header, score_rows), the confusion matrix (confusion_figure) and
  the shares of the levels (shares_figure);
* the sphere: its surface (sphere_surface), the great-circle arcs along it (surface_arc), the cards view's
  connections, and the connections as one trace, straight or along the surface (links_trace); the map's colours and
  sizes the dashboard's decks sphere shares with the page's map (GREY, DECK_COLOR, DOT_SIZE, SELECTED_SIZE,
  DEFAULT_CAMERA, HOVER_FONT);
* T1's margins in bins with the threshold, and the dashboard's focus line (margin_histogram), on the margins' teal
  scale (margin_color);
* the dendrogram of a clustering (dendrogram_figure): scipy's linkage of the same vectors the connections come from
  (models.clustering), laid out by scipy's coordinates, the real merge distances as heights, with the cut-off line
  above the last kept merge, labelled on the plot;
* a selected card's neighbours (neighbours_figure): a bar per neighbour, its length the score, coloured by where
  the neighbour lies.

Every figure takes the size of its slot (no fixed height: the page's figure slot gives it), and no font is below
FONT_SIZE (16 px): the page's type scale (standalone/assets/style.css) reaches no plotly figure, so the sizes are
set here. Colours: the sequential ramp and the two series colours are the dataviz reference steps for a dark
surface, validated against the panel's background; the levels are named on the axes, so no colour carries them
alone. The theme's tokens (INK, MUTED, GOLD) are the viewer's values, written here anew.
"""

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

from .models.difficulty_model import LEVELS, DifficultyReport

INK = "#f7f2e8"
MUTED = "#9fb3c8"
GOLD = "#f5b942"
GRID = "rgba(247, 242, 232, 0.12)"
DARK_INK = "#071827"  # the numbers on the light cells of the matrix
SEQUENTIAL = [[0.0, "#0d366b"], [0.5, "#3987e5"], [1.0, "#cde2fb"]]  # blue, from near the surface to light
RATED_COLOR = "#3987e5"  # blue: the rated quizzes (training)
PREDICTED_COLOR = "#c98500"  # yellow: the predicted unrated quizzes (test)
CUT_COLOR = "#e06c75"  # the cut-off line, as the viewer draws it
FONT_SIZE = 16  # the least font of every figure
LABEL_SIZE = 18  # the axes' titles, the matrix's cells, the labels on the plots
FONT = {"color": INK, "family": "DM Mono, monospace", "size": FONT_SIZE}
LABEL_FONT = {**FONT, "size": LABEL_SIZE}
TABLE_ROWS = (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced accuracy"), ("kappa", "weighted kappa"), ("accuracy", "accuracy"))
SURFACE_ALPHA = 0.1  # faint, so the cards and their connections stand out
SURFACE_STEPS = 24  # the sphere's mesh: this many steps around and half as many from pole to pole
ARC_STEP = math.radians(5)  # an arc is sampled every 5 degrees
ARC_LIFT = 1.01  # the arcs sit a hair above the surface, clear of the mesh's facets
LABEL_LENGTH = 18  # the dendrogram's leaf labels
LEAF_LABELS = 12  # the dendrogram labels its leaves up to this many; beyond, the hover alone names them
LABEL_MARGIN = (28, 9)  # the space under the dendrogram: a base and a step per character of its longest label
CUT_LIFT = 0.04  # a cut-off line above the whole tree sits this share of its height above the top merge
TRANSPARENT = "rgba(0, 0, 0, 0)"
# Where a neighbour lies, and its bar's colour: the entered topic, the entered deck, another deck
HERE, THIS_DECK, ELSEWHERE = "this topic", "this deck", "another deck"
PLACE_COLORS = {HERE: "#cde2fb", THIS_DECK: "#3987e5", ELSEWHERE: "#0d366b"}
GREY = "#7c8b99"  # the viewer's colour of a dot in no group: a flashcard, the cards' sphere
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


# --- D1's metrics ---------------------------------------------------------------------------------------------


def number(value: float) -> str:
    return "–" if value != value else f"{value:.2f}"  # NaN -> dash


def spread(value: tuple[float, float]) -> str:
    mean, sd = value
    return f"{number(mean)} ± {number(sd)}"


def score_header(report: DifficultyReport) -> list[str]:
    """The score table's columns after the row labels: the report's model (D1 or D2), then the baselines."""
    return [report.model] + list(report.baselines)


def score_rows(report: DifficultyReport) -> list[tuple[str, str, list[str]]]:
    """The key scores, one row per metric: its label, the model as mean ± sd, and the baselines' means."""
    return [
        (label, spread(report.metrics[key]), [number(report.baselines[name][key][0]) for name in report.baselines])
        for key, label in TABLE_ROWS
    ]


def _levels(report: DifficultyReport) -> str:
    easy, medium, hard = report.rated_counts
    return f"{easy} easy, {medium} medium, {hard} hard"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}zes"


def metrics_subtitle(report: DifficultyReport) -> str:
    """What was measured: the whole course, a deck's part, or a topic's part with the deck's metrics; D1's out of
    fold in its cross-validation, D2's (no folds: nothing fitted) in one pass."""
    if report.folds:
        run = f"{report.folds} folds × {report.repeats} repeats"
        course, deck_run = f"out of fold ({run})", f"out of fold in the course's cross-validation ({run})"
    else:
        course = deck_run = "in one pass, nothing fitted"
    if report.scope is None:
        return (
            f"All decks: {report.rated} rated quizzes ({_levels(report)}), measured {course}; levels by each "
            f"deck's shares. {report.unrated} unrated quizzes predicted; plain flashcards are not rated."
        )
    if report.parent is not None:  # a topic: its counts, the deck's metrics
        deck = report.parent
        rated = f"its {_count(report.rated, 'rated quiz')} ({_levels(report)})" if report.rated else "no rated quizzes"
        unrated = (
            f"{_count(report.unrated, 'unrated quiz')} ({len(report.predictions)} predicted)" if report.unrated
            else "no unrated quizzes"
        )
        return (
            f"This topic: {rated} and {unrated}. The scores, the matrix and the baselines are the deck's: its {deck.rated} "
            f"rated quizzes ({_levels(deck)}), {deck_run}, by the deck's shares; "
            f"a topic's few rated quizzes are too few to measure on their own."
        )
    predicted = f"{report.unrated} unrated quizzes of this deck predicted." if report.unrated else "No unrated quiz in this deck."
    return (
        f"This deck: its {report.rated} rated quizzes ({_levels(report)}), {deck_run}, by the deck's shares. {predicted} "
        f"The decks view shows all decks."
    )


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


def shares_figure(report: DifficultyReport) -> go.Figure:
    """The share of each level: the rated quizzes next to the predicted unrated quizzes (a series without quizzes,
    as in most decks, is left out)."""
    predicted_counts = report.predicted_counts
    total = sum(predicted_counts)
    series = []
    if report.rated:
        series.append((f"rated ({report.rated})", [count / report.rated for count in report.rated_counts], report.rated_counts, RATED_COLOR))
    if total:
        series.append((f"predicted ({total})", [count / total for count in predicted_counts], predicted_counts, PREDICTED_COLOR))
    figure = go.Figure()
    for name, values, counts, color in series:
        figure.add_trace(
            go.Bar(
                x=list(LEVELS), y=values, name=name, marker={"color": color, "cornerradius": 4},
                text=[f"{value:.0%}" for value in values], textposition="outside", textfont={**FONT, "color": INK},
                customdata=list(counts), hovertemplate=f"{name}, %{{x}}: %{{y:.0%}} (%{{customdata}} quizzes)<extra></extra>",
                cliponaxis=False,
            )
        )
    figure.update_layout(plain_layout(
        margin={"l": 60, "r": 10, "t": 40, "b": 40}, barmode="group", bargap=0.3, bargroupgap=0.1, hoverlabel={"font": FONT},
        legend={"orientation": "h", "x": 0, "y": 1.2, "font": {**FONT, "color": MUTED}},
        xaxis={"fixedrange": True},
        yaxis={"tickformat": ".0%", "range": [0, 1], "gridcolor": GRID, "zeroline": False, "fixedrange": True},
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


def surface_arc(source: Point, target: Point, step: float = ARC_STEP, lift: float = ARC_LIFT) -> tuple[list, list, list]:
    """The points (x, y, z lists) of the great-circle arc between two points, sampled every `step` radians, a `lift`
    share above the surface (the radius runs from the source's to the target's). Two points in the same direction,
    or one at the centre, give a straight line; opposite points take the half circle through a pole (or through the
    x axis, from a pole)."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    radius_a, radius_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    straight = ([float(a[0]), float(b[0])], [float(a[1]), float(b[1])], [float(a[2]), float(b[2])])
    if radius_a < 1e-9 or radius_b < 1e-9:
        return straight
    unit_a, unit_b = a / radius_a, b / radius_b
    angle = math.acos(max(-1.0, min(1.0, float(unit_a @ unit_b))))
    if angle < 1e-9:
        return straight
    # The great circle's second direction, at right angles to the source's in the plane of the two points; for
    # opposite points the plane is any, here the one through a pole
    across = unit_b - unit_a * float(unit_a @ unit_b)
    if np.linalg.norm(across) < 1e-9:
        pivot = np.array([0.0, 0.0, 1.0]) if abs(unit_a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        across = pivot - unit_a * float(unit_a @ pivot)
    across /= np.linalg.norm(across)
    steps = max(1, math.ceil(angle / step))
    shares = np.linspace(0.0, 1.0, steps + 1)
    thetas = shares * angle
    radii = (radius_a + (radius_b - radius_a) * shares) * lift
    points = (np.cos(thetas)[:, None] * unit_a + np.sin(thetas)[:, None] * across) * radii[:, None]
    return points[:, 0].tolist(), points[:, 1].tolist(), points[:, 2].tolist()


def links_trace(points: Mapping, pairs, along_surface: bool, name: str = "Connections", color: str = GOLD,
                dash: str = "dash", width: float = LINK_WIDTH, opacity: float = 1.0, legend: bool = False,
                group: Optional[Mapping] = None) -> Optional[go.Scatter3d]:
    """Pairs of drawn dots as one trace (None between two lines), straight or along the sphere's surface: by default
    the model's connections, gold and dashed; the page's S's neighbour pairs solid on the blue ramp, an entry of the
    legend by their name in the layer's group (`group`: plotly's legendgroup keys). None without a pair of two drawn
    dots."""
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
        hoverinfo="skip", showlegend=legend, name=name, **(group or {}),
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

# --- the dendrogram ---------------------------------------------------------------------------------------------


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= LABEL_LENGTH else text[: LABEL_LENGTH - 1] + "…"


def cut_height(merges: np.ndarray, kept: int) -> float:
    """Where the cut-off line crosses the tree: halfway between the last kept merge and the next one; under the
    first merge with none kept; a little above the top with all kept."""
    heights = [float(height) for height in np.asarray(merges)[:, 2]]
    if kept <= 0:
        return heights[0] / 2
    if kept >= len(heights):
        return heights[-1] * (1 + CUT_LIFT) if heights[-1] > 0 else CUT_LIFT
    return (heights[kept - 1] + heights[kept]) / 2


def dendrogram_figure(
    merges: np.ndarray, ids: Sequence[str], label_of: Mapping[str, str], kept: Optional[int] = None,
    hover_of: Optional[Mapping[str, str]] = None, cut_label: str = "", max_labels: int = LEAF_LABELS,
) -> go.Figure:
    """The tree of a clustering: `merges` is scipy's linkage of the rows that `ids` name (at least two), laid out
    by scipy's own coordinates (U-shaped lines, the leaves left to right in tree order), the merge distances as
    heights. The leaves are labelled by `label_of` (a dot's id without it) up to `max_labels` leaves, beyond that
    by the hover alone; the hover is `hover_of` (a topic's label with its hash), by default the label. With `kept`
    (the merges below the cut-off preset, models.clustering.preset_rank) a dashed line crosses the tree above the
    last kept merge (cut_height), `cut_label` written on it; the figure's meta carries its height. The axis
    "distance" and its few ticks stand in full colour at the figures' size."""
    tree = dendrogram(np.asarray(merges, dtype=float), no_plot=True)
    line_x: list = []
    line_y: list = []
    for xs, ys in zip(tree["icoord"], tree["dcoord"]):
        line_x.extend(list(xs) + [None])
        line_y.extend(list(ys) + [None])
    leaf_ids = [ids[leaf] for leaf in tree["leaves"]]
    leaf_x = [5.0 + 10.0 * position for position in range(len(leaf_ids))]  # scipy puts the leaves at 5, 15, 25, ...
    labels = [label_of.get(dot, dot) for dot in leaf_ids]
    hovers = [(hover_of or label_of).get(dot, dot) for dot in leaf_ids]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": GOLD, "width": 2}, hoverinfo="skip", name="Merges"))
    figure.add_trace(
        go.Scatter(
            x=leaf_x, y=[0] * len(leaf_x), mode="markers", marker={"color": MUTED, "size": 9}, customdata=leaf_ids,
            text=hovers, hovertemplate="%{text}<extra></extra>", name="Leaves",
        )
    )
    meta = None
    if kept is not None:
        height = cut_height(merges, kept)
        figure.add_trace(
            go.Scatter(
                x=[0, 10.0 * len(leaf_ids)], y=[height, height], mode="lines",
                line={"color": CUT_COLOR, "width": 2, "dash": "dash"}, hoverinfo="skip", name="Cut-off",
            )
        )
        if cut_label:
            figure.add_annotation(x=1.0, xref="paper", y=height, text=cut_label, showarrow=False, xanchor="right", yanchor="bottom",
                                  xshift=-4, font={**FONT, "color": CUT_COLOR}, bgcolor="rgba(7, 24, 39, 0.75)")
        meta = {"cut": height, "kept": kept}
    labelled = len(leaf_ids) <= max_labels
    ticks = [_short(label) for label in labels] if labelled else []
    base, step = LABEL_MARGIN
    figure.update_layout(plain_layout(
        meta=meta, margin={"l": 70, "r": 10, "t": 10, "b": base + step * max((len(tick) for tick in ticks), default=0)},
        showlegend=False, hoverlabel={"font": FONT},
        xaxis={
            "tickvals": leaf_x if labelled else [], "ticktext": ticks, "tickangle": -60, "tickfont": FONT,
            "showgrid": False, "zeroline": False, "fixedrange": True,
        },
        yaxis={"title": {"text": "distance", "font": LABEL_FONT}, "tickfont": FONT, "nticks": 4, "rangemode": "tozero",
               "gridcolor": GRID, "zeroline": False, "fixedrange": True},
    ))
    return figure


# --- a card's neighbours -------------------------------------------------------------------------------------


def neighbours_figure(rows: Sequence[tuple[str, float, str]]) -> go.Figure:
    """A selected card's neighbours, best on top: a bar per neighbour, its length the score, its label the card's
    id and its colour where it lies (HERE, THIS_DECK, ELSEWHERE; PLACE_COLORS), each place an entry of the legend
    where a neighbour lies there."""
    figure = go.Figure()
    places = [place if place in PLACE_COLORS else ELSEWHERE for _id, _score, place in rows]
    for place in (HERE, THIS_DECK, ELSEWHERE):
        own = [(card_id, score, where) for (card_id, score, _place), where in zip(rows, places) if where == place]
        if not own:
            continue
        figure.add_trace(go.Bar(
            x=[score for _id, score, _where in own], y=[card_id for card_id, _score, _where in own], orientation="h",
            marker={"color": PLACE_COLORS[place], "cornerradius": 4}, name=place,
            text=[f"{score:.2f}" for _id, score, _where in own], textposition="outside", textfont={**FONT, "color": INK},
            customdata=[where for _id, _score, where in own], hovertemplate="%{y} · %{x:.2f} · %{customdata}<extra></extra>",
            cliponaxis=False,
        ))
    figure.update_layout(plain_layout(
        margin={"l": 90, "r": 60, "t": 40, "b": 50}, barmode="overlay", bargap=0.3, hoverlabel={"font": FONT},
        legend={"orientation": "h", "x": 0, "y": 1.18, "font": {**FONT, "color": MUTED}},
        xaxis={"title": {"text": "cosine", "font": LABEL_FONT}, "range": [0, 1], "gridcolor": GRID, "zeroline": False, "fixedrange": True, "tickfont": FONT},
        yaxis={"categoryorder": "array", "categoryarray": [card_id for card_id, _score, _where in reversed(rows)], "tickfont": FONT, "fixedrange": True},
    ))
    return figure
