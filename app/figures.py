"""The helper's plotly figures (plotly only: no Dash, nothing of the viewer), drawn by the glue's back of the card panel
(glue.difficulty_metrics) and by the standalone page (standalone) alike:

* the difficulty model D1 (models.difficulty_model.DifficultyReport): the line that says what was measured
  (metrics_subtitle), the score table's rows (score_header, score_rows), the confusion matrix (confusion_figure) and
  the shares of the levels (shares_figure);
* the sphere: its surface (sphere_surface) and the great-circle arcs along it (surface_arc), the cards view's
  connections;
* the dendrogram of a clustering (dendrogram_figure): scipy's linkage of the same vectors the connections come from
  (models.clustering), laid out by scipy's coordinates, the real merge distances as heights, with the cut-off line
  above the last kept merge.

Colours: the sequential ramp and the two series colours are the dataviz reference steps for a dark surface,
validated against the panel's background; the levels are named on the axes, so no colour carries them alone. The
theme's tokens (INK, MUTED, GOLD) are the viewer's values, written here anew.
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
FONT = {"color": INK, "family": "DM Mono, monospace", "size": 11}
TABLE_ROWS = (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced accuracy"), ("kappa", "weighted kappa"), ("accuracy", "accuracy"))
SURFACE_ALPHA = 0.2  # faint, so the cards and their connections stand out
SURFACE_STEPS = 24  # the sphere's mesh: this many steps around and half as many from pole to pole
ARC_STEP = math.radians(5)  # an arc is sampled every 5 degrees
ARC_LIFT = 1.01  # the arcs sit a hair above the surface, clear of the mesh's facets
LABEL_LENGTH = 18  # the dendrogram's leaf labels
LABEL_MARGIN = (20, 5)  # the space under the dendrogram: a base and a step per character of its longest label
CUT_LIFT = 0.04  # a cut-off line above the whole tree sits this share of its height above the top merge

Point = tuple[float, float, float]


# --- D1's metrics ---------------------------------------------------------------------------------------------


def number(value: float) -> str:
    return "–" if value != value else f"{value:.2f}"  # NaN -> dash


def spread(value: tuple[float, float]) -> str:
    mean, sd = value
    return f"{number(mean)} ± {number(sd)}"


def score_header(report: DifficultyReport) -> list[str]:
    """The score table's columns after the row labels: D1, then the baselines."""
    return ["D1"] + list(report.baselines)


def score_rows(report: DifficultyReport) -> list[tuple[str, str, list[str]]]:
    """The key scores, one row per metric: its label, D1 as mean ± sd, and the baselines' means."""
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
    """What was measured: the whole course, a deck's part, or a topic's part with the deck's metrics."""
    run = f"{report.folds} folds × {report.repeats} repeats"
    if report.scope is None:
        return (
            f"All decks: {report.rated} rated quizzes ({_levels(report)}), measured out of fold ({run}); levels by each "
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
            f"rated quizzes ({_levels(deck)}), out of fold in the course's cross-validation ({run}), by the deck's shares; "
            f"a topic's few rated quizzes are too few to measure on their own."
        )
    predicted = f"{report.unrated} unrated quizzes of this deck predicted." if report.unrated else "No unrated quiz in this deck."
    return (
        f"This deck: its {report.rated} rated quizzes ({_levels(report)}), out of fold in the course's cross-validation "
        f"({run}), by the deck's shares. {predicted} The decks view shows all decks."
    )


def confusion_figure(report: DifficultyReport) -> go.Figure:
    """Rows: the true level, columns: the predicted level, each row's shares (the diagonal is the recall)."""
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
                font={**FONT, "color": DARK_INK if value > 0.55 else INK},
            )
    figure.update_layout(
        height=210, margin={"l": 64, "r": 8, "t": 8, "b": 40}, paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)",
        font=FONT, dragmode=False,
        xaxis={"title": {"text": "predicted", "font": {"color": MUTED}}, "side": "bottom", "fixedrange": True},
        yaxis={"title": {"text": "true", "font": {"color": MUTED}}, "autorange": "reversed", "fixedrange": True},
    )
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
                text=[f"{value:.0%}" for value in values], textposition="outside", textfont={"color": INK},
                customdata=list(counts), hovertemplate=f"{name}, %{{x}}: %{{y:.0%}} (%{{customdata}} quizzes)<extra></extra>",
                cliponaxis=False,
            )
        )
    figure.update_layout(
        height=190, margin={"l": 40, "r": 8, "t": 30, "b": 28}, paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)",
        font=FONT, dragmode=False, barmode="group", bargap=0.3, bargroupgap=0.1,
        legend={"orientation": "h", "x": 0, "y": 1.18, "font": {"color": MUTED}},
        xaxis={"fixedrange": True},
        yaxis={"tickformat": ".0%", "range": [0, 1], "gridcolor": GRID, "zeroline": False, "fixedrange": True},
    )
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
    merges: np.ndarray, ids: Sequence[str], label_of: Mapping[str, str], kept: Optional[int] = None
) -> go.Figure:
    """The tree of a clustering: `merges` is scipy's linkage of the rows that `ids` name (at least two), laid out
    by scipy's own coordinates (U-shaped lines, the leaves left to right in tree order), the merge distances as
    heights. With `kept` (the merges below the cut-off preset, models.clustering.preset_rank) a dashed line crosses
    the tree above the last kept merge (cut_height); the figure's meta carries its height."""
    tree = dendrogram(np.asarray(merges, dtype=float), no_plot=True)
    line_x: list = []
    line_y: list = []
    for xs, ys in zip(tree["icoord"], tree["dcoord"]):
        line_x.extend(list(xs) + [None])
        line_y.extend(list(ys) + [None])
    leaf_ids = [ids[leaf] for leaf in tree["leaves"]]
    leaf_x = [5.0 + 10.0 * position for position in range(len(leaf_ids))]  # scipy puts the leaves at 5, 15, 25, ...
    labels = [label_of.get(dot, dot) for dot in leaf_ids]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": GOLD, "width": 2}, hoverinfo="skip", name="Merges"))
    figure.add_trace(
        go.Scatter(
            x=leaf_x, y=[0] * len(leaf_x), mode="markers", marker={"color": MUTED, "size": 7}, customdata=leaf_ids,
            text=labels, hovertemplate="%{text}<extra></extra>", name="Leaves",
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
        meta = {"cut": height, "kept": kept}
    ticks = [_short(label) for label in labels]
    base, step = LABEL_MARGIN
    figure.update_layout(
        height=320, meta=meta, margin={"l": 44, "r": 10, "t": 10, "b": base + step * max(len(tick) for tick in ticks)},
        paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": INK, "family": "DM Mono, monospace", "size": 12}, showlegend=False, dragmode=False,
        xaxis={
            "tickvals": leaf_x, "ticktext": ticks, "tickangle": -60,
            "showgrid": False, "zeroline": False, "fixedrange": True,
        },
        yaxis={"title": {"text": "distance"}, "rangemode": "tozero", "gridcolor": GRID, "zeroline": False, "fixedrange": True},
    )
    return figure
