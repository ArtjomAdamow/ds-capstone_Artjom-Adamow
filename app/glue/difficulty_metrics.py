"""The back of the card panel and the window under it (glue over the viewer's overview_back and overview_window
hooks): how well the difficulty model D1 does (models.difficulty_model.DifficultyReport), shown while nothing is
selected after Enter or a click on the panel. The decks view shows the whole course; the topics view shows the
entered deck's part only (its rated quizzes, out of fold under the course-wide model, and its predicted quizzes);
the cards view shows the entered topic's counts and predictions with the deck's metrics (about two rated quizzes
per topic are too few to measure).

* The back (metrics_panel): a table of the key scores, D1 as mean ± sd over the repeats, next to the two baselines
  (always "easy", and the similarity baseline as a classifier); it is also the table view of the numbers.
* The window under the card's window (metrics_window; app roadmap step 25): the confusion matrix of the rated
  quizzes, out of fold (rows are the true level, columns the predicted one, each row's shares, so the diagonal is
  the recall of each level; one hue, dark for 0 and light for 1, on the dark panel) and the shares of the levels
  (the rated quizzes, training, next to the predicted unrated quizzes, test, no labels). A viewer without the
  window hook (sputnik 0.3.0, the capstone's) gets the figures on the back itself (`figures=True`).

The figures, the score table's rows and the subtitle are the helper's shared figures (app.figures), which the
standalone page draws too; this module wraps them in the viewer's panels (Dash).
"""

from typing import Optional

from dash import dcc, html

from ..figures import FONT, INK, MUTED, confusion_figure, metrics_subtitle, score_header, score_rows, shares_figure
from ..models.difficulty_model import DifficultyReport

BACK_HINT = "Enter or a click on this frame turns it back to the dendrogram."
WINDOW_NOTE = "The confusion matrix and the shares of the levels are in the window under this one."
CELL = {"padding": "2px 10px 2px 0", "textAlign": "right"}


def score_table(report: DifficultyReport) -> html.Table:
    """The key scores: D1 (mean ± sd) and the baselines (their means)."""
    header = html.Tr(
        [html.Th("", style=CELL)] + [html.Th(name, style={**CELL, "color": MUTED, "fontWeight": "normal"}) for name in score_header(report)]
    )
    rows = [
        html.Tr(
            [html.Td(label, style={**CELL, "textAlign": "left", "color": MUTED}), html.Td(d1, style={**CELL, "color": INK})]
            + [html.Td(value, style={**CELL, "color": MUTED}) for value in baselines]
        )
        for label, d1, baselines in score_rows(report)
    ]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="difficulty-scores",
                      style={"borderCollapse": "collapse", "fontFamily": FONT["family"], "fontSize": "0.8rem", "margin": "6px 0"})


def _graph(figure, class_name: str) -> dcc.Graph:
    return dcc.Graph(figure=figure, config={"displayModeBar": False}, className=class_name)


def _part(report: DifficultyReport, deck: Optional[str], topic) -> tuple[DifficultyReport, str]:
    """The report's part for the view and the scope it names: the course, the deck, or the topic in the deck."""
    part = report.for_deck(deck) if deck is not None else report
    scope = f" · {deck}" if deck is not None else " · all decks"
    if topic is not None:
        part = part.for_topic(topic.name, topic.card_ids)
        scope = f" · {deck} › {topic.name}"
    return part, scope


def _figures(part: DifficultyReport, margin: str = "10px") -> list:
    """The confusion matrix and the shares of a measured part, with their headings."""
    return [
        html.Div("Rated quizzes: true and predicted level", className="card-label", style={"marginTop": margin}),
        _graph(confusion_figure(part), "difficulty-confusion"),
        html.Div("Shares of the levels", className="card-label", style={"marginTop": "6px"}),
        _graph(shares_figure(part), "difficulty-shares"),
    ]


def metrics_panel(report: DifficultyReport, deck: Optional[str] = None, topic=None, figures: bool = True) -> html.Div:
    """The back of the card panel: what D1 measured and predicted, for the whole course (the decks view), for the
    entered deck only (`deck`, the topics view) or for the entered topic (`topic`, a TopicInfo with `name` and
    `card_ids`, the cards view: the topic's counts and predictions with the deck's metrics); or why there is
    nothing to show. With `figures` the matrix and the shares are on the back too (a viewer without the window
    under the card's window); without, the back says where they are."""
    part, scope = _part(report, deck, topic)
    title = html.Div("Difficulty model D1" + scope, className="card-label")
    hint = html.Div(BACK_HINT, className="card-hint")
    style = {"cursor": "pointer"}
    if not part.measured:
        children = [title, html.P(part.note or "No difficulty model.", className="empty-state")]
        if part.predictions and figures:  # a deck without rated quizzes still has its predicted ones
            children.append(_graph(shares_figure(part), "difficulty-shares"))
        return html.Div(children + [hint], className="card-panel overview-panel difficulty-panel", style=style)
    return html.Div(
        [
            title,
            html.Div(metrics_subtitle(part), className="deck-description"),
            score_table(part),
            html.Div("With the deck shares every level keeps its true count, so macro-F1 equals the balanced accuracy.",
                     className="card-hint"),
        ]
        + (_figures(part) if figures else [html.Div(WINDOW_NOTE, className="card-hint")])
        + [hint],
        className="card-panel overview-panel difficulty-panel",
        style=style,
    )


def metrics_window(report: DifficultyReport, deck: Optional[str] = None, topic=None) -> html.Div:
    """The window under the card's window while the back is up: the confusion matrix and the shares of the levels
    of the view's part (`deck` and `topic` as for metrics_panel); a part without metrics its predicted shares only,
    a part without quizzes its note."""
    part, scope = _part(report, deck, topic)
    children = [html.Div("Difficulty model D1" + scope, className="card-label")]
    if part.measured:
        children += _figures(part, margin="6px")
    elif part.predictions:
        children += [html.Div("Shares of the levels", className="card-label", style={"marginTop": "6px"}), _graph(shares_figure(part), "difficulty-shares")]
    else:
        children.append(html.P(part.note or "No difficulty model.", className="empty-state"))
    return html.Div(children, className="card-panel overview-panel difficulty-window")
