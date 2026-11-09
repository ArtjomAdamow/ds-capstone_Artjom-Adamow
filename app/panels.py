"""The model panels' parts the dashboard and the standalone page share (Dash; nothing of the viewer): the model
letters with their names, the views with the cut labels of the connections' preset, the two helpers of a figure
slot (a graph without plotly's bar, a line instead of a figure) and the labels the results rows are read by (the
decks view's preset, S2's and D2's variant labels of the models the app runs).
"""

from typing import Optional

import plotly.graph_objects as go
from dash import dcc, html

from .models import embeddings, zero_shot

CARDS_VIEW = "cards"
TOPICS_VIEW = "topics"
DECKS_VIEW = "decks"
PANEL_NAMES = {"S": "Similarity", "C": "Course model", "T": "Topics", "D": "Difficulty", "P": "Prompt"}
PANEL_LETTERS = tuple(PANEL_NAMES)
CUT_LABELS = {DECKS_VIEW: "groups of ≤ {size} decks", CARDS_VIEW: "groups of ≤ {size} cards", TOPICS_VIEW: "the full tree"}
DECK_PART = 5  # the decks view's preset: parts of at most five decks (as the viewer's Helper presets them)
ENCODER_SHORT = {embeddings.MINILM: "MiniLM", embeddings.MPNET: "mpnet"}  # as evaluate.py names S2's rows


def graph(figure: go.Figure, class_name: str) -> dcc.Graph:
    """A graph of the page, without plotly's bar, following its slot's size."""
    return dcc.Graph(figure=figure, config={"displayModeBar": False, "responsive": True}, responsive=True, className=class_name,
                     style={"height": "100%", "width": "100%"})


def empty_slot(text: str) -> html.Div:
    """The figure slot, or a part of Details, with a line instead of a figure."""
    return html.Div(html.P(text, className="empty-state"), className="slot-note")


def s2_label(name: str = embeddings.APP_VECTORISER) -> Optional[str]:
    """S2's variant label in results.csv of the vectoriser the app runs (APP_VECTORISER), as evaluate.py names its
    rows: "MiniLM", "hybrid: chosen + MiniLM"; None for TF-IDF."""
    if name == embeddings.TFIDF:
        return None
    encoder = name.removeprefix("hybrid-")
    short = ENCODER_SHORT.get(embeddings.ENCODERS.get(encoder, ""), encoder)
    return f"hybrid: chosen + {short}" if name.startswith("hybrid-") else short


def d2_label(model_name: str = zero_shot.NLI_MODEL) -> str:
    """D2's variant label in difficulty.csv of the app's D2: the criterion template on the model the app reads
    (NLI_MODEL), as evaluate.py names its rows ("criterion, large")."""
    return f"{zero_shot.CRITERION.name}, {zero_shot.size_of(model_name)}"
