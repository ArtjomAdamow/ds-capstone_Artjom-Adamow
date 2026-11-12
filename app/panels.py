"""The model panels' parts of the dashboard (Dash): the model letters with their names, the preset of the decks'
connections with its label, the two helpers of a figure slot (a graph without plotly's bar, a line instead of a
figure) and S2's variant label of the vectoriser the app runs, which its results rows are read by.
"""

from typing import Optional

import plotly.graph_objects as go
from dash import dcc, html

from models import embeddings

PANEL_NAMES = {"S": "Similarity", "C": "Course model", "T": "Topics", "D": "Difficulty", "P": "Prompt"}
PANEL_LETTERS = tuple(PANEL_NAMES)
DECK_PART = 5  # the decks' preset: parts of at most five decks
DECK_CUT = "groups of ≤ {size} decks"  # the preset's label on the decks' connections
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
