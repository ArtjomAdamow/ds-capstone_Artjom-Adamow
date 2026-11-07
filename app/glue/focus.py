"""The margin slider (glue): T1's margins as the viewer's focus, through its `focus` hook
(project.services.actions.ModuleFocus; step 28 of the helper).

The slider under the card panel in the cards view, labelled "Margin", runs on a share scale: the share of the
course's cards in focus, from 100 % (every card, the threshold 0.0) down to 1 % (the eleven cards with the highest
margins on the course); the viewer keeps the value for the session, starting at 100 %, so nothing changes until the
slider moves. A value's threshold is the margin that keeps that share of the model's cards, the highest margins first
(models.topic_model.share_threshold; the scale and its reading: docs/models/topic.md, "The margin on a
share scale"). A card's score is its margin, for the cards of the decks with the model's topics alone (the Helper's
model_decks at the time of the draw), so a scripted deck's cards are never filtered; a card without a margin is never
filtered either. The viewer filters the cards view's shown cards by it before the cut-off line, with the readout of
the cards in focus and the threshold; nothing is stored.
"""

from typing import Optional

FOCUS_LABEL = "Margin"
MARKS = {1: "1 %", 25: "25 %", 50: "50 %", 75: "75 %", 100: "100 %"}  # the share of the course's cards in focus
START = 100  # every card in focus until the slider moves


def helper_focus(helper):
    """The viewer's focus over T1's margins: the share scale's marks, the start at 100 %, the threshold of a value
    (share_threshold over the model's margins) and a card's margin as its score where its deck has the model's
    topics (`helper` is the Helper: its model and its model_decks, read at every call). ModuleFocus is imported
    here, not with the module: a viewer older than sputnik 0.5.0 has none, and the composition calls this only
    where the viewer's compose takes the hook."""
    from project.services.actions import ModuleFocus

    topic_model = helper.model.topic_model

    def threshold_of(value: float) -> float:
        return topic_model.threshold_for_share(float(value) / 100)

    def score_of(card_id: str) -> Optional[float]:
        if helper.model.deck_of.get(card_id) not in helper.model_decks:
            return None
        return topic_model.margin_of.get(card_id)

    return ModuleFocus(FOCUS_LABEL, dict(MARKS), START, threshold_of, score_of)
