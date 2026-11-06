"""The related cards' order and labels of the dashboard (app.dashboard).

* `rank_by_confidence`: the neighbours in the order the learner should take them: the unread cards first, among
  them the quizzes D1 predicted in D1's score order, the easiest first, in the places they hold, the other unread
  ones in the given order; then the read ones by their confidence ascending, the lowest ("review") first. The
  dashboard has no confidence: it passes none, so the read ones keep the given order.
* `shorten`: a card's question as an entry's label, the markdown marks dropped, cut to LABEL_LENGTH characters.
"""

from typing import Mapping, Optional, Sequence

LABEL_LENGTH = 90


def rank_by_confidence(
    ids: Sequence[str], read: Mapping[str, bool], confidence: Mapping[str, int], scores: Optional[Mapping[str, float]] = None,
) -> list[str]:
    """The cards in the order the learner should take them: the unread ones first (`read` by card id, False
    without an entry), then the read ones by their confidence ascending (`confidence` by card id, the lowest, "review",
    first; a card without one counts as 1), ties in the given order. Among the unread cards the ones with a score
    (`scores`: D1's, the lower the easier) take their places in score order; the others keep the given order."""
    scores = scores or {}
    unread = [card_id for card_id in ids if not read.get(card_id, False)]
    scored = sorted((card_id for card_id in unread if card_id in scores), key=lambda card_id: (scores[card_id], unread.index(card_id)))
    scored_places = iter(scored)
    ordered = [next(scored_places) if card_id in scores else card_id for card_id in unread]
    done = sorted((card_id for card_id in ids if read.get(card_id, False)), key=lambda card_id: (confidence.get(card_id, 1), ids.index(card_id)))
    return ordered + done


def shorten(text: str, length: int = LABEL_LENGTH) -> str:
    """The text in one line without markdown marks, cut to `length` characters with an ellipsis."""
    text = " ".join(text.replace("`", "").replace("**", "").split())
    return text if len(text) <= length else text[: length - 1] + "…"
