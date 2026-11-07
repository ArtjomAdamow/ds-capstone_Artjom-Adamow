"""The episodes' log (glue): every ended episode of the Ask dialog or the card section as one row of `episodes.csv`
next to the model state (data/state/ by start.py), through the viewer's `after_episode` hook
(project.services.episode; step 26 of the helper).

The viewer grades what the learner did after a navigation and hands over the record: the source (`prompt` or
`section`), the request's text or the card the section entry was clicked on, the rank of the entry taken (0 when the
dialog's answer navigated by itself, 1 to 3 for its entries picked after an answer that did not; the section entry's
slot, 0 the next card), the target it landed in by the viewer's three ids, the grade (A, B or C), what ended it, the
start time, the seconds and the trail. The glue writes it in the model's terms:

* the target's topic dot id turned into T1's key through the router's topic index: the deck's name and the topic id
  (TopicInfo.deck_name, topic_id); a dot the index no longer holds (the Helper gave the deck new topics while the
  episode was open) keeps the deck's name by its id and an empty topic;
* the decision read off the target, where the learner landed: a card given means a card selected ("card selected"),
  none the topic entered ("topic entered"), the model's words (models.prompt_model); with the rank, whether P1's
  answer took the learner there (0) or the learner picked it among the nearest three;
* the start as local time to the second, the seconds as given, the trail as a JSON list of [seconds, card id,
  turned] steps.

The header is written once, with the first row of a new file. A row that cannot be written (the file open and locked
in a spreadsheet, say) is lost with one printed line: a log never stops the viewer's callback. A log, not state: rows are appended, never rewritten,
and survive a fresh card state, which forgets model_state.json; runtime data of the launcher's folder, never tracked
and never carried by a train. Its readers come later: the online measure of the related cards and of the dialog, and
the D2 upgrade's signal from the trail (docs/draft/models/difficulty.md, "Later: the D2 upgrade"), which caps
the seconds of a learner away from the keyboard.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..models.prompt_model import ENTERED, SELECTED

EPISODES_NAME = "episodes.csv"  # next to the model state
EPISODE_COLUMNS = ("started", "source", "request", "rank", "decision", "deck", "topic", "card", "grade", "end", "seconds", "trail")


def episode_row(state, record) -> dict:
    """The record as a row of the log, in the model's keys (the module docstring); `state` is the viewer's router."""
    deck_id, dot_id, card_id = record.target
    topic = state.topics.by_dot_id.get(dot_id)
    deck = state.decks.by_id.get(deck_id)
    return {
        "started": datetime.fromtimestamp(record.started).isoformat(timespec="seconds"),
        "source": record.source,
        "request": record.request,
        "rank": record.rank,
        "decision": SELECTED if card_id else ENTERED,
        "deck": topic.deck_name if topic is not None else (deck.name if deck is not None else deck_id),
        "topic": topic.topic_id if topic is not None else "",
        "card": card_id or "",
        "grade": record.grade,
        "end": record.end,
        "seconds": record.seconds,
        "trail": json.dumps([[step.seconds, step.card, step.turned] for step in record.trail]),
    }


def append_episode(path: Path, row: dict) -> None:
    """One row at the end of the log, the header first where the file is new or empty."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def helper_episodes(path: Path) -> Callable:
    """The viewer's `after_episode(state, record)`: the ended episode as a row of the log at `path`; a row the file
    refuses is lost with one printed line."""

    def after_episode(state, record) -> None:
        try:
            append_episode(path, episode_row(state, record))
        except OSError as error:
            print(f"An episode was not logged ({path.name}: {error})")

    return after_episode
