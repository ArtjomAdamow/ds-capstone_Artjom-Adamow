"""The two files of the helper's own next to the viewer's state file (data/state/ by start.py):

* The derived course file (DERIVED_NAME, write_derived_course): the raw course file written again with the cards'
  "topic" set by the topic model for every deck (since roadmap step 33 of the helper; until then for the decks the
  Helper button had given the model's topics), the same decks, topic ids and names, cards and ids, in the viewer's
  format, so its parser, TopicIndex, dialogs and state work unchanged. The viewer's compose reads it as its course
  file; the raw file is the models' input. Rewritten in the serving process at every start and never backed up.
* The model state (MODEL_STATE_NAME, write_model_state), the models' results for the whole course, one entry per
  model (version 2 since step 33):

      {"version": 2,
       "course_model": {"positions": {"decks": {deck id: [x, y, z]}, "cards": {card id: [x, y, z]}},
                        "links": {"decks": [[source, target], ...], "topics": {deck id: [[topic id, topic id], ...]},
                                  "cards": [[source, target], ...]},
                        "presets": {"decks": share, "topics": [[deck id, topic id, share], ...]}},
       "difficulty_model": {"levels": {card id: level}}}

  the course model's positions, its connections in merge order (the decks', every deck's topics', every topic's
  cards') and the presets of the cut-off line; D1's predicted level of every quiz it rated, from which the groups are
  filled. Further models get entries of their own. Written at every start after the fit, the record of what the
  viewer got (button.Helper.model_state, the hand-over); read by nothing, holding nothing of the learner's, so it is
  neither written after a save nor kept by the backup question. The generated groups and connections are never in
  the viewer's state file, which holds the user's groups with their kinds and the user's connections.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence

from project.content.parser import card_stack, comb_id

from ..models.topic_model import MODEL_NAME

DERIVED_NAME = "helper_topics.json"  # the model's topics as a course file for the viewer
MODEL_STATE_NAME = "model_state.json"  # one entry per model (next to the derived file and the state file)
MODEL_STATE_VERSION = 2
DIFFICULTY_MODEL = "difficulty_model"  # the entry of D1: its predicted levels
COURSE_MODEL = "course_model"  # the entry of the course model: its positions, connections and presets


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_derived_course(
    source: Path,
    target: Path,
    topic_of: Mapping[str, str],
    hidden: Optional[int] = None,
    decks: Optional[Sequence[str]] = None,
) -> Path:
    """Write the course file again with the cards' "topic" set by the model: the same decks, topics (ids and
    names), cards and ids, so the viewer's parser reads it as it reads the raw file. A card `topic_of` leaves out
    (a deck without the model's topics, or a card skipped by the parser for its empty question) keeps its topic. A
    "helper" entry names the model, the source file, what changed and, if given, the decks with the model's
    topics (deck ids); the file is written atomically. Returns `target`."""
    source, target = Path(source), Path(target)
    data = json.loads(source.read_text(encoding="utf-8"))
    moved = 0
    for deck in data["decks"]:
        for card in deck["cards"]:
            topic = topic_of.get(comb_id(card_stack(card), deck["id"], card["card_id"]))
            if topic is not None and topic != card.get("topic"):
                card["topic"] = topic
                moved += 1
    data["helper"] = {
        "model": MODEL_NAME, "source": source.name, "moved": moved,
        **({} if hidden is None else {"hidden": hidden}), **({} if decks is None else {"decks": list(decks)}),
    }
    _write_atomically(target, json.dumps(data, indent=2))
    return target


def write_model_state(path: Path, entries: Mapping[str, Mapping]) -> None:
    """Store one mapping per model, atomically, like the viewer's state file."""
    text = json.dumps({"version": MODEL_STATE_VERSION, **{str(name): dict(entry) for name, entry in entries.items()}}, indent=2)
    _write_atomically(Path(path), text)
