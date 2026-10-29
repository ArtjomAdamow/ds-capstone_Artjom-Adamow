"""The two files of the helper's own next to the viewer's state file (the launcher's temp/):

* The derived course file (DERIVED_NAME, write_derived_course): the raw course file written again with the cards'
  "topic" set by the topic model for the decks the Helper button has given the model's topics, the same decks,
  topic ids and names, cards and ids, in the viewer's format, so its parser, TopicIndex, dialogs and state work
  unchanged. The viewer's compose reads it as its course file; the raw file is the models' input. Rewritten in the
  serving process at every start and never backed up.
* The model state (MODEL_STATE_NAME, read_model_state, write_model_state; 2026-10-28, roadmap step 21.2.1, until
  then helper_model_state.json with the views of the user's groups), one entry per model:

      {"version": 1, "topic_model": {"decks": [...], "groups": []}, "difficulty_model": {"groups": [...]},
       "course_model": {"links": {"decks": [[source, target], ...], "cards": [...]}}}

  the deck ids with the model's topics, each model's generated groups with their members (CardGroup records; the
  topic model has none yet), and the course model's connections by state (2026-10-29, roadmap step 22): the decks
  state's and the card state's, each in the live list's order (merge order per scope). Further models get entries
  of their own. The generated groups and connections are never in the viewer's state file, which holds the user's
  groups with their kinds and the user's connections. Read at a start with a stored card state, so a restart
  rebuilds the same derived file, presets, groups and connections; ignored and overwritten at a fresh card state.
  Written after every save of the viewer's state and with the derived file.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence

from project.content.parser import card_stack, comb_id

from ..models.topic_model import MODEL_NAME

DERIVED_NAME = "helper_topics.json"  # the model's topics as a course file for the viewer (temp/)
MODEL_STATE_NAME = "model_state.json"  # one entry per model (temp/, next to the derived file and the state file)
MODEL_STATE_VERSION = 1
TOPIC_MODEL = "topic_model"  # the model state's entry of T1: the decks with the model's topics
DIFFICULTY_MODEL = "difficulty_model"  # the entry of D1: its generated groups
COURSE_MODEL = "course_model"  # the entry of the course model: its connections by state
LINK_STATES = ("decks", "cards")  # the course model's connections: the decks state's, the card state's


def stored_links(entry) -> dict[str, list[tuple[str, str]]]:
    """The course model's connections as stored, by state (LINK_STATES); a state without a readable list has none,
    and an entry that is not a pair of two ids is dropped."""
    links = entry.get("links") if isinstance(entry, Mapping) else None
    links = links if isinstance(links, Mapping) else {}
    pairs = {}
    for name in LINK_STATES:
        items = links.get(name)
        pairs[name] = [
            (item[0], item[1]) for item in (items if isinstance(items, list) else [])
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(end, str) for end in item)
        ]
    return pairs


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


def read_model_state(path: Path) -> dict[str, dict]:
    """One mapping per model, as stored; none without a readable file of this version (the maker of each entry
    checks its contents)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != MODEL_STATE_VERSION:
        return {}
    return {str(name): entry for name, entry in payload.items() if name != "version" and isinstance(entry, dict)}


def write_model_state(path: Path, entries: Mapping[str, Mapping]) -> None:
    """Store one mapping per model, atomically, like the viewer's state file."""
    text = json.dumps({"version": MODEL_STATE_VERSION, **{str(name): dict(entry) for name, entry in entries.items()}}, indent=2)
    _write_atomically(Path(path), text)
