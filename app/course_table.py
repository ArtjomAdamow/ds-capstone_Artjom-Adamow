"""The course table: the models' input. Three CSV files in one folder (data/course/), the course file with its text
cleaned, read and written here with the csv module (no pandas):

* cards.csv (CARD_COLUMNS): every card of the course (the decks in name order, the cards in file
  order): the comb_id, the deck's id and name, the scripted topic's id, name and rest flag, the stack, the authored
  difficulty, the question, the answer and the explanation, cleaned, and a quiz's answer
  options, their feedback and the indices of the correct ones as JSON lists in three cells (empty for a card
  without options; the option texts carry markdown and commas);
* decks.csv (DECK_COLUMNS): every deck in deck id order with its description (a placeholder for now);
* topics.csv (TOPIC_COLUMNS): every deck's scripted topics in file order, the empty ones too, which cards.csv
  cannot carry: T1's seeds hold them (an emptied topic keeps its dot); with the course file's headings and sources
  of each topic, joined by " | " (TEXT_SEPARATOR; topic_texts reads them from the course file with the json
  module).

load_table gives what the models take (the CardRecords, the deck ids by name, the TopicSeeds in deck name order and
file order), plus the deck list. An empty difficulty or explanation reads as None.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .cards import CardRecord
from .models.topic_model import TopicSeed

CARDS_NAME = "cards.csv"
DECKS_NAME = "decks.csv"
TOPICS_NAME = "topics.csv"
CARD_COLUMNS = (
    "comb_id", "deck_id", "deck", "topic_id", "topic_name", "rest", "stack", "difficulty", "question", "answer", "explain",
    "options", "answers", "correct",
)
LIST_COLUMNS = ("options", "answers", "correct")  # JSON lists in the cell, empty for a card without options
DECK_COLUMNS = ("deck_id", "name", "description")
TOPIC_COLUMNS = ("deck_id", "topic_id", "name", "rest", "headings", "sources")
TEXT_SEPARATOR = " | "  # a topic's headings in one cell, and its sources (a source's own files are joined by "; ")
TRUE = "true"
FALSE = "false"


@dataclass(frozen=True)
class DeckRow:
    """A deck of the table: its id, its name and its description."""

    deck_id: str
    name: str
    description: str


@dataclass(frozen=True)
class CourseTable:
    """What the models read of the course table."""

    cards: list[CardRecord]  # every card, in the parser's order
    deck_ids: dict[str, str]  # deck name -> deck id
    seeds: list[TopicSeed]  # the scripted topics, in deck name order and file order
    decks: list[DeckRow]  # every deck, in deck id order


def table_files(folder: Path) -> tuple[Path, Path, Path]:
    """The three files of the table in `folder`."""
    folder = Path(folder)
    return folder / CARDS_NAME, folder / DECKS_NAME, folder / TOPICS_NAME


def missing_files(folder: Path) -> list[Path]:
    """The files of the table that are not in `folder`."""
    return [path for path in table_files(folder) if not path.is_file()]


def flag(value: bool) -> str:
    return TRUE if value else FALSE


def encode_list(values: Sequence) -> str:
    """A list cell: the values as a JSON list, an empty string for none."""
    values = list(values or ())
    return json.dumps(values, ensure_ascii=False) if values else ""


def decode_list(cell: str) -> tuple:
    """The values of a list cell (encode_list), an empty tuple for an empty cell."""
    return tuple(json.loads(cell)) if cell else ()


def join_texts(values: Sequence[str]) -> str:
    """A text cell: the headings or the sources joined by TEXT_SEPARATOR, an empty string for none."""
    return TEXT_SEPARATOR.join(values or ())


def split_texts(cell: str) -> tuple[str, ...]:
    """The headings or the sources of a text cell (join_texts), an empty tuple for an empty cell."""
    return tuple(cell.split(TEXT_SEPARATOR)) if cell else ()


def topic_texts(course_file: Path) -> dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Every topic's headings and sources in the course file (version 3), read with the json module: (deck name,
    topic id) -> (headings, sources); a missing list reads as empty."""
    data = json.loads(Path(course_file).read_text(encoding="utf-8"))
    return {
        (deck["name"], topic["topic_id"]): (tuple(topic.get("headings") or ()), tuple(topic.get("sources") or ()))
        for deck in data.get("decks", [])
        for topic in deck.get("topics", [])
    }


def _rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is no course table file: the column(s) {', '.join(missing)} are missing")
        return list(reader)


def _write(path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else row[column] for column in columns})
            count += 1
    return count


def write_table(
    folder: Path,
    cards: Iterable[Mapping[str, object]],
    decks: Iterable[Mapping[str, object]],
    topics: Iterable[Mapping[str, object]],
) -> tuple[int, int, int]:
    """Write the three files (rows by column name; None as empty, the rest flags as TRUE or FALSE); returns the
    number of cards, decks and topics written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cards_path, decks_path, topics_path = table_files(folder)
    return (
        _write(cards_path, CARD_COLUMNS, cards),
        _write(decks_path, DECK_COLUMNS, decks),
        _write(topics_path, TOPIC_COLUMNS, topics),
    )


def _optional(value: str):
    return value if value else None


def load_table(folder: Path) -> CourseTable:
    """The course table read once for the models and the page."""
    cards_path, decks_path, topics_path = table_files(folder)
    decks = [DeckRow(row["deck_id"], row["name"], row["description"]) for row in _rows(decks_path, DECK_COLUMNS)]
    decks.sort(key=lambda deck: deck.deck_id)
    card_rows = _rows(cards_path, CARD_COLUMNS)
    cards = [
        CardRecord(
            comb_id=row["comb_id"], deck=row["deck"], question=row["question"], answer=row["answer"], stack=int(row["stack"]),
            difficulty=_optional(row["difficulty"]), explain=_optional(row["explain"]),
            options=decode_list(row["options"]), answers=decode_list(row["answers"]),
            correct=tuple(int(index) for index in decode_list(row["correct"])),
        )
        for row in card_rows
    ]
    deck_name = {deck.deck_id: deck.name for deck in decks}
    topics = _rows(topics_path, TOPIC_COLUMNS)
    seeds = []
    for name in sorted(deck_name.values()):
        members = [row for row in card_rows if row["deck"] == name]
        for topic in topics:
            if deck_name.get(topic["deck_id"]) != name:
                continue
            ids = tuple(row["comb_id"] for row in members if row["topic_id"] == topic["topic_id"])
            seeds.append(TopicSeed(
                name, topic["topic_id"], topic["name"], topic["rest"] == TRUE, ids,
                split_texts(topic["headings"]), split_texts(topic["sources"]),
            ))
    return CourseTable(cards, {deck.name: deck.deck_id for deck in decks}, seeds, decks)
