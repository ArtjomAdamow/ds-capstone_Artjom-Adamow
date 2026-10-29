"""The course file as the models' input, read with the viewer's parser: every card of the course (the viewer's
Flashcard fits the models' Card), the deck ids by deck name (the decks state's dot ids) and the scripted topics
(TopicSeed, the input of T1). The course file is version 3 (data/topics.json); the parser refuses anything else.

Until the data package of roadmap step 21 this is the only way the models get their cards; the capstone reads the
package's cards table instead, written by the same parser version, so the ids agree with the viewer's.
"""

from dataclasses import dataclass
from pathlib import Path

from project.content.parser import CardsFile, read_decks

from ..cards import Card
from ..models.topic_model import TopicSeed


@dataclass(frozen=True)
class Course:
    """What the models read of a course file."""

    cards: list[Card]  # every card, in file order
    deck_ids: dict[str, str]  # deck name -> deck id
    seeds: list[TopicSeed]  # the scripted topics, in deck name order and file order


def scripted_topics(cards_file: CardsFile) -> list[TopicSeed]:
    """The scripted topics of every deck of the course file (version 3), in deck name order and file order."""
    cards = cards_file.load_decks()
    seeds = []
    for deck in cards_file.deck_names:
        members = [card for card in cards if card.deck == deck]
        for topic in cards_file.topics(deck):
            ids = tuple(card.id for card in members if cards_file.topic_of(card.id) == topic.topic_id)
            seeds.append(TopicSeed(deck, topic.topic_id, topic.name, topic.rest, ids))
    return seeds


def deck_ids_of(path: Path) -> dict[str, str]:
    """The deck ids of the course file by deck name."""
    return {deck["name"]: deck["id"] for deck in read_decks(Path(path))}


def load_course(path: Path) -> Course:
    """The course file read once for the models."""
    path = Path(path)
    cards_file = CardsFile(path)
    return Course(cards_file.load_decks(), deck_ids_of(path), scripted_topics(cards_file))
