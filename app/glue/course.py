"""The course file as the models' input, read with the viewer's parser: every card of the course as the models'
record (the viewer's Flashcard with its quiz's answer options, which the parser keeps apart from the card), the deck
ids by deck name (the decks state's dot ids) and the scripted topics (TopicSeed, the input of T1, with the headings
and sources the parser does not keep, read from the course file with the json module as the export reads them). The
course file is version 3 (data/topics.json); the parser refuses anything else.

The capstone's standalone page reads the course table instead (app.course_table), written by the same parser
version through modules/helper/export_course.py, so the ids and the records agree with the viewer's.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from project.content.parser import CardsFile, Quiz, read_decks

from ..cards import CardRecord
from ..course_table import topic_texts
from ..models.topic_model import TopicSeed


@dataclass(frozen=True)
class Course:
    """What the models read of a course file."""

    cards: list[CardRecord]  # every card, in file order
    deck_ids: dict[str, str]  # deck name -> deck id
    seeds: list[TopicSeed]  # the scripted topics, in deck name order and file order


def record_of(card, quiz: Optional[Quiz]) -> CardRecord:
    """The models' record of a viewer's card: its seven fields and, for a quiz, its options with their feedback and
    the indices of the correct ones."""
    return CardRecord(
        card.id, card.deck, card.question, card.answer, card.stack, card.difficulty, card.explain,
        options=tuple(quiz.options) if quiz else (), answers=tuple(quiz.answers) if quiz else (),
        correct=tuple(quiz.correct) if quiz else (),
    )


def scripted_topics(cards_file: CardsFile) -> list[TopicSeed]:
    """The scripted topics of every deck of the course file (version 3), in deck name order and file order, with
    their headings and sources."""
    cards = cards_file.load_decks()
    texts = topic_texts(cards_file.path)
    seeds = []
    for deck in cards_file.deck_names:
        members = [card for card in cards if card.deck == deck]
        for topic in cards_file.topics(deck):
            ids = tuple(card.id for card in members if cards_file.topic_of(card.id) == topic.topic_id)
            seeds.append(TopicSeed(deck, topic.topic_id, topic.name, topic.rest, ids, *texts[(deck, topic.topic_id)]))
    return seeds


def deck_ids_of(path: Path) -> dict[str, str]:
    """The deck ids of the course file by deck name."""
    return {deck["name"]: deck["id"] for deck in read_decks(Path(path))}


def load_course(path: Path) -> Course:
    """The course file read once for the models: the records with their quiz's options."""
    path = Path(path)
    cards_file = CardsFile(path)
    cards = [record_of(card, cards_file.quiz(card.id)) for card in cards_file.load_decks()]
    return Course(cards, deck_ids_of(path), scripted_topics(cards_file))
