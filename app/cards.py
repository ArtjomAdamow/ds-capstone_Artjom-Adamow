"""The card record the models read: the seven fields of the viewer's Flashcard (project.models) and a
quiz's answer options (the option texts, one feedback per option, the indices of the correct
ones), which the viewer's parser keeps apart from the card (its Quiz). The glue builds the records from the viewer's
cards with their quiz (glue.course.load_course), the course table carries the options in three columns
(course_table), so the models read one record in both modes and need no import of the viewer.

The stacks are the parser's: 1 a flashcard of the lecture notebooks, 2 an unrated quiz, 3 a rated quiz (with a
difficulty), 4 a flashcard of the quiz file. The id is the comb_id: the stack, the deck id and the card id.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

STACK_PLAIN = 1
STACK_QUIZ = 2
STACK_DIFFICULTY = 3
STACK_SOURCE = 4


class Card(Protocol):
    """What the models read of a card: CardRecord, in both modes."""

    deck: str
    question: str
    answer: str
    stack: int
    difficulty: Optional[str]
    explain: Optional[str]
    options: tuple[str, ...]  # a quiz's answer options in file order; empty for a card without
    answers: tuple[str, ...]  # one feedback text per option
    correct: tuple[int, ...]  # the indices of the correct options

    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class CardRecord:
    """A card of the helper's own, in the order of the viewer's Flashcard fields, with the quiz's options after them."""

    comb_id: str
    deck: str
    question: str
    answer: str
    stack: int = STACK_PLAIN
    difficulty: Optional[str] = None
    explain: Optional[str] = None
    options: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    correct: tuple[int, ...] = ()

    @property
    def id(self) -> str:
        return self.comb_id
