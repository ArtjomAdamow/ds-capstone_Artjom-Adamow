"""The card record the models read: a card's seven fields (the id, the deck, the question, the answer, the stack,
the authored difficulty and the explanation) and a quiz's answer options (the option texts, one feedback per
option, the indices of the correct ones). The course table carries the options in three columns (course_table).

The stacks: 1 a flashcard of Q_and_A.json, 2 an unrated quiz, 3 a rated quiz (with a
difficulty), 4 a flashcard of the quiz file. The id is the comb_id: the stack, the deck id and the card id.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

STACK_PLAIN = 1
STACK_QUIZ = 2
STACK_DIFFICULTY = 3
STACK_SOURCE = 4


class Card(Protocol):
    """What the models read of a card: CardRecord."""

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
    """A card as the models read it: its seven fields, with the quiz's options after them."""

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
