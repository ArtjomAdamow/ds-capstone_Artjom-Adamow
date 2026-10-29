"""The card record the models read: the seven fields of the viewer's Flashcard (project.models) as a protocol, so
the viewer's cards fit it as they are and the models need no import of the viewer; and a record of the helper's own
(CardRecord) for the test fixtures and, later, the data package's cards table (roadmap step 21, the table loader).

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
    """What the models read of a card: the viewer's Flashcard and CardRecord both fit."""

    deck: str
    question: str
    answer: str
    stack: int
    difficulty: Optional[str]
    explain: Optional[str]

    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class CardRecord:
    """A card of the helper's own, in the order of the viewer's Flashcard fields."""

    comb_id: str
    deck: str
    question: str
    answer: str
    stack: int = STACK_PLAIN
    difficulty: Optional[str] = None
    explain: Optional[str] = None

    @property
    def id(self) -> str:
        return self.comb_id
