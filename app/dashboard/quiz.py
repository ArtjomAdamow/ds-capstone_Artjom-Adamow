"""A quiz's judgement on the dashboard, as plain functions over the card record (models.cards: the options, their
feedback and the indices of the correct ones).

The rule: an answer is right only when exactly the correct options are checked, so a missing
correct option and an extra wrong one both make it wrong, and nothing checked is no answer (Confirm is disabled,
and the server checks it too). The back of a right answer is the card's feedback (its answer text, which names the
correct options) under "Correct answer"; the back of a wrong one lists every checked option in file order with its
own feedback under "Wrong answer", the correct ones among them marked.
"""

from typing import Iterable, NamedTuple

QUESTION_LABEL = "Question"
ANSWER_LABEL = "Answer"
CORRECT_LABEL = "Correct answer"
WRONG_LABEL = "Wrong answer"


class Feedback(NamedTuple):
    """One checked option on a wrong answer's back: its text, its own feedback, whether it is a correct one."""

    option: str
    answer: str
    correct: bool


def has_options(card) -> bool:
    """Whether the card is a quiz with answer options (a quiz without any turns like a flashcard)."""
    return bool(getattr(card, "options", ()))


def checked_indices(value, count: int) -> list[int]:
    """The checked options as indices in file order, from the checklist's value (ints, or their strings), the ones
    out of range dropped."""
    found: set[int] = set()
    for item in value or ():
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            found.add(index)
    return sorted(found)


def is_right(correct: Iterable[int], checked: Iterable[int]) -> bool:
    """True only when the checked options are exactly the correct ones; nothing checked is no answer."""
    checked = set(checked)
    return bool(checked) and checked == set(correct)


def confirm_disabled(checked) -> bool:
    """Confirm's disabled state: nothing checked."""
    return not list(checked or ())


def judge(card, value) -> tuple[bool, list[Feedback]]:
    """The judgement of the checked options (`value`, the checklist's): whether the answer is right, and for a wrong
    one the feedback blocks of the checked options in file order (empty for a right one: the back shows the card's
    answer)."""
    indices = checked_indices(value, len(card.options))
    if is_right(card.correct, indices):
        return True, []
    answers = list(card.answers) + [""] * (len(card.options) - len(card.answers))
    return False, [Feedback(card.options[index], answers[index], index in card.correct) for index in indices]
