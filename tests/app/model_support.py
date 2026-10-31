"""The models' test fixtures, with nothing of the viewer: the helper's package on the path, the test course as a
table of card records with its deck ids and scripted topics (the same 13 cards the viewer's parser reads from
helper_support.COURSE; test_course.py checks that the two agree), the moving course, and a course for the
difficulty model D1. The model tests (test_similar, test_layout, test_clustering, test_topic_model,
test_course_model, test_difficulty_model) import this file only, so they run in an env without the viewer.
"""

import sys
from pathlib import Path

# The folder that holds the package app/ and start.py: modules/helper/ here, the root in the capstone, where these
# files are exported to tests/app/ (tools/export/export.md); the nearest one above this file
MODULE_DIR = next(folder for folder in Path(__file__).resolve().parents if (folder / "app" / "__init__.py").is_file())
# The real course table (git-ignored data, the tests that read it skip without it): the capstone's data/ next to the
# package, or this workspace's at the repository root
COURSE_TABLE_DIR = next(
    (folder for folder in (MODULE_DIR / "data" / "course", MODULE_DIR.parents[1] / "data" / "course") if folder.is_dir()),
    MODULE_DIR / "data" / "course",
)
sys.dont_write_bytecode = True
if not any(Path(entry or ".").resolve() == MODULE_DIR for entry in sys.path):
    sys.path.insert(0, str(MODULE_DIR))  # the helper's package, app, as the launcher puts it on the path

from app.cards import STACK_DIFFICULTY, STACK_PLAIN, STACK_QUIZ, STACK_SOURCE, CardRecord  # noqa: E402
from app.models.course_model import CourseModel  # noqa: E402
from app.models.topic_model import TopicSeed  # noqa: E402

# Three small decks with real words, so the TF-IDF baseline (terms in at least two cards) has something to match.
# The ids are comb_ids: stack + deck id + card id (stack 1 plain, 2 quiz, 3 quiz with difficulty, 4 a flashcard of
# the JS quiz file). T1 keeps every card in its scripted topic; one card sits between two topics (HIDDEN, between
# Lists and Dictionaries).
DECK_IDS = {"Alpha": "01", "Beta": "02", "Gamma": "03"}
CARDS = [
    CardRecord("101001", "Alpha", "What does a Python list comprehension do?",
               "A list comprehension builds a new list from an iterable in one expression.", STACK_PLAIN,
               explain="Wikipedia's article on list comprehension."),
    CardRecord("201002", "Alpha", "How do you slice a Python list?",
               "Slicing a list with start and stop indices returns a new list.", STACK_QUIZ),
    CardRecord("301003", "Alpha", "What is a Python dictionary?",
               "A dictionary maps keys to values; a lookup by key in a dictionary is fast.", STACK_DIFFICULTY, "easy"),
    CardRecord("101004", "Alpha", "How many items are in a list or a dictionary?",
               "The len function returns the number of items in a list and the number of keys in a dictionary.", STACK_PLAIN),
    CardRecord("401005", "Alpha", "How do you look up a value in a dictionary?",
               "A lookup by key in a dictionary returns the value stored for that key.", STACK_SOURCE),
    CardRecord("102001", "Beta", "What does git commit do?",
               "A git commit records a snapshot of the staged changes in the repository history."),
    CardRecord("102002", "Beta", "What does git branch create?",
               "A git branch is a pointer to a commit, a line of development in the repository."),
    CardRecord("102003", "Beta", "How do you merge a git branch?",
               "git merge joins the history of a branch into the current branch of the repository."),
    CardRecord("102004", "Beta", "What does git log show?",
               "git log lists the commits of the repository history, the newest commit first."),
    CardRecord("103001", "Gamma", "What is a confidence interval?",
               "A confidence interval is a range of values that likely contains the population mean."),
    CardRecord("103002", "Gamma", "What is a p value?",
               "A p value is the probability of a result at least as extreme under the null hypothesis."),
    CardRecord("103003", "Gamma", "What is the null hypothesis?",
               "The null hypothesis states no effect; a small p value rejects the null hypothesis."),
    CardRecord("103004", "Gamma", "What does a 95 percent confidence level mean?",
               "A 95 percent confidence interval contains the population mean in 95 percent of repeated samples."),
]
ALPHA_CARDS = ["101001", "201002", "301003", "101004", "401005"]
BETA_CARDS = ["102001", "102002", "102003", "102004"]
GAMMA_CARDS = ["103001", "103002", "103003", "103004"]
ALL_CARDS = ALPHA_CARDS + BETA_CARDS + GAMMA_CARDS
HIDDEN = "101004"  # the one card between two topics on this course (its margin is about 0.02; the app shows it)
# The scripted topics: (deck id, topic id) -> (name, rest flag); Alpha's "Dictionaries" has two cards and "Empty"
# none, Beta's "Branches" is the rest topic
TOPIC_NAMES = {
    ("01", "01"): ("Lists", False), ("01", "02"): ("Dictionaries", False), ("01", "03"): ("Empty", False),
    ("02", "01"): ("Commits", False), ("02", "02"): ("Branches", True),
    ("03", "01"): ("Intervals", False), ("03", "02"): ("Tests", False),
}
# (deck id, topic id) -> the scripted cards in deck order
SCRIPTED = {
    ("01", "01"): ["101001", "201002", "101004"], ("01", "02"): ["301003", "401005"], ("01", "03"): [],
    ("02", "01"): ["102001", "102004"], ("02", "02"): ["102002", "102003"],
    ("03", "01"): ["103001", "103004"], ("03", "02"): ["103002", "103003"],
}
NAME_OF = {deck_id: name for name, deck_id in DECK_IDS.items()}


def seeds_of(scripted: dict) -> list[TopicSeed]:
    """The scripted topics as the model's input, in deck name order and file order (as the glue reads them)."""
    return [
        TopicSeed(NAME_OF[deck_id], topic_id, *TOPIC_NAMES[(deck_id, topic_id)], tuple(scripted[(deck_id, topic_id)]))
        for deck_id, topic_id in sorted(TOPIC_NAMES, key=lambda key: (NAME_OF[key[0]], key[1]))
    ]


SEEDS = seeds_of(SCRIPTED)

MOVED = "103002"  # the card T1 moves on the moving course: the p value, scripted into Intervals, goes to Tests
MOVING_CARD = CardRecord("103005", "Gamma", "What does a small p value say about the null hypothesis?",
                         "A small p value is evidence against the null hypothesis, which is then rejected.")
MOVING_SCRIPTED = {
    **SCRIPTED, ("03", "01"): ["103001", "103002", "103004"], ("03", "02"): ["103003", "103005"],
}


def moving_cards() -> list[CardRecord]:
    """The test course with a card T1 moves: Gamma's p value is scripted into Intervals, and Tests gets a second card
    about p values, so the model takes it to Tests (the other cards keep their scripted topics)."""
    return CARDS + [MOVING_CARD]


def course_model(cards=None, seeds=None, deck_ids=DECK_IDS, vectoriser=None) -> CourseModel:
    """The course model fitted on the test course (its scripted topics as the input of T1), or on other cards."""
    return CourseModel(CARDS if cards is None else cards, deck_ids, SEEDS if seeds is None else seeds, vectoriser)


# --- a course for the difficulty model D1 -------------------------------------------------------------------

TOPICS = {"Alpha": ("python", "list"), "Beta": ("git", "branch"), "Gamma": ("statistics", "sample")}
TERMS = ["index", "slice", "loop", "value", "method", "module", "object", "string", "number", "error", "file", "class"]


def quiz(deck: str, number: int, level: str) -> tuple[str, str]:
    """A question and answer whose length and form follow the level, as in the course."""
    topic, thing = TOPICS[deck]
    term = TERMS[number % len(TERMS)]
    if level == "easy":
        return f"What is a {term} in {topic}?", f"A {term} is part of a {thing} in {topic}."
    if level == "medium":
        return (
            f"Which {term} would you use for a {thing} in {topic} when it has {number + 3} parts, and why?",
            f"Use the {term}, because a {thing} with many parts in {topic} needs it to stay readable and correct.",
        )
    return (
        f"A team works with a {thing} in {topic} of {number * 100} rows, and the {term} fails after {number + 2} days; "
        f"which step would fix it, and why is it the best one in this scenario?",
        f"Check the {term} first: a {thing} in {topic} that grows fails at its {term}, so fixing the {term} keeps the "
        f"whole {thing} working, which is why it is the best step here.",
    )


def difficulty_course() -> list[CardRecord]:
    """Three decks, each with 12 rated quizzes listed easy → medium → hard (8 / 3 / 1, as in the course), a short and
    a long unrated quiz, three plain flashcards and one flashcard of the quiz file (stack 4)."""
    cards = []
    for deck_number, deck in enumerate(TOPICS, start=1):
        topic, thing = TOPICS[deck]
        levels = ["easy"] * 8 + ["medium"] * 3 + ["hard"]
        for number, level in enumerate(levels, start=1):
            question, answer = quiz(deck, number + deck_number, level)
            cards.append(CardRecord(f"3{deck_number:02d}{number:03d}", deck, question, answer, stack=STACK_DIFFICULTY, difficulty=level))
        cards.append(CardRecord(f"2{deck_number:02d}101", deck, *quiz(deck, 20 + deck_number, "easy"), stack=STACK_QUIZ))
        cards.append(CardRecord(f"2{deck_number:02d}102", deck, *quiz(deck, 21 + deck_number, "hard"), stack=STACK_QUIZ))
        for number in range(3):
            cards.append(CardRecord(f"1{deck_number:02d}2{number:02d}", deck, f"What does {topic} do with a {TERMS[number]}?",
                                    f"{topic} keeps the {TERMS[number]} of a {thing}.", stack=STACK_PLAIN))
        cards.append(CardRecord(f"4{deck_number:02d}301", deck, f"Which {thing} does {topic} start with?",
                                f"{topic} starts with an empty {thing}.", stack=STACK_SOURCE))
    return cards


def difficulty_report(predictions=None):
    """A D1 report measured on difficulty_course(), with other predictions if given (card id -> level)."""
    import dataclasses

    from app.models import similar
    from app.models.difficulty_model import fit_difficulty

    cards = difficulty_course()
    report = fit_difficulty(cards, similar.SimilarityIndex.fit(cards))
    if predictions is None:
        return report
    return dataclasses.replace(report, predictions=dict(predictions), unrated_ids=tuple(predictions))
