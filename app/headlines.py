"""The section's numbers of the dashboard (nothing of Dash, as results.py): each
generation's two headline numbers from the results rows of its newest run, each with its contrast (the random
floor, a baseline, or the generation before), one plain sentence on what it measures, and a verdict with its
explanation in words.

* `headlines(results, level)`: the generation's two `Headline`s from the same newest run as its tables
  (results.Results.run); None without the file or the rows, the section then saying so. `outcomes(results)`: P1's
  outcomes counted on and off the course in its app row, the dashboard's grid.
* The judged measures name Claude as the judge ("judged precision@5", "Claude's levels"), never a hand.
* The verdict comes from a rule, never a hand-written word, by the number's kind (`Headline.verdict`):
  - INTERVAL: a paired difference by whether its interval holds 0: "better", "within noise", "worse";
  - KAPPA: Landis and Koch's words: slight, fair, moderate from 0.41, substantial, almost perfect (poor below 0);
  - FLOOR: by generic criteria on the distance above the floor, (number − floor) / (1 − floor), 0 at the floor and
    1 at the measure's top: below 0 "below the floor", below 0.25 "near the floor", 0.25 to 0.5 "above the floor",
    0.5 to 0.75 "well above the floor", from 0.75 "far above the floor"; the words say how far, not how good, since
    for same-deck precision and T1's agreement 1 is not the goal;
  - RATE: a judged share without a floor, by generic criteria on the share itself: below 50 % "weak", 50 to
    75 % "fair", 75 to 90 % "good", from 90 % "very good"; RATE_LOW, a share where lower is better, mirrored:
    up to 10 % "very good", up to 25 % "good", up to 50 % "fair", above it "weak".
  A generic verdict names its criteria: "well above the floor by generic criteria".
* The explanation in words (`Headline.explanation`): a floor's distance is never shown as a number; it is a share of
  the way, the floor and the top named in words with the floor's value ("about two thirds of the way from random
  positions (4 %) to all five nearest in the card's deck"); below 0.05 "at the level of random positions (4 %)",
  below 0 "below random positions (4 %)". A hit rate, an interval and a kappa need none beyond the sentence and
  the contrast: the number is its own plain reading.
* The format: shares as percentages, a difference in points; the newer generation's numbers as its gain over the
  one before (`Headline.gain`). `PLAIN` gives the chosen variant's plain words by its label in the file; a label
  without them shows as it is (`plain_words`).

The sources, per generation: S1's chosen row of results.csv (same-deck precision@5 against B0's mean, the floor;
judged precision@1, a hit rate); S2's app row (judged precision@5 with its paired difference to S1's chosen row, an
interval; judged precision@1 against S1's, a hit rate); C1's model row (the trustworthiness of the card layouts and of
the decks against the random row, floors); T1's row of scope all in topics.csv (agreement against chance, the floor,
1-NN beside it) and its row of the 25 % most confident (against all cards, in the floor's place); D1's chosen row
of difficulty.csv by the deck shares (the within-deck order on Claude's levels against a coin, the floor, with its
paired difference to the coin and the interval; weighted kappa against the baseline always easy); D2's app row (both
against D1's chosen row: an interval and a kappa); P1's app row of prompt.csv (the share served right, a hit rate,
and the share landed wrong, lower better, each against the TF-IDF card route with its paired difference and the
interval; none where the app runs on TF-IDF).
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from .results import CHOSEN, COURSE_SCOPE, FOCUS_SCOPE, course_row

FLOOR, INTERVAL, KAPPA, RATE, RATE_LOW = "floor", "interval", "kappa", "hit rate", "hit rate, lower better"
GENERIC = "by generic criteria"
NO_CONTRAST = "no contrast in this run"
COIN = 0.5  # the floor of the within-deck order: a coin
FOCUS_25 = f"{FOCUS_SCOPE}25 %"  # T1's row of the 25 % most confident cards
TFIDF_CARD = "TF-IDF: card"  # P1's card route on S1's vectors, the app row's contrast (prompt_model.variant_label)
# The judged measures' words: Claude is the judge of the relevance marks, the quizzes' levels and the requests
JUDGED_P1 = "How often the nearest card is related, as Claude judged it."
ORDER_LABEL = "within-deck order on Claude's levels"
# P1's outcomes counted on and off the course (evaluate.py's columns), the dashboard's grid
OUTCOME_COLUMNS = ("hand_course_right", "hand_course_wrong", "hand_course_missed", "hand_off_right", "hand_off_wrong")
# The floor's bands: the distance above it, (number − floor) / (1 − floor), and the words
FLOOR_BANDS = ((0.0, "below the floor"), (0.25, "near the floor"), (0.5, "above the floor"), (0.75, "well above the floor"))
FAR = "far above the floor"
RATE_BANDS = ((0.5, "weak"), (0.75, "fair"), (0.9, "good"))  # a hit rate: below each edge; from the last "very good"
VERY_GOOD = "very good"
LOW_BANDS = ((0.1, "very good"), (0.25, "good"), (0.5, "fair"))  # lower better: up to each edge; above the last "weak"
WEAK = "weak"
KAPPA_BANDS = ((0.0, "poor"), (0.21, "slight"), (0.41, "fair"), (0.61, "moderate"), (0.81, "substantial"))  # below each edge
ALMOST_PERFECT = "almost perfect"
# The share of the way, the nearest of these; from AT_FLOOR up ("at the level of" below it), "all the way" from ALL
SHARES = (
    (0.1, "a tenth"), (0.2, "a fifth"), (0.25, "a quarter"), (1 / 3, "a third"), (0.5, "half"), (2 / 3, "two thirds"),
    (0.75, "three quarters"), (0.8, "four fifths"), (0.9, "nine tenths"),
)
AT_FLOOR = 0.05
ALL = 0.95
# What the five models do, at most twelve words each: the section's headline
WHAT = {
    "S1": "Finds each card's five nearest cards by their words.",
    "S2": "Finds each card's five nearest cards by their meaning and their words.",
    "C1": "Places the decks, and each topic's cards, on the sphere.",
    "T1": "Sorts each card into the nearest of its deck's topics.",
    "D1": "Predicts the level of the unrated quizzes from the question's statistics.",
    "D2": "Reads the level off each quiz with a language model, nothing fitted.",
    "P1": "Takes a request in the learner's words to the nearest topic.",
}
NO_TAB = "Five models: S finds similar cards, C places them, T sorts them into topics, D rates the quizzes, P answers a request."
# The chosen variant's plain words by its label in the file
PLAIN = {
    "chosen: uni+bigrams, min_df=2, sublinear, explain": "words and word pairs, the explanation read too",
    "hybrid: chosen + MiniLM": "the sentence encoder MiniLM next to the words",
    "hybrid: chosen + mpnet": "the sentence encoder mpnet next to the words",
    "MiniLM": "the sentence encoder MiniLM alone",
    "mpnet": "the sentence encoder mpnet alone",
    "model": "the fitted layout",
    "T1": "the nearest topic centroid, leave-one-out",
    "text (chosen)": "the question's text statistics, labelled by the deck shares",
    "criterion, large": "the written criterion's three demands, read by the large model",
    "criterion, base": "the written criterion's three demands, read by the base model",
    "level names, large": "the level names, read by the large model",
    "level names, base": "the level names, read by the base model",
    "hybrid-minilm: card": "the nearest card on the encoder's vectors",
    "hybrid-mpnet: card": "the nearest card on the encoder's vectors",
    "TF-IDF: card": "the nearest card on the words",
    "char n-grams: card": "the nearest card on character n-grams",
}


def plain_words(label: str) -> str:
    """The variant's plain words, or its label as it is."""
    return PLAIN.get(label, label)


# --- the verdict rule -------------------------------------------------------------------------------------------


def interval_verdict(low: float, high: float) -> str:
    """A paired difference by its interval: "better" above 0, "worse" below it, else "within noise"."""
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "within noise"


def kappa_verdict(kappa: float) -> str:
    """Landis and Koch's words for a kappa."""
    return next((word for edge, word in KAPPA_BANDS if kappa < edge), ALMOST_PERFECT)


def distance(value: float, floor: float) -> float:
    """How far above the floor on the way to 1: (value − floor) / (1 − floor); 0 at the floor, 1 at the top (rounded
    to nine decimals, so a band's edge is its own)."""
    return round((value - floor) / (1 - floor), 9) if floor < 1 else 0.0


def floor_verdict(value: float, floor: float, name: str = "the floor") -> str:
    """The distance above the floor in words that say how far: "well above the floor by generic criteria"; `name`
    in the floor's place where the floor is no floor ("well above all cards")."""
    how_far = distance(value, floor)
    word = next((word for edge, word in FLOOR_BANDS if how_far < edge), FAR)
    return f"{word.replace('the floor', name)} {GENERIC}"


def rate_verdict(share: float, lower_better: bool = False) -> str:
    """A hit rate's band, by generic criteria: weak, fair, good, very good; mirrored where lower is better."""
    if lower_better:
        word = next((word for edge, word in LOW_BANDS if share <= edge), WEAK)
    else:
        word = next((word for edge, word in RATE_BANDS if share < edge), VERY_GOOD)
    return f"{word} {GENERIC}"


def share_words(how_far: float) -> str:
    """The share of the way, the nearest of SHARES; "all" from ALL."""
    if how_far >= ALL:
        return "all"
    return min(SHARES, key=lambda pair: abs(pair[0] - how_far))[1]


def floor_explanation(value: float, floor: float, floor_name: str, top_name: str, floor_shown: str) -> str:
    """The distance above the floor in words, never as a number: "about two thirds of the way from random positions
    (4 %) to all five nearest in the card's deck"; "at the level of random positions (4 %)" below AT_FLOOR, "below
    random positions (4 %)" below the floor."""
    how_far = distance(value, floor)
    if how_far < 0:
        return f"below {floor_name} ({floor_shown})"
    if how_far < AT_FLOOR:
        return f"at the level of {floor_name} ({floor_shown})"
    share = share_words(how_far)
    return f"{'all the way' if share == 'all' else f'about {share} of the way'} from {floor_name} ({floor_shown}) to {top_name}"


# --- the headline ---------------------------------------------------------------------------------------------


def shown(value: Optional[float], percent: bool) -> str:
    """A number as the section shows it: a share as a whole percentage ("69 %"), another number with two decimals;
    a dash for none."""
    if value is None:
        return "–"
    return f"{value * 100:.0f} %" if percent else f"{value:.2f}"


@dataclass(frozen=True)
class Headline:
    """One of a generation's two numbers: the measure's label and value, its kind (the verdict's rule), the
    sentence on what it measures, the contrast's label and value (none without one), a paired difference with
    its interval (INTERVAL), the floor's and the top's names in words (FLOOR), the count behind a share, and the
    file's columns behind it."""

    label: str
    value: float
    kind: str
    sentence: str
    contrast_label: str = ""
    contrast: Optional[float] = None
    diff: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    percent: bool = True
    floor_name: str = ""
    top_name: str = ""
    floor_word: str = "the floor"  # what the verdict calls the floor: "all cards" where the floor is no floor
    count: Optional[int] = None
    columns: tuple[str, ...] = ()

    @property
    def shown(self) -> str:
        return shown(self.value, self.percent)

    @property
    def contrast_shown(self) -> str:
        """The contrast's value, with its label: "random positions 4 %"; "" without a contrast."""
        if self.contrast is None:
            return ""
        return f"{self.contrast_label} {shown(self.contrast, self.percent)}".strip()

    @property
    def of(self) -> str:
        """"of 77 pairs" behind a share counted over pairs; "" without a count."""
        return f"of {self.count} pairs" if self.count else ""

    @property
    def gain(self) -> str:
        """The newer generation's number as its gain over the one before: "+5.4 points (1.8 to 9.3)" with the
        interval, "+3.5 points" or "−0.40" without one; "" where the number stands alone or has no difference."""
        if self.contrast is None:
            return ""
        diff = self.diff if self.diff is not None else self.value - self.contrast
        if self.kind == FLOOR and self.diff is None:
            return ""
        text = f"{diff * 100:+.1f} points" if self.percent else f"{diff:+.2f}"
        if self.low is not None and self.high is not None:
            bounds = (f"{self.low * 100:+.1f}", f"{self.high * 100:+.1f}") if self.percent else (f"{self.low:+.2f}", f"{self.high:+.2f}")
            text += f" ({bounds[0]} to {bounds[1]})"
        return text.replace("-", "−")

    @property
    def verdict(self) -> str:
        """The verdict by the number's kind; NO_CONTRAST for a floor or an interval without its contrast."""
        if self.kind == INTERVAL:
            return interval_verdict(self.low, self.high) if self.low is not None and self.high is not None else NO_CONTRAST
        if self.kind == KAPPA:
            return kappa_verdict(self.value)
        if self.kind == FLOOR:
            return floor_verdict(self.value, self.contrast, self.floor_word) if self.contrast is not None else NO_CONTRAST
        return rate_verdict(self.value, self.kind == RATE_LOW)

    @property
    def explanation(self) -> str:
        """The verdict's explanation in words: a floor's share of the way; "" for the other kinds."""
        if self.kind == FLOOR and self.contrast is not None:
            return floor_explanation(self.value, self.contrast, self.floor_name, self.top_name, shown(self.contrast, self.percent))
        return ""


# --- the sources ----------------------------------------------------------------------------------------------


def _number(row: Optional[Mapping[str, str]], column: str) -> Optional[float]:
    """A row's value as a number; None without the row, the column or a value."""
    if row is None:
        return None
    text = (row.get(column) or "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _first(run: Sequence[Mapping[str, str]], test: Callable[[Mapping[str, str]], bool]) -> Optional[Mapping[str, str]]:
    return next((row for row in run if test(row)), None)


def _interval(row: Mapping[str, str], column: str = "hand_diff") -> tuple[Optional[float], Optional[float], Optional[float]]:
    """A paired difference with its interval: the column and its `_low` and `_high` columns."""
    return _number(row, column), _number(row, f"{column}_low"), _number(row, f"{column}_high")


def _s1(run, chosen) -> list[Headline]:
    row = _first(run, lambda row: row.get("model") == "S1" and CHOSEN in (row.get("variant") or ""))
    floors = [_number(row, "precision_at_5") for row in run if row.get("model") == "B0"]
    floors = [value for value in floors if value is not None]
    found = []
    precision = _number(row, "precision_at_5")
    if precision is not None:
        found.append(Headline(
            "same-deck precision@5", precision, FLOOR, "Of each card's five nearest cards, the share from the card's own deck.",
            "random positions", sum(floors) / len(floors) if floors else None, floor_name="random positions",
            top_name="all five nearest in the card's deck", columns=("precision_at_5",),
        ))
    p1 = _number(row, "hand_p1")
    if p1 is not None:
        found.append(Headline("judged precision@1", p1, RATE, JUDGED_P1, columns=("hand_p1",)))
    return found


def _s2(run, chosen) -> list[Headline]:
    row = _first(run, lambda row: row.get("model") == "S2" and row.get("variant") == chosen.get("S2"))
    reference = _first(run, lambda row: row.get("model") == "S1" and CHOSEN in (row.get("variant") or ""))
    found = []
    precision = _number(row, "hand_precision")
    if precision is not None:
        diff, low, high = _interval(row)
        found.append(Headline(
            "judged precision@5", precision, INTERVAL, "Of the five nearest, the share Claude judged related; its gain over S1, paired over the judged cards.",
            "S1", _number(reference, "hand_precision"), diff, low, high, columns=("hand_precision", "hand_diff", "hand_diff_low", "hand_diff_high"),
        ))
    p1 = _number(row, "hand_p1")
    if p1 is not None:
        found.append(Headline("judged precision@1", p1, RATE, JUDGED_P1, "S1", _number(reference, "hand_p1"), columns=("hand_p1",)))
    return found


def _c1(run, chosen) -> list[Headline]:
    row = _first(run, lambda row: row.get("model") == "C1" and row.get("variant") == "model")
    random = _first(run, lambda row: row.get("model") == "C1" and row.get("variant") == "random")
    found = []
    cards = _number(row, "trust_cards")
    if cards is not None:
        found.append(Headline(
            "trustworthiness of the card layouts", cards, FLOOR, "Whether a card's nearest on its topic's sphere are its nearest by text.",
            "a random layout", _number(random, "trust_cards"), percent=False, floor_name="a random layout",
            top_name="no stranger by text among a card's nearest on the map", columns=("trust_cards",),
        ))
    decks = _number(row, "trust_decks")
    if decks is not None:
        found.append(Headline(
            "trustworthiness of the decks", decks, FLOOR, "Whether a deck's three nearest on the map are its nearest by text.",
            "a random layout", _number(random, "trust_decks"), percent=False, floor_name="a random layout",
            top_name="no stranger by text among a deck's three nearest on the map", columns=("trust_decks",),
        ))
    return found


def _t1(run, chosen) -> list[Headline]:
    course = _first(run, lambda row: row.get("model") == "T1" and row.get("scope") == COURSE_SCOPE)
    chance = _first(run, lambda row: row.get("model") == "chance" and row.get("scope") == COURSE_SCOPE)
    nearest = _first(run, lambda row: row.get("model") == "1-NN" and row.get("scope") == COURSE_SCOPE)
    focus = _first(run, lambda row: row.get("model") == "T1" and row.get("scope") == FOCUS_25)
    found = []
    agreement = _number(course, "agreement")
    if agreement is not None:
        beside = _number(nearest, "agreement")
        sentence = "The share of cards T1 puts into their scripted topic; the rest are its suggestions, not errors."
        if beside is not None:
            sentence += f" The nearest-card rule 1-NN gets {shown(beside, True)}."
        found.append(Headline(
            "agreement with the script", agreement, FLOOR, sentence, "chance", _number(chance, "agreement"),
            floor_name="chance by the topic sizes", top_name="every card in its scripted topic", columns=("agreement",),
        ))
    confident = _number(focus, "agreement")
    if confident is not None:
        found.append(Headline(
            "agreement of the 25 % most confident", confident, FLOOR,
            "The same share over the quarter of the cards T1 is surest of, by their margin.",
            "all cards", agreement, floor_name="all cards", top_name="every card in its scripted topic", floor_word="all cards",
            columns=("agreement",),
        ))
    return found


def _d1_chosen(run) -> Optional[Mapping[str, str]]:
    return _first(run, lambda row: row.get("model") == "D1" and CHOSEN in (row.get("variant") or "")
                  and row.get("label_rule") == "deck shares" and course_row(row))


def _d1(run, chosen) -> list[Headline]:
    row = _d1_chosen(run)
    easy = _first(run, lambda row: row.get("model") == "baseline" and row.get("variant") == "always easy" and course_row(row))
    found = []
    order = _number(row, "hand_within_deck")
    if order is not None:
        pairs = _number(row, "hand_within_pairs")
        diff, low, high = _interval(row, "hand_coin_diff")
        found.append(Headline(
            ORDER_LABEL, order, FLOOR,
            "Of the pairs of quizzes in a deck that Claude put at different levels, the share D1 puts in the same order.", "a coin", COIN,
            diff, low, high, floor_name="a coin", top_name="every pair in order", count=int(pairs) if pairs else None,
            columns=("hand_within_deck", "hand_within_pairs", "hand_coin_diff", "hand_coin_diff_low", "hand_coin_diff_high"),
        ))
    kappa = _number(row, "kappa")
    if kappa is not None:
        found.append(Headline(
            "weighted kappa", kappa, KAPPA, "Agreement with the levels the rated quizzes came with, beyond chance, a near miss counted half; never accuracy, which always easy beats.",
            "always easy", _number(easy, "kappa"), percent=False, columns=("kappa",),
        ))
    return found


def _d2(run, chosen) -> list[Headline]:
    row = _first(run, lambda row: row.get("model") == "D2" and row.get("variant") == chosen.get("D2")
                 and row.get("label_rule") == "deck shares" and course_row(row))
    reference = _d1_chosen(run)
    found = []
    order = _number(row, "hand_within_deck")
    if order is not None:
        diff, low, high = _interval(row)
        pairs = _number(row, "hand_within_pairs")
        found.append(Headline(
            ORDER_LABEL, order, INTERVAL,
            "The share of the pairs Claude put at different levels in the same order; its gain over D1, paired over the pairs.", "D1",
            _number(reference, "hand_within_deck"), diff, low, high, count=int(pairs) if pairs else None,
            columns=("hand_within_deck", "hand_within_pairs", "hand_diff", "hand_diff_low", "hand_diff_high"),
        ))
    kappa = _number(row, "kappa")
    if kappa is not None:
        found.append(Headline(
            "weighted kappa on the rated quizzes", kappa, KAPPA, "Agreement with the levels the rated quizzes came with, beyond chance, next to D1's.",
            "D1", _number(reference, "kappa"), percent=False, columns=("kappa",),
        ))
    return found


def _p1(run, chosen) -> list[Headline]:
    app = chosen.get("P1")
    row = _first(run, lambda row: row.get("model") == "P1" and row.get("variant") == app)
    tfidf = _first(run, lambda row: row.get("model") == "P1" and row.get("variant") == TFIDF_CARD) if app != TFIDF_CARD else None
    found = []
    right = _number(row, "hand_right")
    if right is not None:
        diff, low, high = _interval(row, "hand_tfidf_diff") if tfidf is not None else (None, None, None)
        found.append(Headline(
            "share served right", right, RATE,
            "Of the judged requests, the share taken into an acceptable topic, or told no match on an off-course request.",
            "TF-IDF", _number(tfidf, "hand_right"), diff, low, high,
            columns=("hand_right", "hand_tfidf_diff", "hand_tfidf_diff_low", "hand_tfidf_diff_high"),
        ))
    wrong = _number(row, "hand_wrong")
    if wrong is not None:
        missed = _number(row, "hand_missed")
        sentence = "The share taken into a wrong topic, where lower is better"
        sentence += f"; the rest ({shown(missed, True)}) got no match." if missed is not None else "; the rest got no match."
        diff, low, high = _interval(row, "hand_tfidf_wrong_diff") if tfidf is not None else (None, None, None)
        found.append(Headline(
            "share landed wrong", wrong, RATE_LOW, sentence, "TF-IDF", _number(tfidf, "hand_wrong"), diff, low, high,
            columns=("hand_wrong", "hand_tfidf_wrong_diff", "hand_tfidf_wrong_diff_low", "hand_tfidf_wrong_diff_high"),
        ))
    return found


MAKERS = {"S1": _s1, "S2": _s2, "C1": _c1, "T1": _t1, "D1": _d1, "D2": _d2, "P1": _p1}


def headlines(results, level: str) -> Optional[list[Headline]]:
    """The generation's two numbers from its newest run (results.Results.run, the run its tables show); None without
    the file or the rows."""
    run = results.run(level)
    if not run:
        return None
    return MAKERS[level](run, results.chosen)


def outcomes(results) -> Optional[dict[str, int]]:
    """P1's outcomes counted on and off the course (OUTCOME_COLUMNS) in its app row of the newest run; None without
    the row or without the counts (a run before evaluate.py wrote them)."""
    row = _first(results.run("P1") or [], lambda row: row.get("model") == "P1" and row.get("variant") == results.chosen.get("P1"))
    counts = {column: _number(row, column) for column in OUTCOME_COLUMNS}
    if any(value is None for value in counts.values()):
        return None
    return {column: int(value) for column, value in counts.items()}


def chosen_label(results, level: str) -> str:
    """The chosen variant's label of the generation's newest run: the app's pick where `Results.chosen` names it
    (S2, D2, P1), the row marked chosen (S1, D1), C1's model row, T1 itself; "" without the rows."""
    run = results.run(level) or []
    if level in results.chosen:
        return results.chosen[level]
    if level == "C1":
        return "model" if any(row.get("variant") == "model" for row in run) else ""
    if level == "T1":
        return "T1" if run else ""
    row = _first(run, lambda row: row.get("model") == level and CHOSEN in (row.get("variant") or ""))
    return (row.get("variant") or "") if row else ""

