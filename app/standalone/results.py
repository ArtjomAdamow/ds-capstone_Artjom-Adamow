"""The results files for the standalone page's model panels (the csv module, nothing of Dash): the rows of the
current course, the newest run of each file, one table per level with one row per measure and one column per
variant next to its floor, every deck or topic name of the course table replaced by the deck's id or the topic's
label, and the glossary of the measure columns (app/docs/columns.md).

* `Results(folder, note, names)`: reads results.csv, topics.csv and difficulty.csv of `folder` once; a missing file
  gives the hint how to make one (`hint`). The course's rows are those whose note names the table's decks (`note`,
  "decks 01 to 24", as evaluate.scope_note writes it), so the rows of an older course stay out. A run is the `run`
  column, the date for the rows written before it, and a level shows its newest run.
* `Table`: the heads (one per variant, the floor among them), the rows (a measure and its value per head; a measure
  with its sd reads "mean ± sd", a difference with its interval "diff [low, high]", a share with its count
  "share (count)"), which heads are the chosen variant (the word "chosen" in the name; C1's "model"), the subtitle
  (the run and the bookkeeping the heads share) and the measure columns shown, for the glossary.
* `read_glossary`, `glossary_rows`: the rows of app/docs/columns.md, by column name, for the measures a table shows.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

RESULTS, TOPICS, DIFFICULTY = "results.csv", "topics.csv", "difficulty.csv"
FILES = (RESULTS, TOPICS, DIFFICULTY)
GLOSSARY_PATH = Path(__file__).resolve().parents[1] / "docs" / "columns.md"
BOOKKEEPING = ("params", "features", "folds", "repeats", "seed")  # the subtitle, where every head shares the value
CHOSEN = "chosen"
HINT = "No {name} in {folder} yet. The offline evaluation writes it, next to app/: uv run evaluate.py"
NO_FOLDER = "No results folder. The offline evaluation writes one, next to app/: uv run evaluate.py"
Measure = Union[str, tuple[str, ...]]  # a column, or the columns of one measure (a value with its sd or interval)
S1_MEASURES: tuple[Measure, ...] = (
    "precision_at_5", "kept_precision", "coverage", "distinct", "min_score", "hand_precision", "hand_p1", "hand_p3",
    ("hand_diff", "hand_diff_low", "hand_diff_high"), "judged_cards",
)
C1_MEASURES: tuple[Measure, ...] = (
    "trust_decks", "overlap_decks", "overlap_cards", "overlap_k2", "lift_k2", "overlap_k3", "lift_k3", "trust_cards",
    "overlap_17", "overlap_24",
)
T1_MEASURES: tuple[Measure, ...] = (
    "cards", "topics", "agreement", "kept_share", "coherence_median", "coherence_below_half", "coherence_at_one", "moved",
    "emptied",
)
T1_DECK_MEASURES: tuple[Measure, ...] = ("cards", "topics", "agreement")
T1_STACK_MEASURES: tuple[Measure, ...] = ("cards", "agreement")
T1_MARGIN_MEASURES: tuple[Measure, ...] = ("cards", "kept_share", "agreement")
D1_MEASURES: tuple[Measure, ...] = (
    ("macro_f1", "macro_f1_sd"), ("balanced_accuracy", "balanced_accuracy_sd"), ("kappa", "kappa_sd"), ("accuracy", "accuracy_sd"),
    "spearman", "recall_easy", "recall_medium", "recall_hard", "rated", "unrated", "predicted_easy", "predicted_medium",
    "predicted_hard", "hand_judged", "hand_macro_f1", "hand_kappa", "hand_accuracy", "hand_spearman",
    ("hand_within_deck", "hand_within_pairs"),
)
MARGIN_SCOPE = "margin >= "
STACK_SCOPE = "stack "
T1_JUDGES = ("T1", "1-NN", "LR", "chance")


def course_note(deck_ids: Mapping[str, str]) -> str:
    """The note that marks a row of this course: "decks 01 to 24" (evaluate.scope_note writes the same)."""
    ids = sorted(deck_ids.values())
    return f"decks {ids[0]} to {ids[-1]}" if ids else "no decks"


@dataclass
class Table:
    """One level's results of the newest run: a column per variant, a row per measure."""

    title: str
    heads: list[str]
    rows: list[tuple[str, list[str]]]  # (the measure's label, its value under every head)
    chosen: list[bool]  # per head
    subtitle: str
    measures: list[str] = field(default_factory=list)  # the file's columns behind the rows, for the glossary


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_key(row: Mapping[str, str]) -> str:
    """The run a row belongs to: its `run` column, else its date (the rows written before the column)."""
    return row.get("run") or row.get("date") or ""


def newest(rows: Sequence[Mapping[str, str]]) -> list[dict]:
    """The rows of the newest run among `rows`."""
    if not rows:
        return []
    key = max(run_key(row) for row in rows)
    return [dict(row) for row in rows if run_key(row) == key]


def cell(row: Mapping[str, str], measure: Measure) -> str:
    """A row's value of a measure: the column, or its columns joined ("mean ± sd", "diff [low, high]", "share (count)")."""
    if isinstance(measure, str):
        return row.get(measure) or ""
    values = [row.get(name) or "" for name in measure]
    if not values[0]:
        return ""
    if len(values) == 3:
        return f"{values[0]} [{values[1]}, {values[2]}]" if values[1] and values[2] else values[0]
    if measure[1].endswith("_sd"):
        return f"{values[0]} ± {values[1]}" if values[1] else values[0]
    return f"{values[0]} ({values[1]})" if values[1] else values[0]


def label(measure: Measure) -> str:
    """The row's label: the column, or the first column with how the others are joined."""
    if isinstance(measure, str):
        return measure
    if len(measure) == 3:
        return f"{measure[0]} [low, high]"
    if measure[1].endswith("_sd"):
        return f"{measure[0]} ± sd"
    return f"{measure[0]} ({measure[1].split('_')[-1]})"


def names_of(measure: Measure) -> list[str]:
    return [measure] if isinstance(measure, str) else list(measure)


def transpose(title: str, rows: Sequence[Mapping[str, str]], heads: Sequence[str], chosen: Sequence[bool],
              measures: Sequence[Measure], by_ids=lambda text: text) -> Table:
    """The table: a measure every row of `rows` left empty is no row; the heads and cells pass through `by_ids`."""
    kept_rows: list[tuple[str, list[str]]] = []
    kept_measures: list[str] = []
    for measure in measures:
        values = [by_ids(cell(row, measure)) for row in rows]
        if any(values):
            kept_rows.append((label(measure), values))
            kept_measures += names_of(measure)
    return Table(title, [by_ids(head) for head in heads], kept_rows, list(chosen), by_ids(subtitle_of(rows)), kept_measures)


def subtitle_of(rows: Sequence[Mapping[str, str]]) -> str:
    """The run and the bookkeeping the table's own rows share (the rows of its most frequent model: D1's, not the
    hand row's; C1's three layouts): "run 2026-11-01T14:03 · folds 5 · repeats 3 · seed 42"."""
    if not rows:
        return ""
    models = [row.get("model") or "" for row in rows]
    main = max(set(models), key=models.count)
    own = [row for row in rows if (row.get("model") or "") == main]
    parts = [f"run {run_key(rows[0])}"]
    for key in BOOKKEEPING:
        values = {row.get(key) or "" for row in own}
        if len(values) == 1 and next(iter(values)):
            parts.append(f"{key} {next(iter(values))}")
    return " · ".join(parts)


class Results:
    """The results files of a folder, read once, with the tables per model letter."""

    def __init__(self, folder: Optional[Path], note: str, names: Optional[Mapping[str, str]] = None) -> None:
        self.folder = Path(folder) if folder is not None else None
        self.note = note
        self.names = dict(names or {})
        self._pattern = (
            re.compile("|".join(re.escape(name) for name in sorted(self.names, key=len, reverse=True))) if self.names else None
        )
        self.rows: dict[str, Optional[list[dict]]] = {}
        for name in FILES:
            path = self.folder / name if self.folder is not None else None
            self.rows[name] = self._current(read_rows(path)) if path is not None and path.is_file() else None

    def _current(self, rows: Sequence[Mapping[str, str]]) -> list[dict]:
        return [dict(row) for row in rows if self.note and self.note in (row.get("notes") or "")]

    def by_ids(self, text: str) -> str:
        """The text with every deck or topic name of the course table replaced (full names, longest first)."""
        if not text or self._pattern is None:
            return text
        return self._pattern.sub(lambda match: self.names[match.group(0)], text)

    def hint(self, name: str) -> str:
        """Why there is no table: the file is missing (how to make it), or no folder was given."""
        if self.folder is None:
            return NO_FOLDER
        return HINT.format(name=name, folder=self.folder)

    def tables(self, letter: str, deck_id: Optional[str] = None) -> Optional[list[Table]]:
        """The tables of a model's panel: S (S1 next to B0), C (C1's three layouts), T (the course or the entered
        deck, the stacks, the margins) or D (D1's rows next to the baselines and the hand row); None without the
        file."""
        name = {"S": RESULTS, "C": RESULTS, "T": TOPICS, "D": DIFFICULTY}[letter]
        rows = self.rows.get(name)
        if rows is None:
            return None
        run = newest(rows)
        if letter == "S":
            return [self._similarity(run)]
        if letter == "C":
            return [self._course(run)]
        if letter == "T":
            return self._topics(run, deck_id)
        return [self._difficulty(run)]

    def _similarity(self, run: list[dict]) -> Table:
        variants = [row for row in run if row.get("model") == "S1"]
        floors = [row for row in run if row.get("model") == "B0"]
        rows: list[dict] = []
        heads: list[str] = []
        if floors:
            values = [float(row["precision_at_5"]) for row in floors if row.get("precision_at_5")]
            mean = f"{sum(values) / len(values):.3f}" if values else ""
            rows.append({**floors[0], "precision_at_5": mean})
            heads.append(f"B0 random positions (mean of {len(floors)} seed{'s' if len(floors) != 1 else ''})")
        rows += variants
        heads += [row.get("variant") or "" for row in variants]
        chosen = [False] * len(floors[:1]) + [CHOSEN in (row.get("variant") or "") for row in variants]
        return transpose("S1 next to the random floor B0", rows, heads, chosen, S1_MEASURES, self.by_ids)

    def _course(self, run: list[dict]) -> Table:
        rows = [row for row in run if row.get("model") == "C1"]
        heads = [row.get("variant") or "" for row in rows]
        chosen = [head == "model" for head in heads]
        return transpose("C1 next to the SVD start and random directions", rows, heads, chosen, C1_MEASURES, self.by_ids)

    def _topics(self, run: list[dict], deck_id: Optional[str]) -> list[Table]:
        scope = f"deck {deck_id}" if deck_id else "all"
        judged = [row for row in run if row.get("scope") == scope]
        judged.sort(key=lambda row: T1_JUDGES.index(row["model"]) if row.get("model") in T1_JUDGES else len(T1_JUDGES))
        heads = [row.get("model") or "" for row in judged]
        title = f"T1 next to the judges, deck {deck_id}" if deck_id else "T1 next to the judges, the course"
        tables = [transpose(title, judged, heads, [head == "T1" for head in heads], T1_DECK_MEASURES if deck_id else T1_MEASURES, self.by_ids)]
        stacks = [row for row in run if row.get("model") == "T1" and (row.get("scope") or "").startswith(STACK_SCOPE)]
        if stacks:
            tables.append(transpose("T1 by stack", stacks, [row["scope"] for row in stacks], [False] * len(stacks), T1_STACK_MEASURES, self.by_ids))
        margins = [row for row in run if row.get("model") == "T1" and (row.get("scope") or "").startswith(MARGIN_SCOPE)]
        if margins:
            heads = [row["scope"] for row in margins]
            tables.append(transpose("T1's cards kept at a margin", margins, heads, [False] * len(margins), T1_MARGIN_MEASURES, self.by_ids))
        return tables

    def _difficulty(self, run: list[dict]) -> Table:
        order = {"D1": 0, "baseline": 1, "hand": 2}
        rows = sorted((row for row in run if row.get("model") in order), key=lambda row: order[row["model"]])
        heads = [difficulty_head(row) for row in rows]
        chosen = [CHOSEN in (row.get("variant") or "") and row.get("label_rule") == "deck shares" for row in rows]
        return transpose("D1 next to the baselines and the hand labels", rows, heads, chosen, D1_MEASURES, self.by_ids)


def difficulty_head(row: Mapping[str, str]) -> str:
    """A column of D1's table: the feature set and the label rule, a baseline by its name, the hand row by its kind."""
    variant = row.get("variant") or ""
    if row.get("model") == "D1":
        return f"{variant} · {row.get('label_rule') or ''}"
    if row.get("model") == "baseline":
        return f"baseline: {variant}"
    return f"hand labels: {variant}"


# --- the glossary ---------------------------------------------------------------------------------------------


def read_glossary(path: Path = GLOSSARY_PATH) -> dict[str, tuple[str, str, str]]:
    """The rows of columns.md by column name: (the names cell, what it measures, the range); the heading and the
    separator rows of its tables are skipped."""
    entries: dict[str, tuple[str, str, str]] = {}
    if not Path(path).is_file():
        return entries
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("", "Column") or set(cells[0]) <= set("-: "):
            continue
        for name in cells[0].split(","):
            entries[name.strip()] = (cells[0], cells[1], cells[2])
    return entries


def glossary_rows(measures: Sequence[str], glossary: Mapping[str, tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """The glossary's rows of these measures, in their order, each once."""
    rows: list[tuple[str, str, str]] = []
    for name in measures:
        entry = glossary.get(name)
        if entry is not None and entry not in rows:
            rows.append(entry)
    return rows
