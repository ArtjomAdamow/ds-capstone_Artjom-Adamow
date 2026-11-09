"""The results files for the model panels of the dashboard and the models view (the csv module, nothing of Dash):
the rows of the current course, each level's newest run, its tables with one row per measure and one column per
variant next to its floor, and the glossary of the measure columns (app/docs/columns.md).

* `Results(folder, note, chosen)`: reads results.csv, topics.csv, difficulty.csv and prompt.csv of `folder`
  once; a missing file gives the hint how to make one (`hint`). The course's rows are those whose note names the table's decks (`note`,
  "decks 01 to 24", as evaluate.scope_note writes it), so the rows of an older course stay out. A run is the `run`
  column, the date for the rows written before it, and a level (S1, S2, C1, T1, D1, D2, P1: the model generations)
  shows the newest run that holds its rows (`run`), so a later run of another part or without the encoder hides
  none. `chosen` names the variant the app runs where the file's names do not say: S2's (the hybrid with MiniLM,
  `APP_VECTORISER`), D2's (the criterion template on the large model, `NLI_MODEL`) and P1's (the level's variant
  label); their tables mark the app's column next to the reference column, and the section's numbers
  (app.headlines) read the same row.
* A row's `scope` is the course ("all", or empty on older rows) or a deck ("deck 01"): the difficulty tables take
  the course's rows, and with a deck entered a second table shows the deck's rows of the same run (D1's variants'
  within-deck order on the hand labels; D2's next to D1's chosen row), as T1's deck rows do.
* `Table`: the heads (one per variant, the floor among them), the rows (a measure and its value per head; a measure
  with its sd reads "mean ± sd", a difference with its interval "diff [low, high]", a share with its count
  "share (count)"), which heads are the chosen variant (the word "chosen" in the name; C1's "model"), the subtitle
  (the run and the bookkeeping the heads share), the measure columns shown and the columns behind each row, for the
  row's description, and the file it comes from.
* `read_glossary`: the rows of app/docs/columns.md by column name, what a row of a table means when it is clicked;
  each under its file's key too ("results.csv:hand_diff"), since a column of the same name can mean another measure
  in another file (`glossary_entry`).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

RESULTS, TOPICS, DIFFICULTY, PROMPT = "results.csv", "topics.csv", "difficulty.csv", "prompt.csv"
FILES = (RESULTS, TOPICS, DIFFICULTY, PROMPT)
GLOSSARY_PATH = Path(__file__).resolve().parent / "docs" / "columns.md"
BOOKKEEPING = ("params", "features", "folds", "repeats", "seed")  # the subtitle, where every head shares the value
CHOSEN = "chosen"
HINT = "No {name} in {folder} yet. The offline evaluation writes it, next to app/: uv run evaluate.py"
NO_ROWS = "No {level} rows of this course in {name} yet. The offline evaluation writes them, next to app/: uv run evaluate.py"
# The model generations per file, and the model column that marks a run as the level's
LEVEL_FILES = {"S1": RESULTS, "S2": RESULTS, "C1": RESULTS, "T1": TOPICS, "D1": DIFFICULTY, "D2": DIFFICULTY, "P1": PROMPT}
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
T1_MARGIN_MEASURES: tuple[Measure, ...] = ("cards", "margin", "kept_share", "agreement")
COURSE_SCOPE = "all"  # evaluate.COURSE_SCOPE: a row over the course; a deck's rows read "deck <id>"
D1_DECK_MEASURES: tuple[Measure, ...] = (
    "rated", "unrated", "hand_judged", ("hand_within_deck", "hand_within_pairs"), ("hand_diff", "hand_diff_low", "hand_diff_high"),
)
D1_MEASURES: tuple[Measure, ...] = (
    ("macro_f1", "macro_f1_sd"), ("balanced_accuracy", "balanced_accuracy_sd"), ("kappa", "kappa_sd"), ("accuracy", "accuracy_sd"),
    "spearman", "recall_easy", "recall_medium", "recall_hard", "rated", "unrated", "predicted_easy", "predicted_medium",
    "predicted_hard", "hand_judged", "hand_macro_f1", "hand_kappa", "hand_accuracy", "hand_spearman",
    ("hand_within_deck", "hand_within_pairs"), ("hand_diff", "hand_diff_low", "hand_diff_high"),
)
P1_MEASURES: tuple[Measure, ...] = (
    "hand_right", "hand_wrong", "hand_missed", ("hand_diff", "hand_diff_low", "hand_diff_high"), "hand_topic_at1", "hand_topic_at3",
    "hand_card_at1", "judged_requests", "self_topic_at1", "self_topic_at3", "self_navigated", "min_card", "min_centroid",
    "min_text", "ms_per_request",
)
MARGIN_SCOPE = "margin >= "
FOCUS_SCOPE = "focus "  # the share scale's marks: the share of the course's cards kept (the margin slider of step 28 ran on them; it went with step 33)
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
    measures: list[str] = field(default_factory=list)  # the file's columns behind the rows, in row order
    row_columns: list[list[str]] = field(default_factory=list)  # per row, its columns (the glossary's key is the first)
    file: str = ""  # the results file, whose glossary rows describe the table's


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_key(row: Mapping[str, str]) -> str:
    """The run a row belongs to: its `run` column, else its date (the rows written before the column)."""
    return row.get("run") or row.get("date") or ""


def newest(rows: Sequence[Mapping[str, str]], model: Optional[str] = None) -> list[dict]:
    """The rows of the newest run among `rows`; with `model`, of the newest run that holds a row of it (none
    without one)."""
    keys = [run_key(row) for row in rows if model is None or row.get("model") == model]
    if not keys:
        return []
    key = max(keys)
    return [dict(row) for row in rows if run_key(row) == key]


def course_row(row: Mapping[str, str]) -> bool:
    """Whether the row is over the course (its scope "all", or none on a row written before the column)."""
    return (row.get("scope") or COURSE_SCOPE) == COURSE_SCOPE


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
              measures: Sequence[Measure]) -> Table:
    """The table: a measure every row of `rows` left empty is no row."""
    kept_rows: list[tuple[str, list[str]]] = []
    kept_measures: list[str] = []
    row_columns: list[list[str]] = []
    for measure in measures:
        values = [cell(row, measure) for row in rows]
        if any(values):
            kept_rows.append((label(measure), values))
            kept_measures += names_of(measure)
            row_columns.append(names_of(measure))
    return Table(title, list(heads), kept_rows, list(chosen), subtitle_of(rows), kept_measures, row_columns)


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
    """The results files of a folder, read once, with the tables per model letter: the rows whose notes name the
    course (`note`, course_note's); with no note, without a course table (the dashboard's demo alone), every row, so
    each level shows its newest run whatever course it names."""

    def __init__(self, folder: Optional[Path], note: Optional[str], chosen: Optional[Mapping[str, str]] = None) -> None:
        self.folder = Path(folder) if folder is not None else None
        self.note = note
        self.chosen = dict(chosen or {})  # a level -> the variant the app runs, where its file's names do not say
        self.rows: dict[str, Optional[list[dict]]] = {}
        for name in FILES:
            path = self.folder / name if self.folder is not None else None
            self.rows[name] = self._current(read_rows(path)) if path is not None and path.is_file() else None

    def _current(self, rows: Sequence[Mapping[str, str]]) -> list[dict]:
        return [dict(row) for row in rows if self.note is None or (self.note and self.note in (row.get("notes") or ""))]

    def hint(self, name: str) -> str:
        """Why there is no table: the file is missing (how to make it), or no folder was given."""
        if self.folder is None:
            return NO_FOLDER
        return HINT.format(name=name, folder=self.folder)

    def run(self, level: str) -> Optional[list[dict]]:
        """The rows of the level's newest run (the run its tables and its headline numbers show); None without the
        file, empty without the level's rows."""
        rows = self.rows.get(LEVEL_FILES[level])
        return newest(rows, level) if rows is not None else None

    def tables(self, level: str, deck_id: Optional[str] = None) -> Optional[list[Table]]:
        """The tables of a level, from its newest run: S1 (next to B0), S2 (next to S1's chosen variant, the app's
        variant marked), C1 (its three layouts), T1 (the course or the entered deck, the stacks, the margins, the
        slider's marks), D1 (next to the baselines and the hand row) or D2 (next to D1's chosen row, the app's
        variant marked), the two with the entered deck's rows as a second table, P1 (its variants, the app's
        marked); None without the file, empty without the level's rows."""
        run = self.run(level)
        if run is None:
            return None
        if not run:
            return []
        if level == "S1":
            tables = [self._similarity(run)]
        elif level == "S2":
            tables = [self._embeddings(run)]
        elif level == "C1":
            tables = [self._course(run)]
        elif level == "T1":
            tables = self._topics(run, deck_id)
        elif level == "D1":
            tables = [self._difficulty(run)] + self._deck_difficulty(run, level, deck_id)
        elif level == "P1":
            tables = [self._prompt(run)]
        else:
            tables = [self._tags(run)] + self._deck_difficulty(run, level, deck_id)
        for table in tables:
            table.file = LEVEL_FILES[level]
        return tables

    def no_rows(self, level: str) -> str:
        """Why a level has no table though its file is there."""
        return NO_ROWS.format(level=level, name=LEVEL_FILES[level])

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
        return transpose("S1 next to the random floor B0", rows, heads, chosen, S1_MEASURES)

    def _embeddings(self, run: list[dict]) -> Table:
        """S2's rows of the run (the encoders and the hybrids) next to S1's chosen variant, which every paired
        difference is against; the app's variant (`chosen`) marked next to the reference."""
        reference = [row for row in run if row.get("model") == "S1" and CHOSEN in (row.get("variant") or "")][:1]
        variants = [row for row in run if row.get("model") == "S2"]
        rows = reference + variants
        heads = [f"S1 {row.get('variant') or ''}" for row in reference] + [row.get("variant") or "" for row in variants]
        chosen = [True] * len(reference) + [row.get("variant") == self.chosen.get("S2") for row in variants]
        return transpose("S2 next to S1's chosen variant", rows, heads, chosen, S1_MEASURES)

    def _course(self, run: list[dict]) -> Table:
        rows = [row for row in run if row.get("model") == "C1"]
        heads = [row.get("variant") or "" for row in rows]
        chosen = [head == "model" for head in heads]
        return transpose("C1 next to the SVD start and random directions", rows, heads, chosen, C1_MEASURES)

    def _topics(self, run: list[dict], deck_id: Optional[str]) -> list[Table]:
        scope = f"deck {deck_id}" if deck_id else "all"
        judged = [row for row in run if row.get("scope") == scope]
        judged.sort(key=lambda row: T1_JUDGES.index(row["model"]) if row.get("model") in T1_JUDGES else len(T1_JUDGES))
        heads = [row.get("model") or "" for row in judged]
        title = f"T1 next to the judges, deck {deck_id}" if deck_id else "T1 next to the judges, the course"
        tables = [transpose(title, judged, heads, [head == "T1" for head in heads], T1_DECK_MEASURES if deck_id else T1_MEASURES)]
        stacks = [row for row in run if row.get("model") == "T1" and (row.get("scope") or "").startswith(STACK_SCOPE)]
        if stacks:
            tables.append(transpose("T1 by stack", stacks, [row["scope"] for row in stacks], [False] * len(stacks), T1_STACK_MEASURES))
        margins = [row for row in run if row.get("model") == "T1" and (row.get("scope") or "").startswith(MARGIN_SCOPE)]
        if margins:
            heads = [row["scope"] for row in margins]
            tables.append(transpose("T1's cards kept at a margin", margins, heads, [False] * len(margins), T1_MARGIN_MEASURES))
        focus = [row for row in run if row.get("model") == "T1" and (row.get("scope") or "").startswith(FOCUS_SCOPE)]
        if focus:
            heads = [row["scope"] for row in focus]
            tables.append(transpose("The margin on the share scale: the cards kept at its marks", focus, heads, [False] * len(focus), T1_MARGIN_MEASURES))
        return tables

    def _difficulty(self, run: list[dict]) -> Table:
        order = {"D1": 0, "baseline": 1, "hand": 2}
        rows = sorted((row for row in run if row.get("model") in order and course_row(row)), key=lambda row: order[row["model"]])
        heads = [difficulty_head(row) for row in rows]
        chosen = [CHOSEN in (row.get("variant") or "") and row.get("label_rule") == "deck shares" for row in rows]
        return transpose("D1 next to the baselines and the hand labels", rows, heads, chosen, D1_MEASURES)

    def _tags(self, run: list[dict]) -> Table:
        """D2's rows of the run (the templates, each by the deck shares and by the most likely class) next to D1's
        chosen row, which every paired difference is against; the app's variant (`chosen`, by the deck shares)
        marked next to the reference."""
        reference = [row for row in run if row.get("model") == "D1" and CHOSEN in (row.get("variant") or "")
                     and row.get("label_rule") == "deck shares" and course_row(row)][:1]
        variants = [row for row in run if row.get("model") == "D2" and course_row(row)]
        rows = reference + variants
        heads = [f"D1 {difficulty_head(row)}" for row in reference] + [difficulty_head(row) for row in variants]
        chosen = [True] * len(reference) + [self.app_tags(row) for row in variants]
        return transpose("D2 next to D1's chosen row", rows, heads, chosen, D1_MEASURES)

    def app_tags(self, row: Mapping[str, str]) -> bool:
        """Whether a D2 row is the app's: its variant the app's template and model (`chosen`), by the deck shares."""
        return row.get("variant") == self.chosen.get("D2") and row.get("label_rule") == "deck shares"

    def _prompt(self, run: list[dict]) -> Table:
        """P1's variants of the run (the route sets on each vectoriser, the character n-grams' card route), the
        variant the app runs marked, the hand columns first."""
        rows = [row for row in run if row.get("model") == "P1"]
        heads = [row.get("variant") or "" for row in rows]
        chosen = [head == self.chosen.get("P1") for head in heads]
        return transpose("P1's variants: the judged requests and the self set", rows, heads, chosen, P1_MEASURES)

    def _deck_difficulty(self, run: list[dict], level: str, deck_id: Optional[str]) -> list[Table]:
        """The entered deck's rows of the run (scope "deck <id>"): D1's variants' within-deck
        order on the hand labels, or D2's variants next to D1's chosen row; none without a deck or its rows."""
        if not deck_id:
            return []
        deck_rows = [row for row in run if row.get("scope") == f"deck {deck_id}"]
        if level == "D1":
            rows = [row for row in deck_rows if row.get("model") == "D1"]
            heads = [difficulty_head(row) for row in rows]
            chosen = [CHOSEN in (row.get("variant") or "") for row in rows]
            title = f"D1 within deck {deck_id}: the hand order"
        else:
            reference = [row for row in deck_rows if row.get("model") == "D1" and CHOSEN in (row.get("variant") or "")][:1]
            variants = [row for row in deck_rows if row.get("model") == "D2"]
            rows = reference + variants
            heads = [f"D1 {difficulty_head(row)}" for row in reference] + [difficulty_head(row) for row in variants]
            chosen = [True] * len(reference) + [self.app_tags(row) for row in variants]
            title = f"D2 within deck {deck_id} next to D1's chosen row"
        if not rows:
            return []
        return [transpose(title, rows, heads, chosen, D1_DECK_MEASURES)]


def difficulty_head(row: Mapping[str, str]) -> str:
    """A column of the difficulty tables: D1's feature set and label rule, D2's template and label rule, a baseline
    by its name, the hand row by its kind."""
    variant = row.get("variant") or ""
    if row.get("model") == "D1":
        return f"{variant} · {row.get('label_rule') or ''}"
    if row.get("model") == "D2":
        return f"D2 {variant} · {row.get('label_rule') or ''}"
    if row.get("model") == "baseline":
        return f"baseline: {variant}"
    return f"hand labels: {variant}"


# --- the glossary ---------------------------------------------------------------------------------------------


def read_glossary(path: Path = GLOSSARY_PATH) -> dict[str, tuple[str, str, str]]:
    """The rows of columns.md by column name: (the names cell, what it measures, the range), each under its name (the
    first file's row where two files share one) and under its file's key ("file:name", the file its "## file" heading
    names); the heading and the separator rows of its tables are skipped."""
    entries: dict[str, tuple[str, str, str]] = {}
    if not Path(path).is_file():
        return entries
    file = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            file = line[3:].strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("", "Column") or set(cells[0]) <= set("-: "):
            continue
        for name in cells[0].split(","):
            entries.setdefault(name.strip(), (cells[0], cells[1], cells[2]))
            entries[f"{file}:{name.strip()}"] = (cells[0], cells[1], cells[2])
    return entries


def glossary_entry(glossary: Mapping[str, tuple[str, str, str]], file: str, column: str) -> Optional[tuple[str, str, str]]:
    """A column's row of the glossary: its file's, else the column's by name alone."""
    return glossary.get(f"{file}:{column}") or glossary.get(column)
