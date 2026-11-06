"""The results files for the model tiles of the dashboard (the csv module, nothing of Dash): the rows of the
current course and each level's newest run.

* `Results(folder, note, chosen)`: reads results.csv, topics.csv, difficulty.csv and prompt.csv of `folder`
  once; a missing file gives the hint how to make one (`hint`), a file without the level's rows the line that says
  so (`no_rows`). The course's rows are those whose note names the table's decks (`note`, "decks 01 to 24", as
  evaluate.scope_note writes it), so the rows of an older course stay out. A run is the `run` column, the date for
  the rows written before it, and a level (S1, S2, C1, T1, D1, D2, P1: the files' model generations) shows the newest run that
  holds its rows (`run`), so a later run of another part or without the encoder hides none. `chosen` names the
  variant the app runs where the file's names do not say: S2's (the hybrid with MiniLM, `APP_VECTORISER`) and P1's
  (the level's variant label); the section's numbers (app.headlines) read that row.
* A row's `scope` is the course ("all", or empty on older rows: `course_row`), a deck ("deck 01") or a part of T1's
  cards (FOCUS_SCOPE, the share scale's marks).
"""

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence

RESULTS, TOPICS, DIFFICULTY, PROMPT = "results.csv", "topics.csv", "difficulty.csv", "prompt.csv"
FILES = (RESULTS, TOPICS, DIFFICULTY, PROMPT)
CHOSEN = "chosen"
HINT = "No {name} in {folder} yet. The offline evaluation writes it, next to app/: uv run evaluate.py"
NO_ROWS = "No {level} rows of this course in {name} yet. The offline evaluation writes them, next to app/: uv run evaluate.py"
# The model generations per file, and the model column that marks a run as the level's
LEVEL_FILES = {"S1": RESULTS, "S2": RESULTS, "C1": RESULTS, "T1": TOPICS, "D1": DIFFICULTY, "D2": DIFFICULTY, "P1": PROMPT}
NO_FOLDER = "No results folder. The offline evaluation writes one, next to app/: uv run evaluate.py"
COURSE_SCOPE = "all"  # evaluate.COURSE_SCOPE: a row over the course; a deck's rows read "deck <id>"
FOCUS_SCOPE = "focus "  # the share scale's marks: the share of the course's cards kept


def course_note(deck_ids: Mapping[str, str]) -> str:
    """The note that marks a row of this course: "decks 01 to 24" (evaluate.scope_note writes the same)."""
    ids = sorted(deck_ids.values())
    return f"decks {ids[0]} to {ids[-1]}" if ids else "no decks"


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


class Results:
    """The results files of a folder, read once: the rows whose notes name the course (`note`, course_note's); with
    no note, without a course table (the dashboard's demo alone), every row, so each level shows its newest run
    whatever course it names."""

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
        """Why a level has no numbers: the file is missing (how to make it), or no folder was given."""
        if self.folder is None:
            return NO_FOLDER
        return HINT.format(name=name, folder=self.folder)

    def run(self, level: str) -> Optional[list[dict]]:
        """The rows of the level's newest run (the run its headline numbers show); None without the file, empty
        without the level's rows."""
        rows = self.rows.get(LEVEL_FILES[level])
        return newest(rows, level) if rows is not None else None

    def no_rows(self, level: str) -> str:
        """Why a level has no numbers though its file is there."""
        return NO_ROWS.format(level=level, name=LEVEL_FILES[level])
