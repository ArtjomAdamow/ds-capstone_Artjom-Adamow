"""Start the guided learning app: the dashboard, one card at a time with its related cards, a request box as the way
in and the models' numbers in a row of tiles. The capstone's launcher, next to the package app/ in the capstone's
root. It needs no arguments: the folders and the commands are written below. From the capstone's root:

    uv run start.py

It asks one question, the backup question (app.launch.prepare_state): with the dashboard's state file in
data/state/, y backs up dashboard_state.json (_01.bak, _02.bak, ...) and starts fresh, n or no input keeps it.

The dashboard (app.dashboard) runs on the course table in data/course/ (cards.csv, decks.csv, topics.csv) and the
results next to evaluate.py; its state goes into data/state/dashboard_state.json; its Demo on deck 00's table in
data/demo/dashboard/. Without the course table it starts empty, with the Demo. It serves without the reloader and with
no process next to it.

The port is checked before anything serves or asks (app.launch.choose_port): a port counts as held when a server
answers on it. The dashboard takes the next free port from Dash's default and prints the URL it serves on; a port
named by `--port N` and held stops it.

The app's vectoriser runs the models (models.embeddings.app_vectoriser): S2's sentence encoder with the course's
embeddings cached in data/cache/embeddings/ where sentence-transformers is installed (the embeddings extra), TF-IDF
otherwise, with one line that says so.

data/ lies next to this file, or in the folder above it where there is none (app.launch.data_dir). It is
git-ignored but for deck 00: the course data is not in the repository. A missing library stops the start with the
command that fixes it, both tables missing with the data hint.
Until then it imports only the standard library and app.launch, so it starts on any interpreter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from app.launch import CACHE_FOLDER, STATE_FOLDER, choose_port, data_dir, prepare_state, requested_port

ROOT = Path(__file__).resolve().parent  # start.py sits next to the package: the capstone's root
DATA_DIR = data_dir(ROOT)  # data/ next to it, or in the folder above it where there is none
TABLE_DIR = DATA_DIR / "course"  # the course table: the dashboard's input
DEMO_TABLE_DIR = DATA_DIR / "demo" / "dashboard"  # deck 00's table, 00-deck-dashboard: the dashboard's Demo
RESULTS_DIR = ROOT / "results"  # the offline evaluation's rows, next to evaluate.py: the dashboard's tiles read them
STATE_DIR = DATA_DIR / STATE_FOLDER  # the dashboard's state file dashboard_state.json
CACHE_DIR = DATA_DIR / CACHE_FOLDER  # the embeddings' cache of S2 (models.embeddings), a derived file, no state

# The commands it prints
INSTALL = "uv sync --all-groups"
START = "uv run start.py"
DATA_HINT = (
    "The course data is not in the repository: its table belongs in data/course/, deck 00's in data/demo/dashboard/."
)

LIBRARIES = ("numpy", "scipy", "sklearn", "dash", "plotly")  # what the models and the dashboard import


def missing_libraries(find_spec: Callable = importlib.util.find_spec) -> list[str]:
    """The libraries of the models and the dashboard that do not import here (none is imported)."""
    missing = []
    for name in LIBRARIES:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def run_dashboard(table_dir: Path = TABLE_DIR, results_dir: Path = RESULTS_DIR, state_dir: Path = STATE_DIR,
                  cache_dir: Path = CACHE_DIR, port: Optional[int] = None, demo_dir: Path = DEMO_TABLE_DIR) -> None:
    """The dashboard on the course table, with the results next to evaluate.py, its state file in the state folder,
    the app's vectoriser and the embeddings' cache, and its Demo on deck 00's table, on `port` (Dash's default without
    one); without the course table it starts empty with the Demo, and without either table it stops with the data
    hint."""
    from app.course_table import missing_files

    missing = missing_files(table_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        if missing_files(demo_dir):
            raise SystemExit(f"No course table in {table_dir} ({names} missing). {DATA_HINT}")
        print(f"No course table in {table_dir} ({names} missing): the dashboard starts empty, with its Demo. {DATA_HINT}")
    from app.dashboard import main as dashboard_main

    dashboard_main(table_dir, results_dir, state_dir, cache_dir=cache_dir, port=port, demo_dir=demo_dir)


def main(ask: Callable = input, arguments: Optional[Sequence[str]] = None) -> None:
    """The start: the libraries, the port (the port check, app.launch.choose_port: the next free one from Dash's
    default, or the one `--port` among `arguments`, sys.argv's by default, names), the backup question, then the
    dashboard."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    missing = missing_libraries()
    if missing:
        raise SystemExit(f"{', '.join(missing)} not installed. Install the env: {INSTALL}, then start with: {START}")
    port = choose_port(requested_port(arguments), moves=True)  # before anything serves or asks
    prepare_state(STATE_DIR, ask)
    run_dashboard(port=port)


if __name__ == "__main__":
    main()
