"""Start the guided learning app: the launcher of the capstone and of the flashcards workspace. It is developed with
the package app/ in the flashcards workspace and exported next to it into the capstone's root, where only the export
changes it. It needs no arguments: the folders and the commands are written below. From the capstone's root:

    uv run start.py

It searches for sputnik, the viewer (the capstone's `viewer` extra, imported as `project`). Found, it asks "Launch
Sputnik? (y/n)": y, or no input, starts the full app, n the standalone page. Not found, it starts the standalone page
without asking and prints the command that installs the viewer. In the flashcards workspace the viewer is the
workspace's own package (no sputnik), and the full app starts without the question. `--mode full|standalone|dashboard`
skips the question; the dashboard starts by the argument alone. Before the full app and before the dashboard it asks
the backup question (app.launch.prepare_state): with a state file in data/state/, y backs up user_state.json and
dashboard_state.json (_01.bak, _02.bak, ...) and starts fresh, n or no input keeps them; the episodes' log and the
model state, written at every start, stay.

    standalone  the models on a page of their own (app.standalone), on the course table in data/course/
                (cards.csv, decks.csv, topics.csv); nothing is stored; `--port` names its port
    full        sputnik's map with the models on from the start (app.glue; the Helper's switch shows the model's
                connections), on the course file data/topics.json with the card images in data/images/; the
                state goes into data/state/: the viewer's state file user_state.json, the derived course file
                helper_topics.json, the model state model_state.json and the episodes' log episodes.csv; on port
                8050, Dash's default, or the one `--port` names. The standalone page runs next to it as a process
                of its own on the next free port above it (this file again with `--mode standalone --port 8051`,
                app.launch.start_standalone): the viewer's top bar has the Models button, a link that opens the
                page in a new tab (where sputnik's top bar takes links, since 0.4.0), and the page a Sputnik link
                back to the viewer; Ctrl+C stops both
    dashboard   the dashboard (app.dashboard), one card at a time without the viewer, on the course table in
                data/course/ and the results next to evaluate.py; its state goes into data/state/dashboard_state.json;
                its Demo on deck 00's table in data/demo/dashboard/; without the course table it starts empty, with
                the Demo; on Dash's default port, without the reloader and with no process next to it; `--port`
                names its port

The ports are checked before anything serves (app.launch.choose_port): a port counts as held when a server answers
on it. The standalone page and the dashboard take the next free port from Dash's default and print the URL they
serve on; the full app stops when its port is held, naming the process that holds it (its PID and command line)
with the hint to stop it or to use `--port N`. A port named by `--port` and held stops every mode.

Both modes run the app's vectoriser (app.models.embeddings.app_vectoriser): S2's sentence encoder with the course's
embeddings cached in data/cache/embeddings/ where sentence-transformers is installed (the embeddings extra), TF-IDF
otherwise, with one line that says so.

data/ lies next to this file, or, in the flashcards workspace, in the folder above it (app.launch.data_dir). In the
capstone it is git-ignored: the course data comes from the flashcards workspace with a train, as the package and this
file do. A missing library, a missing
data file or a sputnik older than app.launch.VIEWER_RELEASE stops the start with the command that fixes it. Until
then it imports only the standard library and app.launch, so it starts on any interpreter.

The full app runs with Dash's debug reloader: the reloader's child (WERKZEUG_RUN_MAIN) runs this file again, asks
nothing and runs the mode its parent chose (MODE_VARIABLE); only the child fits the models and serves.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import uuid
from importlib import metadata
from pathlib import Path
from typing import Callable, MutableMapping, Optional, Sequence

from app.launch import (
    CACHE_FOLDER, DASHBOARD, FULL, MODE_VARIABLE, MODELS_URL, STANDALONE, STATE_FOLDER, VIEWER_DISTRIBUTION, VIEWER_PORT,
    VIEWER_RELEASE, VIEWER_URL, choose_mode, choose_port, data_dir, mode_of_child, prepare_state, requested_port,
    standalone_url, start_standalone, stop_standalone, viewer_available, viewer_here,
)

ROOT = Path(__file__).resolve().parent  # start.py sits next to the package: the capstone's root
DATA_DIR = data_dir(ROOT)  # data/ next to it, or in the folder above it (the flashcards workspace's root)
COURSE_FILE = DATA_DIR / "topics.json"  # the course file, version 3: the full app's input
TABLE_DIR = DATA_DIR / "course"  # the course table: the standalone page's input
DEMO_TABLE_DIR = DATA_DIR / "demo" / "dashboard"  # deck 00's table, 00-deck-dashboard: the dashboard's Demo
RESULTS_DIR = ROOT / "results"  # the offline evaluation's rows, next to evaluate.py: the standalone page's panels read them
STATE_FILE = DATA_DIR / STATE_FOLDER / "user_state.json"  # the viewer's state file; the derived course file, the
# model state and the episodes' log go next to it (app.glue.composition)
CACHE_DIR = DATA_DIR / CACHE_FOLDER  # the embeddings' cache of S2 (app.models.embeddings), a derived file, no state
RELOADER = True  # the full app restarts on code changes; the reloader's child fits the models and serves

# The commands it prints
INSTALL = "uv sync --all-groups"
INSTALL_VIEWER = "uv sync --all-groups --extra viewer"
START = "uv run start.py"
DATA_HINT = (
    "The course data is not in the repository: it comes into data/ with a train from the flashcards workspace."
)

LIBRARIES = ("numpy", "scipy", "sklearn", "dash", "plotly")  # what the models and the page import


def missing_libraries(find_spec: Callable = importlib.util.find_spec) -> list[str]:
    """The libraries of the models and the page that do not import here (none is imported)."""
    missing = []
    for name in LIBRARIES:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def viewer_too_old(version: Callable = metadata.version) -> Optional[str]:
    """The installed sputnik's version when it is older than VIEWER_RELEASE, else None (also where no distribution
    named sputnik is installed: the flashcards workspace's viewer)."""
    try:
        installed = version(VIEWER_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None
    return installed if _release(installed) < _release(VIEWER_RELEASE) else None


def run_standalone(table_dir: Path = TABLE_DIR, cache_dir: Path = CACHE_DIR, port: Optional[int] = None, viewer_url: Optional[str] = None) -> None:
    """The standalone page on the course table, with the app's vectoriser and the embeddings' cache, on `port`
    (Dash's default without one), with the Sputnik link to `viewer_url` (the viewer served next to it); without the
    table, stop with the data hint."""
    from app.course_table import missing_files

    missing = missing_files(table_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise SystemExit(f"No course table in {table_dir} ({names} missing). {DATA_HINT}")
    from app.standalone import main as standalone_main

    standalone_main(table_dir, RESULTS_DIR, cache_dir=cache_dir, port=port, viewer_url=viewer_url)


def run_dashboard(table_dir: Path = TABLE_DIR, results_dir: Path = RESULTS_DIR, state_dir: Path = STATE_FILE.parent,
                  cache_dir: Path = CACHE_DIR, port: Optional[int] = None, demo_dir: Path = DEMO_TABLE_DIR) -> None:
    """The dashboard on the course table, with the results next to evaluate.py, its state file in the state folder
    (next to the viewer's), the app's vectoriser and the embeddings' cache, and its Demo on deck 00's table, on
    `port` (Dash's default without one); without the course table it starts empty with the Demo, and without either
    table it stops with the data hint."""
    from app.course_table import missing_files

    missing = missing_files(table_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        if missing_files(demo_dir):
            raise SystemExit(f"No course table in {table_dir} ({names} missing). {DATA_HINT}")
        print(f"No course table in {table_dir} ({names} missing): the dashboard starts empty, with its Demo. {DATA_HINT}")
    from app.dashboard import main as dashboard_main

    dashboard_main(table_dir, results_dir, state_dir, cache_dir=cache_dir, port=port, demo_dir=demo_dir)


def run_full(
    root: Path = DATA_DIR.parent, course_file: Path = COURSE_FILE, state_file: Path = STATE_FILE,
    cache_dir: Path = CACHE_DIR, environ: MutableMapping[str, str] = os.environ, port: int = VIEWER_PORT,
) -> None:
    """Sputnik's app with the models on from the start, served with the debug reloader on `port`, on `root`, the data folder's parent (the
    images under data/images/). The reloader's parent composes the viewer's app on the raw course file without the
    models and stores nothing (its random positions would read as stored ones); the child fits the models, writes the
    derived course file and saves the state at the start. The Models button links the page's URL where the
    environment names it (MODELS_URL, set by main) and the viewer's top bar takes links."""
    if not course_file.is_file():
        raise SystemExit(f"No course file at {course_file}. {DATA_HINT}")
    too_old = viewer_too_old()
    if too_old is not None:
        raise SystemExit(
            f"sputnik {too_old} is installed, the app needs {VIEWER_RELEASE} or newer: {INSTALL_VIEWER}, then {START}"
        )
    from app.glue.composition import compose_model, models_links, serving_process
    from app.models.embeddings import app_vectoriser

    serving = serving_process(RELOADER)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # The app's vectoriser at its own minimum, made in the serving process only (the parent fits nothing)
    composition = compose_model(root, uuid.uuid4().hex, course_file, state_file, fit=serving, links=models_links(environ),
                                vectoriser=app_vectoriser(cache_dir) if serving else None, min_score=None)
    if serving:
        # The startup save writes everything: the checked decks and topics, the shown cards, the positions,
        # connections and groups, and the model state
        composition.user_state.save()
    composition.dash_app.run(debug=True, use_reloader=RELOADER, port=port)


def main(ask: Callable = input, environ: MutableMapping[str, str] = os.environ, arguments: Optional[Sequence[str]] = None) -> None:
    """The start: the mode (the question, or `--mode` among `arguments`, sys.argv's by default), the port (the
    port check, app.launch.choose_port: the page and the dashboard move to the next free port, the full app stops
    when its port is held), then the page alone, the dashboard after the backup question, or, after the backup
    question, the full app with the page as a process of its own next to it on the next free port above the viewer's
    (the reloader's child asks nothing, checks no port, since its parent holds it, and starts no second page; the
    page's own process runs this file with `--mode standalone --port`)."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    missing = missing_libraries()
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not installed. Install the env: {INSTALL} (with sputnik: {INSTALL_VIEWER}), "
            f"then start with: {START}"
        )
    found = viewer_available()
    child = environ.get("WERKZEUG_RUN_MAIN") == "true"
    port = requested_port(arguments)
    if child:
        mode = mode_of_child(environ, found)  # the reloader's child asks nothing
        port = VIEWER_PORT if port is None else port  # the port its parent checked
    else:
        if not found:
            print(f"Sputnik not found: starting the standalone page. For the full app: {INSTALL_VIEWER}")
        mode = choose_mode(found, ask, arguments, here=viewer_here())
        environ[MODE_VARIABLE] = mode  # the reloader's child runs the same mode
        port = choose_port(port, moves=mode != FULL)  # before anything serves or asks
        if mode in (FULL, DASHBOARD):
            prepare_state(STATE_FILE.parent, ask)
    if mode == STANDALONE:
        run_standalone(port=port, viewer_url=environ.get(VIEWER_URL) or None)
        return
    if mode == DASHBOARD:
        run_dashboard(port=port)
        return
    page = None
    if not child:  # the page next to the viewer, started once; the child inherits the URL
        page_port = choose_port(None, moves=True, start=port + 1)
        page = start_standalone(Path(__file__), page_port, environ=environ, viewer_port=port)
        environ[MODELS_URL] = standalone_url(page_port)
    try:
        run_full(environ=environ, port=port)
    finally:
        if page is not None:
            stop_standalone(page)


if __name__ == "__main__":
    main()
