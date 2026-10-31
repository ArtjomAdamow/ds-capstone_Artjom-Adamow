"""Start the guided learning app: the capstone's launcher. It is developed with the package app/ in the flashcards
workspace (modules/helper/start.py) and exported next to it into the capstone's root, where only the export changes
it (tools/export/export.md). It takes no arguments: the folders and the commands are written below.
From the capstone's root:

    uv run start.py

It searches for sputnik, the viewer (the capstone's `viewer` extra, imported as `project`). Found, it asks "Launch
Sputnik? (y/n)": y, or no input, starts the full app, n the standalone page. Not found, it starts the standalone page
without asking and prints the command that installs the viewer.

    standalone  the models on a page of their own (app.standalone), on the course table in data/course/
                (cards.csv, decks.csv, topics.csv); nothing is stored
    full        sputnik's map with the models behind the Helper button (app.glue), on the course file
                data/topics.json with the card images in data/images/; the state goes into data/state/: the
                viewer's state file user_state.json, the derived course file helper_topics.json and the model
                state model_state.json

data/ is git-ignored: the course data comes from the flashcards workspace with the helper's migration (its
tools/train.py), as the package and this file do. A missing library, a missing
data file or a sputnik older than app.launch.VIEWER_RELEASE stops the start with the command that fixes it. Until
then it imports only the standard library and app.launch, so it starts on any interpreter.

The full app runs with Dash's debug reloader: the reloader's child (WERKZEUG_RUN_MAIN) runs this file again, asks
nothing and runs the mode its parent chose (MODE_VARIABLE); only the child fits the models and serves.
"""

from __future__ import annotations

import importlib.util
import os
import re
import uuid
from importlib import metadata
from pathlib import Path
from typing import Callable, MutableMapping, Optional

from app.launch import (
    MODE_VARIABLE, STANDALONE, VIEWER_RELEASE, choose_mode, mode_of_child, viewer_available,
)

ROOT = Path(__file__).resolve().parent  # the capstone's root: start.py sits next to the package
DATA_DIR = ROOT / "data"
COURSE_FILE = DATA_DIR / "topics.json"  # the course file, version 3: the full app's input
TABLE_DIR = DATA_DIR / "course"  # the course table: the standalone page's input
STATE_FILE = DATA_DIR / "state" / "user_state.json"  # the viewer's state file; the derived course file and the
# model state go next to it (app.glue.composition)
RELOADER = True  # the full app restarts on code changes; the reloader's child fits the models and serves

# The commands it prints
INSTALL = "uv sync --all-groups"
INSTALL_VIEWER = "uv sync --all-groups --extra viewer"
START = "uv run start.py"
DATA_HINT = (
    "The course data is not in the repository: it comes into data/ with the helper's migration from the flashcards "
    "workspace (tools/train.py there)."
)

LIBRARIES = ("numpy", "scipy", "sklearn", "dash", "plotly")  # what the models and the page import
VIEWER_DISTRIBUTION = "sputnik"  # the viewer's installed name (here the flashcards package has another one)


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


def run_standalone(table_dir: Path = TABLE_DIR) -> None:
    """The standalone page on the course table; without the table, stop with the data hint."""
    from app.course_table import missing_files

    missing = missing_files(table_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise SystemExit(f"No course table in {table_dir} ({names} missing). {DATA_HINT}")
    from app.standalone import main as standalone_main

    standalone_main(table_dir)


def run_full(root: Path = ROOT, course_file: Path = COURSE_FILE, state_file: Path = STATE_FILE) -> None:
    """Sputnik's app with the Helper button, served with the debug reloader. The reloader's parent composes the
    viewer's app on the raw course file without the models and stores nothing (its random positions would read as
    stored ones); the child fits the models, writes the derived course file and saves the state at the start."""
    if not course_file.is_file():
        raise SystemExit(f"No course file at {course_file}. {DATA_HINT}")
    too_old = viewer_too_old()
    if too_old is not None:
        raise SystemExit(
            f"sputnik {too_old} is installed, the app needs {VIEWER_RELEASE} or newer: {INSTALL_VIEWER}, then {START}"
        )
    from app.glue.composition import compose_model, serving_process

    serving = serving_process(RELOADER)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    composition = compose_model(root, uuid.uuid4().hex, course_file, state_file, fit=serving)
    if serving:
        # The startup save writes everything: the checked decks and topics, the shown cards, the positions,
        # connections and groups, and the model state
        composition.user_state.save()
    composition.dash_app.run(debug=True, use_reloader=RELOADER)


def main(ask: Callable = input, environ: MutableMapping[str, str] = os.environ) -> None:
    missing = missing_libraries()
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not installed. Install the env: {INSTALL} (with sputnik: {INSTALL_VIEWER}), "
            f"then start with: {START}"
        )
    found = viewer_available()
    if environ.get("WERKZEUG_RUN_MAIN") == "true":
        mode = mode_of_child(environ, found)  # the reloader's child asks nothing
    else:
        if not found:
            print(f"Sputnik not found: starting the standalone page. For the full app: {INSTALL_VIEWER}")
        mode = choose_mode(found, ask)
        environ[MODE_VARIABLE] = mode  # the reloader's child runs the same mode
    if mode == STANDALONE:
        run_standalone()
    else:
        run_full()


if __name__ == "__main__":
    main()
