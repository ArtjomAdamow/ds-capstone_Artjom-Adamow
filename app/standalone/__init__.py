"""The standalone page (roadmap step 22; step 21's mode (a)): the models on a page of their own, without the viewer,
in its theme. It reads the course table (app.course_table), not the course file, since the viewer's
parser is not there; it fits the course model and D1 at the start and stores nothing.

* `app`: `Page`, the fitted models and what the three views show of them; `create_app`, the Dash app with its one
  callback (the views' interaction only: select, Open, Back).
* `layout`: the layout, the map figure and the panels (the dot's ids or the dendrogram, D1's metrics).
* `assets/style.css`: the viewer's tokens, written anew; `assets/map_side.js`: the square map's side.

The launcher starts it with `main` (modules/helper/helper.py, the mode question of app.launch); it
imports nothing of the viewer and nothing of the glue.
"""

import time
from pathlib import Path

from ..course_table import load_table, missing_files
from .app import Page, create_app


def main(course_dir: Path, debug: bool = False) -> None:
    """Serve the page on the course table in `course_dir`; without the table, stop and name the missing files."""
    course_dir = Path(course_dir)
    missing = missing_files(course_dir)
    if missing:
        raise SystemExit(f"No course table in {course_dir}: {', '.join(path.name for path in missing)} missing")
    started = time.perf_counter()
    table = load_table(course_dir)
    page = Page(table)
    print(f"Fitted the models on {len(table.cards)} cards of {len(table.decks)} decks in {time.perf_counter() - started:.1f} s")
    create_app(page).run(debug=debug)
