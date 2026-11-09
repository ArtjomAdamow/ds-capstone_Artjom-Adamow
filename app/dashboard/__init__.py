"""The dashboard: the capstone's app once the viewer leaves it, one card at a time with what a learner sees of
the models, in place of sputnik; the models view is no part of the package. It reads the
course table (app.course_table) and the results files of the offline evaluation, fits the course model, D1 and P1
at the start, and keeps its own state file in `data/state/`; it imports nothing of the viewer and nothing of the
glue. Beside the course it reads deck 00's table (data/demo/dashboard/, docs/demo/) for its Demo; without the course
table it runs on deck 00 alone, empty outside the demo.

* `app`: `Dashboard`, the fitted models, the state and what the regions show; `DemoBoard`, deck 00's; `Boards`, the
  board the view names; `create_app`, the Dash app with its callback (the card's turn and a quiz's judgement,
  Related and its entries, Back, Demo, the request box and its entries, the tiles, the diagram's clicks) and
  Confirm's.
* `layout`: the regions: the top bar with the request box, Back and Demo, the diagram, the card with its options and its
  Related list, the slide row and the model's panel, its numbers on the left and its figure on the right.
* `panel_figures`: the panels' figures drawn here, each the whole course's: where the related cards lie (S), the
  decks on the sphere with the shown card's deck in gold (C), P1's judged requests on and off the course (P); T's
  histogram and D's matrix are the models view's.
* `state`: the state file `dashboard_state.json`: the read and explored flags, the shown card, the trail.
* `quiz`: a quiz's judgement as plain functions.
* `assets/style.css`: the theme's tokens and the type scale, written anew; `assets/keys.js`: Enter into the hidden
  input the callback reads; `assets/diagram.js`: a click in the diagram into another.

The launcher starts it with `main` (start.py, after the port check and the backup question; on Dash's default port
or the next free one, without the reloader and with no process next to it).
"""

import time
from pathlib import Path
from typing import Optional

from ..course_table import load_table, missing_files
from ..models.embeddings import app_vectoriser, uncached_copy
from .app import RESULTS_DIR, Dashboard, DemoBoard, create_app
from .state import STATE_NAME, DashboardState


def main(course_dir: Path, results_dir: Optional[Path] = None, state_dir: Optional[Path] = None, cache_dir: Optional[Path] = None,
         debug: bool = False, port: Optional[int] = None, demo_dir: Optional[Path] = None) -> None:
    """Serve the dashboard on the course table in `course_dir`, the results files in `results_dir` (by default
    `results/` next to `evaluate.py`) and the state file in `state_dir` (none: the state is kept in memory alone),
    on `port` (Dash's default without one), with the app's vectoriser (S2's encoder with the course's embeddings
    cached in `cache_dir`, or TF-IDF) at its own minimum; with deck 00's table in `demo_dir` its Demo (DemoBoard, the
    same kind of vectoriser without the cache). Without the course table it runs on deck 00 alone and neither reads
    nor writes the state file; without either table, or with a state file of another version, stop and say so."""
    course_dir = Path(course_dir)
    missing = missing_files(course_dir)
    demo_there = demo_dir is not None and not missing_files(demo_dir)
    if missing and not demo_there:
        raise SystemExit(f"No course table in {course_dir}: {', '.join(path.name for path in missing)} missing")
    started = time.perf_counter()
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    vectoriser = app_vectoriser(cache_dir)
    demo_vectoriser = uncached_copy(vectoriser)  # before the course's fit, which fits the vectoriser it is given
    board, state_path = None, None
    if not missing:
        table = load_table(course_dir)
        state_path = Path(state_dir) / STATE_NAME if state_dir is not None else None
        try:
            state = DashboardState.load(state_path, {card.id for card in table.cards})
        except ValueError as error:
            raise SystemExit(str(error))
        board = Dashboard(table, state, results_dir=results_dir, vectoriser=vectoriser, min_score=None)
    demo = DemoBoard(load_table(demo_dir), results_dir, demo_vectoriser, min_score=None) if demo_there else None
    fitted = [f"{len(board.table.cards)} cards of {len(board.table.decks)} decks"] if board is not None else []
    fitted += [f"deck 00's {len(demo.table.cards)} cards"] if demo is not None else []
    print(f"Fitted the models on {' and '.join(fitted)} in {time.perf_counter() - started:.1f} s; the results of {results_dir}; "
          f"the state in {state_path if state_path is not None else 'memory'}")
    create_app(board, demo).run(debug=debug, **({"port": port} if port is not None else {}))
