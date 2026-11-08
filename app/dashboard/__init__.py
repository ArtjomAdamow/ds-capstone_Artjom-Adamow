"""The dashboard: the capstone's app once the viewer leaves it, one card at a time with what a learner sees of
the models, in place of sputnik; the models view (app.standalone) stays next to it until its move. It reads the
course table (app.course_table) and the results files of the offline evaluation, fits the course model, D1 and P1
at the start, and keeps its own state file in `data/state/`; it imports nothing of the viewer and nothing of the
glue.

* `app`: `Dashboard`, the fitted models, the state and what the regions show; `create_app`, the Dash app with its
  one callback (the card's turn and a quiz's judgement, Related and its entries, Back, the request box and its
  entries, the tiles).
* `layout`: the regions: the top bar with the request box and Back, the diagram, the card with its options and its
  Related list, the slide row and the model's panel.
* `state`: the state file `dashboard_state.json`: the read and explored flags, the shown card, the trail.
* `quiz`: a quiz's judgement as plain functions.
* `assets/style.css`: the theme's tokens and the type scale, written anew; `assets/keys.js`: Enter into the hidden
  input the callback reads.

The launcher starts it with `main` (start.py's third mode, `--mode dashboard`, after the backup question; on Dash's
default port, without the reloader and with no process next to it).
"""

import time
from pathlib import Path
from typing import Optional

from ..course_table import load_table, missing_files
from ..models.embeddings import app_vectoriser
from .app import RESULTS_DIR, Dashboard, create_app
from .state import STATE_NAME, DashboardState


def main(course_dir: Path, results_dir: Optional[Path] = None, state_dir: Optional[Path] = None, cache_dir: Optional[Path] = None,
         debug: bool = False, port: Optional[int] = None) -> None:
    """Serve the dashboard on the course table in `course_dir`, the results files in `results_dir` (by default
    `results/` next to `evaluate.py`) and the state file in `state_dir` (none: the state is kept in memory alone),
    on `port` (Dash's default without one), with the app's vectoriser (S2's encoder with the course's embeddings
    cached in `cache_dir`, or TF-IDF) at its own minimum; without the table, or with a state file of another
    version, stop and say so."""
    course_dir = Path(course_dir)
    missing = missing_files(course_dir)
    if missing:
        raise SystemExit(f"No course table in {course_dir}: {', '.join(path.name for path in missing)} missing")
    started = time.perf_counter()
    table = load_table(course_dir)
    state_path = Path(state_dir) / STATE_NAME if state_dir is not None else None
    try:
        state = DashboardState.load(state_path, {card.id for card in table.cards})
    except ValueError as error:
        raise SystemExit(str(error))
    board = Dashboard(table, state, results_dir=Path(results_dir) if results_dir is not None else RESULTS_DIR,
                      vectoriser=app_vectoriser(cache_dir), min_score=None)
    print(f"Fitted the models on {len(table.cards)} cards of {len(table.decks)} decks in {time.perf_counter() - started:.1f} s; "
          f"the results of {board.results.folder}; the state in {state_path if state_path is not None else 'memory'}")
    create_app(board).run(debug=debug, **({"port": port} if port is not None else {}))
