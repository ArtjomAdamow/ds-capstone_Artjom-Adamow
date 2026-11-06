"""The standalone page: the models on a page of their own, without
the viewer, in its theme. It reads the course table (app.course_table), not the course file, since the viewer's
parser is not there, and the results files of the offline evaluation; it fits the course model and D1 at the start
and stores nothing.

* `app`: `Page`, the fitted models and what the three views, the map's layers and the right column's regions show
  of them; `create_app`, the Dash app with its one callback (select, Open, Back, the layers' check boxes, the tabs,
  Details, a results row's description, P's request).
* `layout`: the two columns, the map figure drawn by layers with the legend's strip, the tab bar, the generation
  tabs, the section, the tag strip, the figure slot and Details.
* `headlines`: the section's numbers from the results rows, with the verdict rule and its explanation in words.
* `results`: the results files (the newest run per level, the tables with the app's pick marked, the glossary).
* `assets/style.css`: the viewer's tokens, written anew, and the type scale; `assets/map_side.js`: the map's side,
  the height under the top bar; `assets/panel_rows.js`: a generation tab's, a results row's or a target's click
  into the hidden input the callback reads.

The launchers start it with `main` (start.py after the mode question of app.launch; both launchers as a process
next to the viewer in full mode, on a port of its own, with the viewer's URL for the page's Sputnik link in the top
bar); it imports nothing of the viewer and nothing of the glue: the link is a URL.
"""

import time
from pathlib import Path
from typing import Optional

from ..course_table import load_table, missing_files
from ..models.embeddings import app_vectoriser
from ..models.zero_shot import cached_tags
from .app import RESULTS_DIR, Page, create_app


def main(course_dir: Path, results_dir: Optional[Path] = None, debug: bool = False, port: Optional[int] = None,
         cache_dir: Optional[Path] = None, viewer_url: Optional[str] = None) -> None:
    """Serve the page on the course table in `course_dir` and the results files in `results_dir` (by default
    `results/` next to `evaluate.py`), on `port` (Dash's default without one), with the app's vectoriser
    (app.models.embeddings.app_vectoriser: S2's encoder with the course's embeddings cached in `cache_dir`, or
    TF-IDF) at its own minimum, and D2's tags where the same cache holds them (zero_shot.cached_tags: no model is
    loaded); with `viewer_url` the top bar's Sputnik link to the viewer served next to the page; without the table,
    stop and name the missing files."""
    course_dir = Path(course_dir)
    missing = missing_files(course_dir)
    if missing:
        raise SystemExit(f"No course table in {course_dir}: {', '.join(path.name for path in missing)} missing")
    started = time.perf_counter()
    table = load_table(course_dir)
    tags = cached_tags(table.cards, cache_dir)
    page = Page(table, results_dir=Path(results_dir) if results_dir is not None else RESULTS_DIR,
                vectoriser=app_vectoriser(cache_dir), min_score=None, tags=tags.scores if tags is not None else None)
    print(f"Fitted the models on {len(table.cards)} cards of {len(table.decks)} decks in {time.perf_counter() - started:.1f} s; "
          f"the results of {page.results.folder}")
    create_app(page, viewer_url).run(debug=debug, **({"port": port} if port is not None else {}))
