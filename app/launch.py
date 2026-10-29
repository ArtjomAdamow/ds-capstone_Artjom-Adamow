"""The launcher's mode question (roadmap step 22), in the package, so both launchers ask it: modules/helper/helper.py
here and start.py, the capstone's launcher exported with the package (docs/draft/helper/protocol_export.md): which
of the two modes starts.

* full (FULL): the viewer with the Helper button (the flashcards package, sputnik in the capstone);
* standalone (STANDALONE): the models on their own page (app.standalone), without the viewer.

The viewer is there when its package imports (viewer_available: `project`, its import name in both places). Without
it the mode is standalone and nothing is asked; with it the launcher asks LAUNCH_QUESTION: n starts standalone,
anything else the full app. Without input (a scripted start, the tests) the mode is the full app. `--mode full` or
`--mode standalone` among the arguments skips the question; `--mode full` without the viewer stops with the hint to
install the viewer extra. The launcher passes the chosen mode on to the debug reloader's child in the environment
(MODE_VARIABLE), which asks nothing (mode_of_child). start.py passes no arguments: it always asks when the viewer
is found.

VIEWER_RELEASE is the sputnik release whose hooks the glue uses: an export names it, the capstone pins its tag, and
start.py refuses the full app on an older sputnik.
"""

import importlib.util
from typing import Callable, Mapping, Optional, Sequence

FULL = "full"
STANDALONE = "standalone"
MODES = (FULL, STANDALONE)
VIEWER_PACKAGE = "project"  # the viewer's import name: sputnik in the capstone, the flashcards package here
VIEWER_RELEASE = "0.3.0"  # the sputnik release the glue needs (its hooks: the module action's dialog, origins, Display)
LAUNCH_QUESTION = "Launch Sputnik? (y/n): "
MODE_OPTION = "--mode"
MODE_VARIABLE = "HELPER_MODE"  # the chosen mode, for the debug reloader's child
VIEWER_MISSING = (
    "--mode full needs the viewer (the package imported as 'project'): install the viewer extra "
    "(uv sync --extra viewer) or start with --mode standalone."
)


def viewer_available(find_spec: Callable = importlib.util.find_spec) -> bool:
    """Whether the viewer's package imports (it is not imported here)."""
    try:
        return find_spec(VIEWER_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def requested_mode(arguments: Sequence[str]) -> Optional[str]:
    """The mode named by `--mode <mode>` or `--mode=<mode>`, or None; another value stops with the choices."""
    arguments = list(arguments)
    value = None
    for position, argument in enumerate(arguments):
        if argument == MODE_OPTION:
            value = arguments[position + 1] if position + 1 < len(arguments) else ""
        elif argument.startswith(MODE_OPTION + "="):
            value = argument.split("=", 1)[1]
    if value is not None and value not in MODES:
        raise SystemExit(f"{MODE_OPTION} takes {' or '.join(MODES)}, not {value!r}")
    return value


def choose_mode(found: bool, ask: Callable[[str], str] = input, arguments: Sequence[str] = ()) -> str:
    """FULL or STANDALONE (the module docstring). `found` says whether the viewer is there (viewer_available);
    `ask` is asked the question only when the viewer is found and no mode is requested."""
    requested = requested_mode(arguments)
    if requested == FULL and not found:
        raise SystemExit(VIEWER_MISSING)
    if requested is not None:
        return requested
    if not found:
        return STANDALONE
    try:
        answer = ask(LAUNCH_QUESTION).strip().lower()
    except EOFError:
        return FULL
    return STANDALONE if answer in ("n", "no") else FULL


def mode_of_child(environ: Mapping[str, str], found: bool) -> str:
    """The mode of the debug reloader's child: the one its parent chose, else what a start without input chooses."""
    mode = environ.get(MODE_VARIABLE)
    if mode in MODES:
        return mode
    return FULL if found else STANDALONE
