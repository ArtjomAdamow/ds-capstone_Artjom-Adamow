"""The launcher's mode question, in the package: start.py, the capstone's launcher exported with the package
(tools/export/export.md), asks which of the two modes starts; modules/helper/helper.py here asks nothing and
takes the modes' names from here.

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

The way between the two apps (step 28), shared by both launchers: in full mode the launcher starts the standalone
page as a process of its own (start_standalone: the launcher again, `--mode standalone --port 8051`, outside the
viewer's reloader, with the viewer's URL in its environment, VIEWER_URL) and puts the page's URL into the environment
(MODELS_URL), where the composition reads it for the viewer's Models button; once the viewer has stopped it waits for
the page's process (stop_standalone). The page draws a Sputnik link to the viewer's URL (standalone.main's viewer_url).
Two tabs, not a switch in place: one server for both would make the page a route of the viewer's app.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

FULL = "full"
STANDALONE = "standalone"
MODES = (FULL, STANDALONE)
VIEWER_PACKAGE = "project"  # the viewer's import name: sputnik in the capstone, the flashcards package here
VIEWER_RELEASE = "0.7.0"  # the sputnik release the glue needs (its hooks: the dialog, the links, the window, the card section, the focus, the Ask dialog, the episodes)
LAUNCH_QUESTION = "The optional interactive viewer Sputnik is available.\n Launch Sputnik? (y/n): "
MODE_OPTION = "--mode"
MODE_VARIABLE = "HELPER_MODE"  # the chosen mode, for the debug reloader's child
VIEWER_MISSING = (
    "--mode full needs the viewer (the package imported as 'project'): install the viewer extra "
    "(uv sync --extra viewer) or start with --mode standalone."
)
# The page next to the viewer: its own process on its own port (the viewer takes Dash's default, 8050)
VIEWER_PORT = 8050
STANDALONE_PORT = 8051
PORT_OPTION = "--port"
STANDALONE_GRACE = 10.0  # seconds the launcher waits for the page's process after the viewer has stopped
MODELS_URL = "HELPER_MODELS_URL"  # the page's URL for the viewer's Models button (glue.composition.models_links)
VIEWER_URL = "HELPER_VIEWER_URL"  # the viewer's URL for the page's Sputnik link (standalone.main's viewer_url)


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


# --- the page next to the viewer -------------------------------------------------------------------------------


def requested_port(arguments: Sequence[str]) -> Optional[int]:
    """The port named by `--port <n>` or `--port=<n>`, or None; anything but a number stops."""
    arguments = list(arguments)
    value = None
    for position, argument in enumerate(arguments):
        if argument == PORT_OPTION:
            value = arguments[position + 1] if position + 1 < len(arguments) else ""
        elif argument.startswith(PORT_OPTION + "="):
            value = argument.split("=", 1)[1]
    if value is None:
        return None
    if not value.isdigit():
        raise SystemExit(f"{PORT_OPTION} takes a number, not {value!r}")
    return int(value)


def standalone_url(port: int = STANDALONE_PORT) -> str:
    """Where the page's process serves: the link the viewer's Models button opens."""
    return f"http://127.0.0.1:{port}/"


def viewer_url(port: int = VIEWER_PORT) -> str:
    """Where the viewer serves: the link the page's Sputnik link opens."""
    return f"http://127.0.0.1:{port}/"


def start_standalone(
    launcher: Path, port: int = STANDALONE_PORT, popen: Callable = subprocess.Popen, environ: Mapping[str, str] = {},
    executable: str = sys.executable, output: Callable = print, viewer_port: int = VIEWER_PORT, ready: Optional[str] = None,
):
    """The page as a process of its own next to the viewer: `launcher` (this launcher's file) again with
    `--mode standalone --port <port>` on `executable`, outside the viewer's reloader (WERKZEUG_RUN_MAIN left out of
    its environment), with the viewer's URL in its environment (VIEWER_URL, for the page's Sputnik link) and, with
    `ready`, that variable set to "1" (helper.py's mark that skips its env check). Returns the process; the viewer
    serves in this process meanwhile."""
    env = {key: value for key, value in environ.items() if key != "WERKZEUG_RUN_MAIN"}
    env[VIEWER_URL] = viewer_url(viewer_port)
    if ready:
        env[ready] = "1"
    command = [executable, str(Path(launcher).resolve()), MODE_OPTION, STANDALONE, PORT_OPTION, str(port)]
    output(f"The models' page (app.standalone) starts in a process of its own: {standalone_url(port)}")
    return popen(command, env=env)


def stop_standalone(process, grace: float = STANDALONE_GRACE) -> int:
    """Once the viewer has stopped: Ctrl+C reached the page's process too, so wait for it to stop on its own; one
    still running after `grace` seconds is killed. Returns its exit code."""
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()
