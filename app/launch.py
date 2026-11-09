"""The launcher's mode question, in the package: start.py, the launcher exported with the package, asks which of
the two modes starts, full or standalone; the third, the dashboard, is started by `--mode dashboard` alone. The
module launcher of the flashcards workspace asks nothing and takes the modes' names from here.

* full (FULL): the viewer with the models on from the start (the flashcards package, sputnik in the capstone);
* standalone (STANDALONE): the models on their own page (app.standalone), without the viewer;
* dashboard (DASHBOARD): the dashboard (app.dashboard), one card at a time without the viewer, on its own state file
  next to the viewer's; the question never names it.

The viewer is there when its package imports (viewer_available: `project`, its import name in both places). Without
it the mode is standalone and nothing is asked; with sputnik the launcher asks LAUNCH_QUESTION: n starts standalone,
anything else the full app. Without input (a scripted start, the tests) the mode is the full app. With the flashcards
workspace's own viewer (viewer_here: its distribution, no sputnik) the full app starts without the question.
`--mode full`, `--mode standalone` or `--mode dashboard` among the arguments skips the question; `--mode full` without the viewer stops
with the hint to install the viewer extra. The launcher passes the chosen mode on to the debug reloader's child in the
environment (MODE_VARIABLE), which asks nothing (mode_of_child). start.py passes no arguments.

The folders (data_dir): the data folder lies next to the launcher (the capstone), else in the folder above it (the
flashcards workspace's root): the course file, the course table and the images, the state in data/state/ and the
embeddings' cache in data/cache/.

The backup question (prepare_state), asked before the full app and before the dashboard by the launcher's first
process, not by the reloader's child: with a state file in data/state/, y backs up the viewer's state file and the
dashboard's state file (STATE_NAMES; backup_path: user_state.json_01.bak, _02.bak, ...) and starts fresh, n or no
input keeps them. The episodes' log is never touched, nor the model state (since roadmap step 33 of the helper written
at every start and holding nothing of the learner's).

VIEWER_RELEASE is the sputnik release whose hooks the glue uses: an export names it, the capstone pins its tag, and
start.py refuses the full app on an older sputnik.

The way between the two apps (step 28), shared by both launchers: in full mode the launcher starts the standalone
page as a process of its own (start_standalone: the launcher again, `--mode standalone --port 8051`, outside the
viewer's reloader, with the viewer's URL in its environment, VIEWER_URL) and puts the page's URL into the environment
(MODELS_URL), where the composition reads it for the viewer's Models button; once the viewer has stopped it waits for
the page's process (stop_standalone). The page draws a Sputnik link to the viewer's URL (standalone.main's viewer_url).
Two tabs, not a switch in place: one server for both would make the page a route of the viewer's app.

The ports (choose_port), taken by the launcher's first process before anything serves: on Windows a second server
binds a port a live one holds (Werkzeug's SO_REUSEADDR) and the older process keeps getting the requests, so a port
counts as held when a connect to it is answered (port_in_use), never by a bind. The dashboard and the standalone page
take the next free port from Dash's default (free_port) and print the URL they serve on; the full app stops with the
hint (port_hint: the PID and the command line of the process listening there, port_holder) when its port is held,
since the page's URL and the viewer's name the two ports, and its page takes the next free port above it.
`--port N` names the port and stops when it is held.
"""

import importlib.util
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

FULL = "full"
STANDALONE = "standalone"
DASHBOARD = "dashboard"
MODES = (FULL, STANDALONE, DASHBOARD)
VIEWER_PACKAGE = "project"  # the viewer's import name: sputnik in the capstone, the flashcards package in the workspace
VIEWER_DISTRIBUTION = "sputnik"  # the viewer's installed name in the capstone
WORKSPACE_DISTRIBUTION = "spatial-flashcards"  # the viewer's installed name in the flashcards workspace (editable)
VIEWER_RELEASE = "0.10.0"  # the sputnik release the glue needs (its hooks: the links, the window, the card section, the Ask dialog, the episodes, the model state; the learner's confidence by that name; step 33's closed start and explored cards, the router's checked topics and cards, its presets and its All decks flag, the module's switch and Display's)
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
        raise SystemExit(f"{MODE_OPTION} takes {', '.join(MODES[:-1])} or {MODES[-1]}, not {value!r}")
    return value


def viewer_here(version: Callable = metadata.version) -> bool:
    """Whether the viewer is the flashcards workspace's own package (WORKSPACE_DISTRIBUTION installed, no sputnik):
    there the full app starts without the question."""

    def installed(name: str) -> bool:
        try:
            version(name)
        except metadata.PackageNotFoundError:
            return False
        return True

    return installed(WORKSPACE_DISTRIBUTION) and not installed(VIEWER_DISTRIBUTION)


def choose_mode(found: bool, ask: Callable[[str], str] = input, arguments: Sequence[str] = (), here: bool = False) -> str:
    """FULL, STANDALONE or DASHBOARD (the module docstring). `found` says whether the viewer is there (viewer_available),
    `here` whether it is the workspace's own (viewer_here); `ask` is asked the question only when sputnik is found and
    no mode is requested; DASHBOARD comes by the argument alone."""
    requested = requested_mode(arguments)
    if requested == FULL and not found:
        raise SystemExit(VIEWER_MISSING)
    if requested is not None:
        return requested
    if not found:
        return STANDALONE
    if here:
        return FULL
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


def app_url(port: int) -> str:
    """Where an app of the launcher serves on the port."""
    return f"http://127.0.0.1:{port}/"


def standalone_url(port: int = STANDALONE_PORT) -> str:
    """Where the page's process serves: the link the viewer's Models button opens."""
    return app_url(port)


def viewer_url(port: int = VIEWER_PORT) -> str:
    """Where the viewer serves: the link the page's Sputnik link opens."""
    return app_url(port)


def start_standalone(
    launcher: Path, port: int = STANDALONE_PORT, popen: Callable = subprocess.Popen, environ: Mapping[str, str] = {},
    executable: str = sys.executable, output: Callable = print, viewer_port: int = VIEWER_PORT, ready: Optional[str] = None,
):
    """The page as a process of its own next to the viewer: `launcher` (this launcher's file) again with
    `--mode standalone --port <port>` on `executable`, outside the viewer's reloader (WERKZEUG_RUN_MAIN left out of
    its environment), with the viewer's URL in its environment (VIEWER_URL, for the page's Sputnik link) and, with
    `ready`, that variable set to "1" (a module launcher's mark that skips its env check). Returns the process; the viewer
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


# --- the ports ------------------------------------------------------------------------------------------------

HOST = "127.0.0.1"  # where Dash serves by default
PORT_PROBE = 0.25  # seconds a connect waits for an answer: a closed port on Windows answers only after two seconds
PORT_SCAN = 100  # ports free_port tries upwards
HOLDER_TIMEOUT = 10.0  # seconds the lookup of a port's holder may take


def port_in_use(port: int, host: str = HOST, timeout: float = PORT_PROBE) -> bool:
    """Whether a server answers on the port: a connect, since on Windows a bind succeeds on a held port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def free_port(start: int, in_use: Callable[[int], bool] = port_in_use, scan: int = PORT_SCAN) -> int:
    """The first port from `start` upwards that no server answers on; none among `scan` ports stops."""
    for port in range(start, start + scan):
        if not in_use(port):
            return port
    raise SystemExit(f"No free port from {start} to {start + scan - 1}")


def _output(command: Sequence[str], run: Callable) -> Optional[str]:
    try:
        result = run(list(command), capture_output=True, text=True, errors="replace", timeout=HOLDER_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def port_holder(port: int, run: Callable = subprocess.run, platform: str = sys.platform) -> Optional[tuple[int, Optional[str]]]:
    """The PID and the command line of the process listening on the port, or None. On Windows the PID from
    `netstat -ano` (a TCP row with the port local and no remote address: the state's name is the system language's)
    and the command line from the process list (Win32_Process through PowerShell); elsewhere the PID from `lsof`
    where it is there and the command line from `ps`. The command line is None where it is not found."""
    if platform == "win32":
        listing = _output(["netstat", "-ano", "-p", "TCP"], run) or ""
        pid = None
        for row in listing.splitlines():
            parts = row.split()
            if len(parts) == 5 and parts[0] == "TCP" and parts[1].rsplit(":", 1)[-1] == str(port) and parts[2].endswith(":0") and parts[4].isdigit():
                pid = int(parts[4])
                break
        if pid is None:
            return None
        query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
        command = _output(["powershell", "-NoProfile", "-Command", query], run)
    else:
        listing = _output(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], run) or ""
        pids = [int(line) for line in listing.split() if line.isdigit()]
        if not pids:
            return None
        pid = pids[0]
        command = _output(["ps", "-o", "command=", "-p", str(pid)], run)
    return pid, (command or "").strip() or None


def port_held(port: int, holder: Optional[tuple[int, Optional[str]]]) -> str:
    """What is known of the port's holder: Port 8050 is already in use (PID 1234, command line ...)."""
    if holder is None:
        return f"Port {port} is already in use"
    pid, command = holder
    return f"Port {port} is already in use (PID {pid}" + (f", command line {command})" if command else ")")


def port_hint(port: int, holder: Optional[tuple[int, Optional[str]]]) -> str:
    """The stop's text when the port the app needs is held: the holder, then what to do."""
    return f"{port_held(port, holder)}. Stop it or use {PORT_OPTION} N"


def choose_port(
    requested: Optional[int], moves: bool, start: int = VIEWER_PORT, in_use: Optional[Callable[[int], bool]] = None,
    holder: Optional[Callable] = None, output: Callable = print,
) -> int:
    """The port an app serves on, taken before it serves: the requested one (`--port N`) when free, else the stop
    with the hint; without one `start` (Dash's default) when free, else, where the app `moves` (the dashboard, the
    standalone page), the next free port with the URL printed, or the stop with the hint (the full app). `in_use` and
    `holder` are looked up at the call (port_in_use, port_holder) unless given."""
    in_use = in_use or port_in_use
    holder = holder or port_holder
    port = start if requested is None else requested
    if not in_use(port):
        return port
    if requested is not None or not moves:
        raise SystemExit(port_hint(port, holder(port)))
    free = free_port(port + 1, in_use)
    output(f"{port_held(port, holder(port))}: serving on {app_url(free)}")
    return free


# --- the folders and the state ---------------------------------------------------------------------------------

DATA_FOLDER = "data"
STATE_FOLDER = "state"  # in the data folder: the state files, the derived course file and the episodes' log
CACHE_FOLDER = "cache"  # in the data folder: the embeddings' cache (app.models.embeddings), a derived file, no state
# What the backup question backs up: the viewer's state file and the dashboard's state file (app.dashboard.state),
# not the log, nor the model state (app.glue.derived), written at every start
STATE_NAMES = ("user_state.json", "dashboard_state.json")


def data_dir(folder: Path) -> Path:
    """The data folder of a launcher in `folder`: data/ next to it (the capstone), else the one in the folder above
    it (the flashcards workspace's root); data/ next to it where neither exists."""
    folder = Path(folder)
    candidates = (folder / DATA_FOLDER, folder.parent / DATA_FOLDER)
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def backup_path(state_file: Path) -> Path:
    """The next free backup name next to the state file: user_state.json_01.bak, _02.bak, ..."""
    for number in range(1, 100):
        candidate = state_file.parent / f"{state_file.name}_{number:02d}.bak"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not find a free backup name (100 existing backups)")


def prepare_state(
    state_dir: Path, ask: Callable[[str], str] = input, output: Callable = print, names: Sequence[str] = STATE_NAMES,
) -> list[Path]:
    """The backup question: with a state file of `names` in `state_dir`, y backs them up and starts fresh; n, anything
    else or no input keeps them. Returns the backups made."""
    existing = [Path(state_dir) / name for name in names if (Path(state_dir) / name).exists()]
    if not existing:
        output("No existing state file found.")
        return []
    listed = ", ".join(path.name for path in existing)
    output(f"Found existing state files in {existing[0].parent}: {listed}")
    try:
        answer = ask(f"Backup existing {listed}? (y/n): ").strip().lower()
    except EOFError:  # a non-interactive start (piped input)
        output("No input available; keeping existing state files.")
        return []
    if answer not in ("y", "yes"):
        output("Keeping existing state files.")
        return []
    backups = []
    for state_file in existing:
        backup = backup_path(state_file)
        state_file.rename(backup)
        output(f"Backed up existing state to: {backup}")
        backups.append(backup)
    output("Starting with fresh state files.")
    return backups
