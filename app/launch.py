"""The launcher's part of the package: start.py, the launcher exported with the package, starts the dashboard alone
with it; the flashcards workspace's launchers take the same calls. The standard library only, so a launcher starts
on any interpreter and stops with the fix.

The port argument (requested_port, over option_value): `--port N` or `--port=N`.

The ports (choose_port), taken by the launcher's first process before anything serves: on Windows a second server
binds a port a live one holds (Werkzeug's SO_REUSEADDR) and the older process keeps getting the requests, so a port
counts as held when a connect to it is answered (port_in_use), never by a bind. An app that moves (the dashboard)
takes the next free port from Dash's default (free_port) and prints the URL it serves on; one that does not stops with
the hint (port_hint: the PID and the command line of the process listening there, port_holder) when its port is held.
`--port N` names the port and stops when it is held.

The folders (data_dir): the data folder lies next to the launcher (the capstone), else in the folder above it (the
flashcards workspace's root): the course table, the state in data/state/ and the embeddings' cache in data/cache/.

The backup question (prepare_state), asked before an app that stores state starts, by the launcher's first process:
with a state file of the given names in the state folder, y backs them up (backup_path: dashboard_state.json_01.bak,
_02.bak, ...) and starts fresh, n or no input keeps them. start.py asks it over the dashboard's state file
(STATE_NAMES); the flashcards workspace's launchers pass their own names.
"""

import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

DASH_PORT = 8050  # Dash's default: where a launcher's port check starts
PORT_OPTION = "--port"


# --- the port argument -------------------------------------------------------------------------------------


def option_value(arguments: Sequence[str], option: str) -> Optional[str]:
    """The value of `<option> <value>` or `<option>=<value>` among the arguments (the last one given), "" for the
    option without a value, or None."""
    arguments = list(arguments)
    value = None
    for position, argument in enumerate(arguments):
        if argument == option:
            value = arguments[position + 1] if position + 1 < len(arguments) else ""
        elif argument.startswith(option + "="):
            value = argument.split("=", 1)[1]
    return value


def requested_port(arguments: Sequence[str]) -> Optional[int]:
    """The port named by `--port <n>` or `--port=<n>`, or None; anything but a number stops."""
    value = option_value(arguments, PORT_OPTION)
    if value is None:
        return None
    if not value.isdigit():
        raise SystemExit(f"{PORT_OPTION} takes a number, not {value!r}")
    return int(value)


def app_url(port: int) -> str:
    """Where an app of the launcher serves on the port."""
    return f"http://127.0.0.1:{port}/"


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
    requested: Optional[int], moves: bool, start: int = DASH_PORT, in_use: Optional[Callable[[int], bool]] = None,
    holder: Optional[Callable] = None, output: Callable = print,
) -> int:
    """The port an app serves on, taken before it serves: the requested one (`--port N`) when free, else the stop
    with the hint; without one `start` (Dash's default) when free, else, where the app `moves` (the dashboard, the
    models view), the next free port with the URL printed, or the stop with the hint (the viewer). `in_use` and
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
# What start.py's backup question backs up: the dashboard's state file (app.dashboard.state.STATE_NAME, not imported:
# the dashboard's package imports dash)
STATE_NAMES = ("dashboard_state.json",)


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
