"""The launcher (start.py, next to the package): it starts the dashboard; the port check comes before anything
serves or asks (the dashboard moves to the next
free port and says so, a held `--port` stops it), then the backup question over the dashboard's state file alone; a
missing library or a missing data file stops the start with the command that fixes it (the dashboard starts with deck
00 alone without the course table, and stops without either); its folders follow the data folder's rule; it loads
and stops without the models' libraries (a subprocess with them blocked). The run, the backup question and the ports
are mocked, so nothing is served and no state is touched."""

import model_support  # noqa: F401  (the package and start.py on the path)

import contextlib
import inspect
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import start
from app import launch


class StartTest(unittest.TestCase):
    def start(self, answer=None, arguments=(), held=()):
        """main with the ports in `held` answering a connect, and the run, the backup question and the holder's
        lookup mocked: the ask mock."""
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        with (
            mock.patch.object(start, "missing_libraries", return_value=[]),
            mock.patch.object(launch, "port_in_use", side_effect=lambda port: port in held),
            mock.patch.object(launch, "port_holder", return_value=(4242, "python start.py")),
            mock.patch.object(start, "prepare_state") as prepare_state,
            mock.patch.object(start, "run_dashboard") as dashboard,
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            self.dashboard, self.prepare_state = dashboard, prepare_state
            try:
                start.main(ask, list(arguments))
            finally:
                self.printed = printed.getvalue().splitlines()
        return ask

    def test_the_dashboard_starts_after_the_backup_question_and_asks_nothing_else(self):
        ask = self.start("n")
        self.dashboard.assert_called_once_with(port=8050)
        self.prepare_state.assert_called_once_with(start.STATE_DIR, ask)  # data/state/, over launch.STATE_NAMES
        ask.assert_not_called()  # the backup question itself is mocked: nothing else is asked
        self.start("n", ["--port", "8053"])
        self.dashboard.assert_called_once_with(port=8053)

    def test_an_argument_it_does_not_know_changes_nothing(self):
        self.start("n", ["--mode", "full"])
        self.dashboard.assert_called_once_with(port=8050)

    def test_the_dashboard_moves_to_the_next_free_port_and_says_so(self):
        self.start("n", held=(8050, 8051))
        self.assertEqual(self.dashboard.call_args[1]["port"], 8052)
        (moved,) = [line for line in self.printed if "already in use" in line]
        self.assertEqual(moved, "Port 8050 is already in use (PID 4242, command line python start.py): serving on http://127.0.0.1:8052/")

    def test_a_named_port_that_is_held_stops_before_anything_is_asked(self):
        with self.assertRaises(SystemExit) as stop:
            self.start("n", ["--port", "8060"], held=(8060,))
        self.assertEqual(stop.exception.code, "Port 8060 is already in use (PID 4242, command line python start.py). Stop it or use --port N")
        self.prepare_state.assert_not_called()  # the port is checked before the backup question
        self.dashboard.assert_not_called()

    def test_the_backup_question_is_over_the_dashboards_state_file_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            state_dir = Path(folder)
            for name in ("user_state.json", "dashboard_state.json", "model_state.json"):
                (state_dir / name).write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(start, "missing_libraries", return_value=[]),
                mock.patch.object(start, "choose_port", return_value=8050),
                mock.patch.object(start, "STATE_DIR", state_dir),
                mock.patch.object(start, "run_dashboard"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                start.main(mock.Mock(return_value="y"), [])
            self.assertEqual(sorted(path.name for path in state_dir.iterdir()),
                             ["dashboard_state.json_01.bak", "model_state.json", "user_state.json"])

    def test_a_missing_library_stops_with_the_install_command(self):
        with (
            mock.patch.object(start, "missing_libraries", return_value=["scipy"]),
            mock.patch.object(start, "run_dashboard") as dashboard,
        ):
            with self.assertRaises(SystemExit) as stop:
                start.main(mock.Mock(), [])
        message = str(stop.exception.code)
        for part in ("scipy", start.INSTALL, start.START):
            self.assertIn(part, message)
        dashboard.assert_not_called()

    def test_missing_libraries_names_the_ones_not_found(self):
        self.assertEqual(start.missing_libraries(lambda name: None if name == "sklearn" else object()), ["sklearn"])
        self.assertEqual(start.missing_libraries(mock.Mock(side_effect=ValueError)), list(start.LIBRARIES))
        self.assertEqual(start.missing_libraries(lambda name: object()), [])


class StartDataTest(unittest.TestCase):
    def test_the_dashboard_without_either_table_stops_with_the_data_hint_and_passes_its_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaises(SystemExit) as stop:
                start.run_dashboard(folder / "course", demo_dir=folder / "demo")
            message = str(stop.exception.code)
            self.assertIn("cards.csv", message)
            self.assertIn(start.DATA_HINT, message)
            with mock.patch("app.dashboard.main") as dashboard_main:
                table = model_support.write_test_table(folder / "course")
                start.run_dashboard(table, folder / "results", folder / "state", folder / "cache", port=8053, demo_dir=folder / "demo")
            dashboard_main.assert_called_once_with(table, folder / "results", folder / "state", cache_dir=folder / "cache", port=8053,
                                                   demo_dir=folder / "demo")
        parameters = inspect.signature(start.run_dashboard).parameters
        self.assertEqual(parameters["state_dir"].default, start.DATA_DIR / "state")
        self.assertEqual(parameters["demo_dir"].default, start.DATA_DIR / "demo" / "dashboard")

    def test_the_dashboard_without_the_course_table_starts_with_deck_00_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            demo = model_support.write_demo_table(folder / "demo")
            with mock.patch("app.dashboard.main") as dashboard_main, mock.patch("builtins.print") as printed:
                start.run_dashboard(folder / "course", folder / "results", folder / "state", folder / "cache", demo_dir=demo)
            dashboard_main.assert_called_once_with(folder / "course", folder / "results", folder / "state", cache_dir=folder / "cache",
                                                   port=None, demo_dir=demo)
            self.assertIn("starts empty, with its Demo", printed.call_args.args[0])

    def test_the_folders_follow_the_data_folders_rule(self):
        self.assertTrue((start.ROOT / "app" / "__init__.py").is_file())
        self.assertEqual(start.DATA_DIR, launch.data_dir(start.ROOT))  # data/ next to the package, or above it
        self.assertEqual(start.TABLE_DIR, start.DATA_DIR / "course")
        self.assertEqual(start.STATE_DIR, start.DATA_DIR / "state")
        self.assertEqual(start.CACHE_DIR, start.DATA_DIR / "cache")
        self.assertEqual(start.RESULTS_DIR, start.ROOT / "results")  # next to evaluate.py


# A process without the models' libraries: start.py loads (the standard library and app.launch only)
# and stops with the command
NO_LIBRARIES = """
import sys
for name in ("numpy", "scipy", "sklearn", "dash", "plotly"):
    sys.modules[name] = None  # `import <name>` raises ImportError from here on
sys.path.insert(0, {module_dir!r})
import start
try:
    start.main(lambda question: "y", [])
except SystemExit as stop:
    print("STOPPED", stop.code)
"""


class StartWithoutLibrariesTest(unittest.TestCase):
    def test_it_loads_and_stops_with_the_command_without_the_libraries(self):
        module_dir = model_support.MODULE_DIR
        script = NO_LIBRARIES.format(module_dir=str(module_dir))
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, cwd=str(module_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STOPPED numpy, scipy, sklearn, dash, plotly not installed", result.stdout)
        self.assertIn(start.START, result.stdout)


if __name__ == "__main__":
    unittest.main()
