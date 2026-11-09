"""The launcher (start.py, exported next to the package): it searches for sputnik and asks only when it finds it (on
the flashcards workspace's own viewer the full app starts without the question), asks the backup question before the
full app and before the dashboard (`--mode dashboard`, the third mode, by the argument alone), the reloader's child asks nothing, and a missing library, a missing data file or an older sputnik stops the
start with the command that fixes it (the dashboard starts with deck 00 alone without the course table, and stops
without either); the port check comes before anything serves (the dashboard and the page move to the next free port
and say so, the full app stops naming the holder); its folders follow the data folder's rule; it loads and stops
without the models' libraries (a subprocess with them blocked). The two runs, the backup question and the ports are
mocked, so nothing is served, no state is touched and nothing of the viewer is imported."""

import model_support  # noqa: F401  (the package and start.py on the path)

import contextlib
import inspect
import io
import subprocess
import sys
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

import start
from app import launch


class StartModeTest(unittest.TestCase):
    def start(self, found, answer=None, environ=None, arguments=(), here=False, held=()):
        """main with the viewer found or not (the workspace's own with `here`), the ports in `held` answering a
        connect, and the runs, the backup question and the holder's lookup mocked: (the mode that ran, the ask mock,
        the environ)."""
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        environ = {} if environ is None else environ
        with (
            mock.patch.object(start, "missing_libraries", return_value=[]),
            mock.patch.object(start, "viewer_available", return_value=found),
            mock.patch.object(start, "viewer_here", return_value=here),
            mock.patch.object(launch, "port_in_use", side_effect=lambda port: port in held),
            mock.patch.object(launch, "port_holder", return_value=(4242, "python start.py")),
            mock.patch.object(start, "prepare_state") as prepare_state,
            mock.patch.object(start, "run_standalone") as standalone,
            mock.patch.object(start, "run_full") as full,
            mock.patch.object(start, "run_dashboard") as dashboard,
            mock.patch.object(start, "start_standalone", return_value="page") as page,
            mock.patch.object(start, "stop_standalone") as stop,
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            self.page, self.stop, self.standalone, self.dashboard, self.prepare_state = page, stop, standalone, dashboard, prepare_state
            self.full = full
            try:
                start.main(ask, environ, list(arguments))
            finally:
                self.printed = printed.getvalue().splitlines()
        self.assertEqual(standalone.call_count + full.call_count + dashboard.call_count, 1)
        ran = launch.STANDALONE if standalone.called else (launch.DASHBOARD if dashboard.called else launch.FULL)
        return ran, ask, environ

    def test_the_full_app_starts_the_page_next_to_it_and_stops_it_after(self):
        ran, _ask, environ = self.start(True, "y")
        self.assertEqual(ran, launch.FULL)
        self.page.assert_called_once()  # the page's process: this file again, --mode standalone --port 8051
        self.assertEqual(str(self.page.call_args[0][0]), str(Path(start.__file__)))
        self.assertIs(self.page.call_args[1]["environ"], environ)
        self.assertEqual((self.page.call_args[0][1], self.page.call_args[1]["viewer_port"]), (8051, 8050))
        self.assertEqual(environ[launch.MODELS_URL], "http://127.0.0.1:8051/")  # the viewer's Models button reads it
        self.assertEqual(self.full.call_args[1]["port"], 8050)
        self.stop.assert_called_once_with("page")  # waited for once the viewer has stopped
        ran, _ask, _environ = self.start(True, "y", {"WERKZEUG_RUN_MAIN": "true", launch.MODE_VARIABLE: launch.FULL})
        self.assertEqual(ran, launch.FULL)
        self.page.assert_not_called()  # the reloader's child starts no second page
        self.stop.assert_not_called()

    def test_the_page_is_stopped_even_when_the_full_app_fails(self):
        with (
            mock.patch.object(start, "missing_libraries", return_value=[]),
            mock.patch.object(start, "viewer_available", return_value=True),
            mock.patch.object(launch, "port_in_use", return_value=False),
            mock.patch.object(start, "prepare_state"),
            mock.patch.object(start, "run_full", side_effect=RuntimeError("the viewer")),
            mock.patch.object(start, "start_standalone", return_value="page"),
            mock.patch.object(start, "stop_standalone") as stop,
        ):
            with self.assertRaises(RuntimeError):
                start.main(lambda question: "y", {}, [])
        stop.assert_called_once_with("page")

    def test_mode_standalone_with_the_port_runs_the_page_alone_with_the_viewers_url(self):
        ran, ask, environ = self.start(True, "y", {launch.VIEWER_URL: "http://127.0.0.1:8050/"}, ["--mode", "standalone", "--port", "8051"])
        self.assertEqual(ran, launch.STANDALONE)
        ask.assert_not_called()  # the argument skips the question: the page's own process asks nothing
        self.standalone.assert_called_once_with(port=8051, viewer_url="http://127.0.0.1:8050/")
        self.page.assert_not_called()
        ran, _ask, _environ = self.start(False)
        self.standalone.assert_called_once_with(port=8050, viewer_url=None)  # a start without the viewer: no link, Dash's default port

    def test_mode_dashboard_runs_the_dashboard_after_the_backup_question_with_no_page_next_to_it(self):
        for found in (True, False):
            with self.subTest(found=found):
                ran, ask, environ = self.start(found, "n", arguments=["--mode", "dashboard", "--port", "8053"])
                self.assertEqual(ran, launch.DASHBOARD)
                self.assertNotIn(mock.call(launch.LAUNCH_QUESTION), ask.call_args_list)  # the argument alone starts it
                self.dashboard.assert_called_once_with(port=8053)
                self.prepare_state.assert_called_once_with(start.STATE_FILE.parent, ask)  # the backup question, as before the full app
                self.page.assert_not_called()  # no page next to it
                self.stop.assert_not_called()
                self.assertEqual(environ[launch.MODE_VARIABLE], launch.DASHBOARD)
        ran, _ask, _environ = self.start(True, "y", arguments=["--mode=dashboard"], here=True)
        self.assertEqual(ran, launch.DASHBOARD)
        self.dashboard.assert_called_once_with(port=8050)

    def test_the_dashboard_and_the_page_move_to_the_next_free_port_and_say_so(self):
        for arguments, run in ((["--mode", "dashboard"], "dashboard"), (["--mode", "standalone"], "standalone")):
            with self.subTest(mode=arguments[1]):
                self.start(True, "n", arguments=arguments, held=(8050, 8051))
                self.assertEqual(getattr(self, run).call_args[1]["port"], 8052)
                (moved,) = [line for line in self.printed if "already in use" in line]
                self.assertEqual(moved, "Port 8050 is already in use (PID 4242, command line python start.py): serving on http://127.0.0.1:8052/")

    def test_the_full_app_stops_naming_the_holder_when_its_port_is_held_before_anything_is_asked(self):
        with self.assertRaises(SystemExit) as stop:
            self.start(True, "y", held=(8050,))
        self.assertEqual(stop.exception.code, "Port 8050 is already in use (PID 4242, command line python start.py). Stop it or use --port N")
        self.prepare_state.assert_not_called()  # the port is checked before the backup question
        self.page.assert_not_called()
        self.start(True, "y", arguments=["--port", "8060"], held=(8050,))  # another named port that is free
        self.assertEqual(self.full.call_args[1]["port"], 8060)
        self.assertEqual((self.page.call_args[0][1], self.page.call_args[1]["viewer_port"]), (8061, 8060))

    def test_a_named_port_that_is_held_stops_every_mode(self):
        for mode in launch.MODES:
            with self.subTest(mode=mode), self.assertRaises(SystemExit) as stop:
                self.start(True, "n", arguments=["--mode", mode, "--port", "8060"], held=(8060,))
            self.assertIn("Port 8060 is already in use (PID 4242", str(stop.exception.code))

    def test_the_full_apps_page_takes_the_next_free_port_above_the_viewers(self):
        _ran, _ask, environ = self.start(True, "y", held=(8051, 8052))
        self.assertEqual((self.page.call_args[0][1], self.page.call_args[1]["viewer_port"]), (8053, 8050))
        self.assertEqual(environ[launch.MODELS_URL], "http://127.0.0.1:8053/")
        self.assertEqual(self.full.call_args[1]["port"], 8050)

    def test_the_reloaders_child_checks_no_port_its_parent_holds_it(self):
        self.start(True, "y", {"WERKZEUG_RUN_MAIN": "true", launch.MODE_VARIABLE: launch.FULL}, held=(8050,))
        self.assertEqual(self.full.call_args[1]["port"], 8050)
        self.start(True, "y", {"WERKZEUG_RUN_MAIN": "true", launch.MODE_VARIABLE: launch.FULL}, ["--port", "8060"], held=(8060,))
        self.assertEqual(self.full.call_args[1]["port"], 8060)

    def test_without_sputnik_the_standalone_page_starts_and_nothing_is_asked(self):
        ran, ask, environ = self.start(False, "y")
        self.assertEqual(ran, launch.STANDALONE)
        ask.assert_not_called()
        self.assertEqual(environ[launch.MODE_VARIABLE], launch.STANDALONE)

    def test_with_sputnik_it_asks_y_starts_the_full_app_and_n_the_standalone_page(self):
        for answer, wanted in (("y", launch.FULL), ("n", launch.STANDALONE)):
            with self.subTest(answer=answer):
                ran, ask, environ = self.start(True, answer)
                self.assertEqual(ran, wanted)
                ask.assert_called_once_with(launch.LAUNCH_QUESTION)
                self.assertEqual(environ[launch.MODE_VARIABLE], wanted)

    def test_with_sputnik_and_no_input_the_full_app_starts(self):
        self.assertEqual(self.start(True)[0], launch.FULL)

    def test_on_the_workspaces_own_viewer_the_full_app_starts_without_the_mode_question(self):
        ran, ask, environ = self.start(True, "n", here=True)
        self.assertEqual(ran, launch.FULL)
        self.assertNotIn(mock.call(launch.LAUNCH_QUESTION), ask.call_args_list)
        self.assertEqual(environ[launch.MODE_VARIABLE], launch.FULL)
        self.page.assert_called_once()  # the page next to the viewer, as with sputnik
        ran, _ask, _environ = self.start(True, "y", arguments=["--mode", "standalone"], here=True)
        self.assertEqual(ran, launch.STANDALONE)

    def test_the_backup_question_comes_before_the_full_app_in_the_first_process_only(self):
        ran, ask, _environ = self.start(True, "y")
        self.assertEqual(ran, launch.FULL)
        self.prepare_state.assert_called_once_with(start.STATE_FILE.parent, ask)  # data/state/, with the same input
        self.start(True, "n")  # the standalone page stores nothing: no question
        self.prepare_state.assert_not_called()
        self.start(True, "y", {"WERKZEUG_RUN_MAIN": "true", launch.MODE_VARIABLE: launch.FULL})  # the reloader's child
        self.prepare_state.assert_not_called()
        self.start(True, "n", here=True)  # the workspace's own viewer: no mode question, the backup question
        self.prepare_state.assert_called_once()

    def test_the_reloaders_child_asks_nothing_and_runs_its_parents_mode(self):
        environ = {"WERKZEUG_RUN_MAIN": "true", launch.MODE_VARIABLE: launch.STANDALONE}
        ran, ask, _ = self.start(True, "y", environ)
        self.assertEqual(ran, launch.STANDALONE)
        ask.assert_not_called()

    def test_a_missing_library_stops_with_the_install_commands(self):
        with (
            mock.patch.object(start, "missing_libraries", return_value=["scipy"]),
            mock.patch.object(start, "run_standalone") as standalone,
            mock.patch.object(start, "run_full") as full,
        ):
            with self.assertRaises(SystemExit) as stop:
                start.main(mock.Mock(), {})
        message = str(stop.exception.code)
        for part in ("scipy", start.INSTALL, start.INSTALL_VIEWER, start.START):
            self.assertIn(part, message)
        standalone.assert_not_called()
        full.assert_not_called()

    def test_missing_libraries_names_the_ones_not_found(self):
        self.assertEqual(start.missing_libraries(lambda name: None if name == "sklearn" else object()), ["sklearn"])
        self.assertEqual(start.missing_libraries(mock.Mock(side_effect=ValueError)), list(start.LIBRARIES))
        self.assertEqual(start.missing_libraries(lambda name: object()), [])


class StartDataTest(unittest.TestCase):
    def test_standalone_without_the_course_table_stops_with_the_data_hint(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit) as stop:
                start.run_standalone(Path(folder) / "course")
        message = str(stop.exception.code)
        self.assertIn("cards.csv", message)
        self.assertIn(start.DATA_HINT, message)

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
        self.assertEqual(parameters["state_dir"].default, start.STATE_FILE.parent)  # next to the viewer's state
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

    def test_standalone_passes_the_port_and_the_viewers_url_to_the_page(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("app.standalone.main") as standalone_main:
            table = model_support.write_test_table(Path(folder) / "course")
            start.run_standalone(table, Path(folder) / "cache", port=8051, viewer_url="http://127.0.0.1:8050/")
        standalone_main.assert_called_once_with(table, start.RESULTS_DIR, cache_dir=Path(folder) / "cache", port=8051, viewer_url="http://127.0.0.1:8050/")

    def test_the_full_app_without_the_course_file_stops_with_the_data_hint_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(SystemExit) as stop:
                start.run_full(root, root / "data" / "topics.json", root / "data" / "state" / "user_state.json")
            self.assertEqual(list(root.iterdir()), [])
        self.assertIn(start.DATA_HINT, str(stop.exception.code))

    def test_an_older_sputnik_stops_the_full_app_with_the_install_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            course_file = root / "topics.json"
            course_file.write_text("{}", encoding="utf-8")
            with mock.patch.object(start, "viewer_too_old", return_value="0.2.0"):
                with self.assertRaises(SystemExit) as stop:
                    start.run_full(root, course_file, root / "state" / "user_state.json")
            self.assertFalse((root / "state").exists())
        message = str(stop.exception.code)
        for part in ("0.2.0", launch.VIEWER_RELEASE, start.INSTALL_VIEWER):
            self.assertIn(part, message)

    def test_the_viewer_is_too_old_below_the_release_the_glue_needs(self):
        def installed(version):
            return lambda name: version

        self.assertEqual(start.viewer_too_old(installed("0.2.0")), "0.2.0")
        self.assertIsNone(start.viewer_too_old(installed(launch.VIEWER_RELEASE)))
        self.assertIsNone(start.viewer_too_old(installed("0.10.0")))
        self.assertIsNone(start.viewer_too_old(mock.Mock(side_effect=metadata.PackageNotFoundError)))

    def test_the_folders_follow_the_data_folders_rule(self):
        self.assertTrue((start.ROOT / "app" / "__init__.py").is_file())
        self.assertEqual(start.DATA_DIR, launch.data_dir(start.ROOT))  # data/ next to the package, or above it
        self.assertEqual(start.COURSE_FILE, start.DATA_DIR / "topics.json")
        self.assertEqual(start.TABLE_DIR, start.DATA_DIR / "course")
        self.assertEqual(start.STATE_FILE, start.DATA_DIR / "state" / "user_state.json")
        self.assertEqual(start.CACHE_DIR, start.DATA_DIR / "cache")
        self.assertEqual(start.RESULTS_DIR, start.ROOT / "results")  # next to evaluate.py
        # compose's root is the data folder's parent: the images under data/images/
        self.assertEqual(inspect.signature(start.run_full).parameters["root"].default, start.DATA_DIR.parent)


# A process without the models' libraries and the viewer: start.py loads (the standard library and app.launch only)
# and stops with the commands
NO_LIBRARIES = """
import sys
for name in ("numpy", "scipy", "sklearn", "dash", "plotly", "project"):
    sys.modules[name] = None  # `import <name>` raises ImportError from here on
sys.path.insert(0, {module_dir!r})
import start
try:
    start.main(lambda question: "y", {{}})
except SystemExit as stop:
    print("STOPPED", stop.code)
"""


class StartWithoutLibrariesTest(unittest.TestCase):
    def test_it_loads_and_stops_with_the_commands_without_the_libraries(self):
        module_dir = model_support.MODULE_DIR
        script = NO_LIBRARIES.format(module_dir=str(module_dir))
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, cwd=str(module_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STOPPED numpy, scipy, sklearn, dash, plotly not installed", result.stdout)
        self.assertIn(start.START, result.stdout)


if __name__ == "__main__":
    unittest.main()
