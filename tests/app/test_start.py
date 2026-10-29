"""The capstone's launcher (start.py, exported next to the package): it searches for sputnik and asks only when it
finds it, the reloader's child asks nothing, and a missing library, a missing data file or an older sputnik stops the
start with the command that fixes it; it loads and stops without the models' libraries (a subprocess with them
blocked). The two runs are mocked, so nothing is served and nothing of the viewer is imported."""

import model_support  # noqa: F401  (the package and start.py on the path)

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
    def start(self, found, answer=None, environ=None):
        """main with the viewer found or not and the runs mocked: (the mode that ran, the ask mock, the environ)."""
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        environ = {} if environ is None else environ
        with (
            mock.patch.object(start, "missing_libraries", return_value=[]),
            mock.patch.object(start, "viewer_available", return_value=found),
            mock.patch.object(start, "run_standalone") as standalone,
            mock.patch.object(start, "run_full") as full,
            mock.patch("builtins.print"),
        ):
            start.main(ask, environ)
        self.assertEqual(standalone.call_count + full.call_count, 1)
        return (launch.STANDALONE if standalone.called else launch.FULL), ask, environ

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

    def test_the_folders_are_the_capstones_data_next_to_the_package(self):
        self.assertTrue((start.ROOT / "app" / "__init__.py").is_file())
        self.assertEqual(start.COURSE_FILE, start.ROOT / "data" / "topics.json")
        self.assertEqual(start.TABLE_DIR, start.ROOT / "data" / "course")
        self.assertEqual(start.STATE_FILE.parent, start.ROOT / "data" / "state")


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
