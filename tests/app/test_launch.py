"""The launchers' part of the package (app.launch): the mode question start.py asks with sputnik, and the full app
without it on the flashcards workspace's own viewer; the page's process next to the viewer; the data folder; the
backup question both launchers ask before the full app. Nothing of the viewer: the viewer's presence is passed in or
faked."""

import model_support  # noqa: F401  (the helper's package on the path)

import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

from app import launch


class ModeQuestionTest(unittest.TestCase):
    def choose(self, found, answer=None, arguments=()):
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        return launch.choose_mode(found, ask, arguments), ask

    def test_without_the_viewer_the_mode_is_standalone_and_nothing_is_asked(self):
        mode, ask = self.choose(False, "y")
        self.assertEqual(mode, launch.STANDALONE)
        ask.assert_not_called()

    def test_with_the_viewer_y_starts_the_full_app_and_n_the_standalone_page(self):
        for answer, wanted in (("y", launch.FULL), ("yes", launch.FULL), (" N ", launch.STANDALONE), ("no", launch.STANDALONE)):
            with self.subTest(answer=answer):
                mode, ask = self.choose(True, answer)
                self.assertEqual(mode, wanted)
                ask.assert_called_once_with(launch.LAUNCH_QUESTION)
        self.assertEqual(launch.LAUNCH_QUESTION, "The optional interactive viewer Sputnik is available.\n Launch Sputnik? (y/n): ")

    def test_without_input_the_viewer_decides(self):
        self.assertEqual(self.choose(True)[0], launch.FULL)
        self.assertEqual(self.choose(False)[0], launch.STANDALONE)

    def test_the_mode_argument_skips_the_question(self):
        for arguments, wanted in ((["--mode", "standalone"], launch.STANDALONE), (["--mode=full"], launch.FULL)):
            with self.subTest(arguments=arguments):
                mode, ask = self.choose(True, "n", arguments)
                self.assertEqual(mode, wanted)
                ask.assert_not_called()
        self.assertEqual(self.choose(False, None, ["--mode", "standalone"])[0], launch.STANDALONE)

    def test_full_without_the_viewer_and_an_unknown_mode_stop(self):
        with self.assertRaises(SystemExit) as stop:
            self.choose(False, None, ["--mode", "full"])
        self.assertIn("viewer extra", str(stop.exception.code))
        with self.assertRaises(SystemExit):
            self.choose(True, None, ["--mode", "both"])

    def test_the_viewer_is_found_by_its_import_name_without_importing_it(self):
        find_spec = mock.Mock(return_value=object())
        self.assertTrue(launch.viewer_available(find_spec))
        find_spec.assert_called_once_with("project")
        self.assertFalse(launch.viewer_available(mock.Mock(return_value=None)))
        self.assertFalse(launch.viewer_available(mock.Mock(side_effect=ImportError)))

    def test_on_the_workspaces_own_viewer_the_full_app_starts_without_the_question(self):
        ask = mock.Mock(return_value="n")
        self.assertEqual(launch.choose_mode(True, ask, (), here=True), launch.FULL)
        ask.assert_not_called()
        self.assertEqual(launch.choose_mode(True, ask, ["--mode", "standalone"], here=True), launch.STANDALONE)
        self.assertEqual(launch.choose_mode(False, ask, (), here=True), launch.STANDALONE)  # no viewer, no full app
        ask.assert_not_called()

    def test_the_viewer_is_the_workspaces_own_where_its_distribution_is_installed_and_sputnik_is_not(self):
        def installed(*names):
            def version(name):
                if name not in names:
                    raise metadata.PackageNotFoundError(name)
                return "0.1.0"

            return version

        self.assertTrue(launch.viewer_here(installed("spatial-flashcards")))
        self.assertFalse(launch.viewer_here(installed("sputnik")))  # the capstone: the question stays
        self.assertFalse(launch.viewer_here(installed("spatial-flashcards", "sputnik")))
        self.assertFalse(launch.viewer_here(installed()))
        self.assertEqual((launch.WORKSPACE_DISTRIBUTION, launch.VIEWER_DISTRIBUTION), ("spatial-flashcards", "sputnik"))

    def test_the_reloaders_child_takes_the_mode_from_the_environment(self):
        self.assertEqual(launch.mode_of_child({launch.MODE_VARIABLE: "standalone"}, True), launch.STANDALONE)
        self.assertEqual(launch.mode_of_child({launch.MODE_VARIABLE: "full"}, True), launch.FULL)
        self.assertEqual(launch.mode_of_child({}, True), launch.FULL)
        self.assertEqual(launch.mode_of_child({launch.MODE_VARIABLE: "odd"}, False), launch.STANDALONE)


class PageProcessTest(unittest.TestCase):
    """The way between the two apps (step 28), shared by both launchers: the page as a process of its own next to
    the viewer, the URLs the two link each other by, the port argument."""

    def test_the_port_argument(self):
        self.assertIsNone(launch.requested_port([]))
        self.assertEqual(launch.requested_port(["--mode", "standalone", "--port", "8051"]), 8051)
        self.assertEqual(launch.requested_port(["--port=8052"]), 8052)
        with self.assertRaises(SystemExit):
            launch.requested_port(["--port", "eight"])
        self.assertEqual((launch.STANDALONE_PORT, launch.VIEWER_PORT), (8051, 8050))  # next to the viewer's 8050, Dash's default
        self.assertEqual(launch.standalone_url(), "http://127.0.0.1:8051/")
        self.assertEqual(launch.viewer_url(), "http://127.0.0.1:8050/")
        self.assertEqual((launch.MODELS_URL, launch.VIEWER_URL), ("HELPER_MODELS_URL", "HELPER_VIEWER_URL"))

    def test_start_standalone_runs_the_launcher_again_in_standalone_mode_on_the_port_with_the_viewers_url(self):
        commands = []
        messages = []

        def popen(command, env=None):
            commands.append((list(command), env))
            return "process"

        environ = {"PATH": "x", "WERKZEUG_RUN_MAIN": "true"}
        process = launch.start_standalone("C:/somewhere/start.py", 8051, popen=popen, environ=environ, executable="python.exe", output=messages.append)
        self.assertEqual(process, "process")
        (command, env), = commands
        self.assertEqual(command[0], "python.exe")
        self.assertTrue(command[1].endswith("start.py"))
        self.assertEqual(command[2:], ["--mode", "standalone", "--port", "8051"])
        self.assertEqual(env, {"PATH": "x", launch.VIEWER_URL: "http://127.0.0.1:8050/"})  # outside the viewer's reloader, with its URL
        self.assertEqual(environ, {"PATH": "x", "WERKZEUG_RUN_MAIN": "true"})  # the launcher's own environment untouched
        self.assertIn("http://127.0.0.1:8051/", messages[0])
        launch.start_standalone("start.py", 8052, popen=popen, environ={}, executable="py", output=messages.append, viewer_port=9000, ready="READY")
        (command, env) = commands[-1]
        self.assertEqual(command[2:], ["--mode", "standalone", "--port", "8052"])
        self.assertEqual(env, {launch.VIEWER_URL: "http://127.0.0.1:9000/", "READY": "1"})  # a module launcher's mark that skips its env check

    def test_stop_standalone_waits_for_the_page_and_kills_one_that_stays(self):
        import subprocess

        class FakeProcess:
            def __init__(self, outcome, killed_code=1):
                self.outcome, self.killed_code, self.killed, self.waits = outcome, killed_code, False, []

            def wait(self, timeout=None):
                self.waits.append(timeout)
                if self.killed:
                    return self.killed_code
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return self.outcome

            def kill(self):
                self.killed = True

        process = FakeProcess(0)
        self.assertEqual(launch.stop_standalone(process, grace=5), 0)
        self.assertEqual((process.waits, process.killed), ([5], False))
        process = FakeProcess(subprocess.TimeoutExpired(["python"], 5), killed_code=1)
        self.assertEqual(launch.stop_standalone(process, grace=5), 1)
        self.assertEqual((process.waits, process.killed), ([5, None], True))
        self.assertEqual(launch.STANDALONE_GRACE, 10.0)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class DataFolderTest(TempDirTest):
    def test_the_data_folder_next_to_the_launcher_comes_before_the_one_above_it(self):
        launcher_dir = self.tmp / "root" / "modules"
        launcher_dir.mkdir(parents=True)
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir / "data")  # neither: the capstone's place
        (launcher_dir.parent / "data").mkdir()
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir.parent / "data")  # the flashcards workspace's
        (launcher_dir / "data").mkdir()
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir / "data")  # the capstone's, next to start.py
        self.assertEqual((launch.STATE_FOLDER, launch.CACHE_FOLDER), ("state", "cache"))


class BackupQuestionTest(TempDirTest):
    """The backup question both launchers ask before the full app (prepare_state), on the viewer's state file and the
    model state; the episodes' log is never touched."""

    def setUp(self):
        super().setUp()
        self.state = self.tmp / "user_state.json"
        self.model_state = self.tmp / "model_state.json"
        self.log = self.tmp / "episodes.csv"
        for path in (self.state, self.model_state, self.log):
            path.write_text("{}", encoding="utf-8")
        self.messages = []

    def prepare(self, answer=None, names=launch.STATE_NAMES):
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        return launch.prepare_state(self.tmp, ask, self.messages.append, names), ask

    def test_the_state_files_are_the_viewers_and_the_model_state(self):
        self.assertEqual(launch.STATE_NAMES, ("user_state.json", "model_state.json"))

    def test_backup_names_count_up(self):
        self.assertEqual(launch.backup_path(self.state).name, "user_state.json_01.bak")
        (self.tmp / "user_state.json_01.bak").write_text("{}", encoding="utf-8")
        self.assertEqual(launch.backup_path(self.state).name, "user_state.json_02.bak")

    def test_yes_backs_up_both_and_starts_fresh(self):
        backups, ask = self.prepare("y")
        self.assertFalse(self.state.exists() or self.model_state.exists())
        self.assertEqual([path.name for path in backups], ["user_state.json_01.bak", "model_state.json_01.bak"])
        self.assertTrue(all(path.exists() for path in backups))
        self.assertTrue(self.log.exists())  # the episodes' log stays
        self.assertEqual(ask.call_args[0][0], "Backup existing user_state.json, model_state.json? (y/n): ")
        self.assertEqual(self.messages[-1], "Starting with fresh state files.")

    def test_no_keeps_the_state_files(self):
        backups, _ask = self.prepare("n")
        self.assertEqual(backups, [])
        self.assertTrue(self.state.exists() and self.model_state.exists())

    def test_no_input_keeps_the_state_files(self):
        backups, _ask = self.prepare()
        self.assertEqual(backups, [])
        self.assertTrue(self.state.exists() and self.model_state.exists())
        self.assertIn("No input available", self.messages[-1])

    def test_without_a_state_file_nothing_is_asked(self):
        self.state.unlink()
        self.model_state.unlink()
        backups, ask = self.prepare("y")
        self.assertEqual(backups, [])
        ask.assert_not_called()
        self.assertTrue(self.log.exists())

    def test_one_state_file_alone_is_backed_up_alone(self):
        self.model_state.unlink()
        backups, _ask = self.prepare("yes")
        self.assertEqual([path.name for path in backups], ["user_state.json_01.bak"])


if __name__ == "__main__":
    unittest.main()
