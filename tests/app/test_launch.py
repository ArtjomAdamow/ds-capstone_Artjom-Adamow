"""The mode question of the package (app.launch), which both launchers ask: helper.py here after the env check,
start.py in the capstone. Nothing of the viewer: the viewer's presence is passed in or faked."""

import model_support  # noqa: F401  (the helper's package on the path)

import unittest
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
        self.assertEqual(env, {launch.VIEWER_URL: "http://127.0.0.1:9000/", "READY": "1"})  # helper.py's mark that skips its env check

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


if __name__ == "__main__":
    unittest.main()
