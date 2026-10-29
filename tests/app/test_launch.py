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
        self.assertEqual(launch.LAUNCH_QUESTION, "Launch Sputnik? (y/n): ")

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


if __name__ == "__main__":
    unittest.main()
