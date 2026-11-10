"""The launcher's part of the package (app.launch): the option reader and the port argument, the port check, the data
folder and the backup question. The port check runs on a socket the test binds, the holder's lookup on faked command
output."""

import model_support  # noqa: F401  (the package on the path)

import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import launch


class PortArgumentTest(unittest.TestCase):
    def test_an_options_value_the_last_one_given(self):
        self.assertIsNone(launch.option_value([], "--mode"))
        self.assertEqual(launch.option_value(["--mode", "full", "--port", "8051"], "--mode"), "full")
        self.assertEqual(launch.option_value(["--mode=demo"], "--mode"), "demo")
        self.assertEqual(launch.option_value(["--mode", "full", "--mode=dashboard"], "--mode"), "dashboard")
        self.assertEqual(launch.option_value(["--mode"], "--mode"), "")
        self.assertIsNone(launch.option_value(["--model", "x"], "--mode"))

    def test_the_port_argument(self):
        self.assertIsNone(launch.requested_port([]))
        self.assertEqual(launch.requested_port(["--mode", "full", "--port", "8051"]), 8051)
        self.assertEqual(launch.requested_port(["--port=8052"]), 8052)
        with self.assertRaises(SystemExit):
            launch.requested_port(["--port", "eight"])
        self.assertEqual(launch.DASH_PORT, 8050)  # Dash's default
        self.assertEqual(launch.app_url(8051), "http://127.0.0.1:8051/")


class PortCheckTest(unittest.TestCase):
    """The port check (the ports finding): a port is held when a server answers a connect; an app that moves (the
    dashboard) takes the next free one, an app that does not and a named port stop with the holder's hint."""

    def test_a_port_is_in_use_while_a_server_listens_on_it(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            self.assertTrue(launch.port_in_use(port))
        self.assertFalse(launch.port_in_use(port))

    def test_free_port_scans_upwards_from_the_start(self):
        held = {8050, 8051, 8053}
        self.assertEqual(launch.free_port(8050, held.__contains__), 8052)
        self.assertEqual(launch.free_port(8052, held.__contains__), 8052)
        with self.assertRaises(SystemExit) as stop:
            launch.free_port(8050, lambda port: True, scan=3)
        self.assertIn("8050 to 8052", str(stop.exception.code))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            self.assertNotEqual(launch.free_port(port), port)  # the real check passes over the bound socket

    def test_the_holder_on_windows_from_netstat_and_the_process_list(self):
        netstat = (
            "\nAktive Verbindungen\n\n  Proto  Lokale Adresse         Remoteadresse          Status           PID\n"
            "  TCP    0.0.0.0:135            0.0.0.0:0              ABH\ufffdREN         1944\n"
            "  TCP    127.0.0.1:8050         127.0.0.1:60000        HERGESTELLT     777\n"
            "  TCP    127.0.0.1:8050         0.0.0.0:0              ABH\ufffdREN         4242\n"
            "  TCP    [::1]:18050            [::]:0                 ABH\ufffdREN         99\n"
        )
        commands = []

        def run(command, **options):
            commands.append(command)
            output = netstat if command[0] == "netstat" else "C:\\py\\python.exe modules\\start.py\r\n"
            return subprocess.CompletedProcess(command, 0, stdout=output)

        self.assertEqual(launch.port_holder(8050, run, "win32"), (4242, "C:\\py\\python.exe modules\\start.py"))
        self.assertIn("ProcessId=4242", commands[-1][-1])  # the listening row, whatever the state's language
        self.assertIsNone(launch.port_holder(8060, run, "win32"))
        self.assertEqual(launch.port_holder(18050, run, "win32")[0], 99)

    def test_the_holder_elsewhere_from_lsof_and_ps_or_nothing_without_them(self):
        def run(command, **options):
            output = {"lsof": "4242\n", "ps": "python start.py --mode dashboard\n"}[command[0]]
            return subprocess.CompletedProcess(command, 0, stdout=output)

        self.assertEqual(launch.port_holder(8050, run, "linux"), (4242, "python start.py --mode dashboard"))

        def without_ps(command, **options):
            if command[0] == "ps":
                raise FileNotFoundError(command[0])
            return run(command)

        self.assertEqual(launch.port_holder(8050, without_ps, "darwin"), (4242, None))  # the PID alone
        self.assertIsNone(launch.port_holder(8050, mock.Mock(side_effect=FileNotFoundError), "linux"))  # no lsof
        self.assertIsNone(launch.port_holder(8050, lambda command, **options: subprocess.CompletedProcess(command, 1, stdout=""), "linux"))

    def test_the_hint_names_the_holder_and_the_way_out(self):
        self.assertEqual(
            launch.port_hint(8050, (4242, "python start.py")),
            "Port 8050 is already in use (PID 4242, command line python start.py). Stop it or use --port N",
        )
        self.assertEqual(launch.port_hint(8050, (4242, None)), "Port 8050 is already in use (PID 4242). Stop it or use --port N")
        self.assertEqual(launch.port_hint(8050, None), "Port 8050 is already in use. Stop it or use --port N")

    def test_choose_port_moves_where_the_app_moves_and_stops_where_it_does_not(self):
        held = {8050, 8051}.__contains__
        holder = mock.Mock(return_value=(4242, "python start.py"))
        messages = []
        self.assertEqual(launch.choose_port(None, True, in_use=held, holder=holder, output=messages.append), 8052)
        self.assertEqual(messages, ["Port 8050 is already in use (PID 4242, command line python start.py): serving on http://127.0.0.1:8052/"])
        with self.assertRaises(SystemExit) as stop:  # an app that does not move
            launch.choose_port(None, False, in_use=held, holder=holder, output=messages.append)
        self.assertEqual(stop.exception.code, launch.port_hint(8050, (4242, "python start.py")))
        with self.assertRaises(SystemExit):  # a named port that is held stops, moving or not
            launch.choose_port(8051, True, in_use=held, holder=holder)
        messages.clear()
        self.assertEqual(launch.choose_port(None, False, in_use=held, start=8060, holder=holder, output=messages.append), 8060)
        self.assertEqual(launch.choose_port(8070, False, in_use=held, holder=holder, output=messages.append), 8070)
        self.assertEqual(messages, [])  # a free port is taken without a word
        with mock.patch.object(launch, "port_in_use", return_value=False):  # looked up at the call
            self.assertEqual(launch.choose_port(None, False), launch.DASH_PORT)


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
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir / "data")  # neither: next to the launcher
        (launcher_dir.parent / "data").mkdir()
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir.parent / "data")  # the one above it, where there is none next to it
        (launcher_dir / "data").mkdir()
        self.assertEqual(launch.data_dir(launcher_dir), launcher_dir / "data")  # next to the launcher comes first
        self.assertEqual((launch.STATE_FOLDER, launch.CACHE_FOLDER), ("state", "cache"))


class BackupQuestionTest(TempDirTest):
    """The backup question (prepare_state) on the state files a launcher names: start.py's the dashboard's alone, two
    names here (PAIR); a file of the folder that is not named is never touched."""

    PAIR = ("user_state.json", "dashboard_state.json")

    def setUp(self):
        super().setUp()
        self.state = self.tmp / "user_state.json"
        self.model_state = self.tmp / "model_state.json"
        self.log = self.tmp / "episodes.csv"
        for path in (self.state, self.model_state, self.log):
            path.write_text("{}", encoding="utf-8")
        self.messages = []

    def prepare(self, answer=None, names=PAIR):
        ask = mock.Mock(side_effect=EOFError) if answer is None else mock.Mock(return_value=answer)
        return launch.prepare_state(self.tmp, ask, self.messages.append, names), ask

    def test_start_pys_state_file_is_the_dashboards_alone(self):
        self.assertEqual(launch.STATE_NAMES, ("dashboard_state.json",))
        dashboard = self.tmp / "dashboard_state.json"
        dashboard.write_text("{}", encoding="utf-8")
        backups = launch.prepare_state(self.tmp, mock.Mock(return_value="y"), self.messages.append)
        self.assertEqual([path.name for path in backups], ["dashboard_state.json_01.bak"])
        self.assertTrue(self.state.exists())  # a file start.py does not name stays

    def test_backup_names_count_up(self):
        self.assertEqual(launch.backup_path(self.state).name, "user_state.json_01.bak")
        (self.tmp / "user_state.json_01.bak").write_text("{}", encoding="utf-8")
        self.assertEqual(launch.backup_path(self.state).name, "user_state.json_02.bak")

    def test_yes_backs_up_the_state_file_and_starts_fresh_the_model_state_and_the_log_stay(self):
        backups, ask = self.prepare("y")
        self.assertFalse(self.state.exists())
        self.assertEqual([path.name for path in backups], ["user_state.json_01.bak"])
        self.assertTrue(all(path.exists() for path in backups))
        self.assertTrue(self.log.exists() and self.model_state.exists())
        self.assertEqual(ask.call_args[0][0], "Backup existing user_state.json? (y/n): ")
        self.assertEqual(self.messages[-1], "Starting with fresh state files.")

    def test_no_keeps_the_state_files(self):
        backups, _ask = self.prepare("n")
        self.assertEqual(backups, [])
        self.assertTrue(self.state.exists())

    def test_no_input_keeps_the_state_files(self):
        backups, _ask = self.prepare()
        self.assertEqual(backups, [])
        self.assertTrue(self.state.exists())
        self.assertIn("No input available", self.messages[-1])

    def test_without_a_state_file_nothing_is_asked(self):
        self.state.unlink()
        backups, ask = self.prepare("y")
        self.assertEqual(backups, [])
        ask.assert_not_called()
        self.assertTrue(self.log.exists() and self.model_state.exists())

    def test_two_named_state_files_are_backed_up_together(self):
        dashboard = self.tmp / "dashboard_state.json"
        dashboard.write_text("{}", encoding="utf-8")
        backups, ask = self.prepare("y")
        self.assertEqual([path.name for path in backups], ["user_state.json_01.bak", "dashboard_state.json_01.bak"])
        self.assertFalse(dashboard.exists())
        self.assertEqual(ask.call_args[0][0], "Backup existing user_state.json, dashboard_state.json? (y/n): ")

    def test_one_state_file_alone_is_backed_up_alone(self):
        dashboard = self.tmp / "dashboard_state.json"
        dashboard.write_text("{}", encoding="utf-8")
        self.state.unlink()
        backups, _ask = self.prepare("yes")
        self.assertEqual([path.name for path in backups], ["dashboard_state.json_01.bak"])


if __name__ == "__main__":
    unittest.main()
