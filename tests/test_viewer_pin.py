"""The capstone's sputnik pin.

The tag pyproject.toml pins is app.launch.VIEWER_RELEASE or newer (the release whose hooks the
glue uses), uv.lock locks that same tag, and an installed sputnik is the pinned release (a stale
install after a pin bump wants `uv sync --all-groups --extra viewer`).
"""

import re
import tomllib
import unittest
from importlib import metadata
from pathlib import Path

from app import launch

ROOT = Path(__file__).resolve().parents[1]
LOCKED_TAG = re.compile(r"sputnik\.git\?tag=([^\"#&]+)")


def release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def pinned_tag() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["uv"]["sources"]["sputnik"]["tag"]


class ViewerPinTest(unittest.TestCase):
    def test_the_pin_is_the_glues_release_or_newer(self):
        self.assertGreaterEqual(release(pinned_tag()), release(launch.VIEWER_RELEASE))

    def test_the_lock_locks_the_pinned_tag(self):
        locked = set(LOCKED_TAG.findall((ROOT / "uv.lock").read_text(encoding="utf-8")))
        self.assertEqual(locked, {pinned_tag()})

    def test_an_installed_sputnik_is_the_pinned_release(self):
        try:
            installed = metadata.version("sputnik")
        except metadata.PackageNotFoundError:
            self.skipTest("sputnik is not installed (the viewer extra)")
        self.assertEqual(release(installed), release(pinned_tag()))


if __name__ == "__main__":
    unittest.main()
