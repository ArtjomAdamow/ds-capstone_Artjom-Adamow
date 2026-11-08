"""The glue on viewers of every release: it imports on the installed viewer whatever its release, takes only the
newer hooks that viewer's compose has (glue.composition.viewer_hooks), and makes the Models link only where the
viewer takes links; on a viewer as old as sputnik 0.3.0 (no TopBarLink, ModuleSection or SectionEntry,
none of the newer hooks, no ModulePrompt, PromptTarget or PromptAnswer, no ModelState of step 33; a subprocess with
those names taken out of the installed viewer) it still imports, takes no hook and makes no link. Which release the capstone pins is the capstone's own check, not this one's."""

import model_support  # noqa: F401  (the helper's package on the path)

import importlib.util
import subprocess
import sys
import unittest

ROOT = model_support.MODULE_DIR
HAS_VIEWER = importlib.util.find_spec("project") is not None
NEWER_NAMES = ("TopBarLink", "ModuleSection", "SectionEntry", "ModulePrompt", "PromptTarget", "PromptAnswer", "ModelState")
URL = "http://127.0.0.1:8051/"
# A viewer as old as sputnik 0.3.0: the installed one without the newer names in its actions and without the newer
# hooks in compose; the glue imports, takes no hook and makes no Models link
OLD_VIEWER = """
import sys
import project.composition  # the viewer loaded first: its own modules may import the newer names
import project.services.actions as actions
for name in {names!r}:
    if hasattr(actions, name):
        delattr(actions, name)
sys.path.insert(0, {root!r})
from app.glue import composition

def old_compose(base_dir, boot_id, cards_path, state_path, card_positions=None, action=None, overview_back=None,
                refill_groups=None, after_save=None):
    pass

assert composition.viewer_hooks(old_compose) == set(), composition.viewer_hooks(old_compose)
assert composition.models_links({{composition.MODELS_URL: {url!r}}}, old_compose) == []
print("ok")
"""


@unittest.skipUnless(HAS_VIEWER, "the viewer is not installed")
class GlueOnTheViewerTest(unittest.TestCase):
    def test_the_glue_imports_on_the_installed_viewer_and_takes_its_hooks(self):
        from app.glue import composition

        self.assertLessEqual(composition.viewer_hooks(), set(composition.NEWER_HOOKS))

    def test_the_models_link_needs_the_url_and_a_viewer_that_takes_links(self):
        from app.glue import composition

        self.assertEqual(composition.models_links({}), [])
        links = composition.models_links({composition.MODELS_URL: URL})
        if composition.LINKS_HOOK not in composition.viewer_hooks():
            self.assertEqual(links, [])
            return
        [link] = links
        self.assertEqual((link.label, link.url), (composition.MODELS_LABEL, URL))

    def test_on_an_older_viewer_the_glue_imports_and_leaves_the_newer_hooks_out(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", OLD_VIEWER.format(names=NEWER_NAMES, root=str(ROOT), url=URL)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
