"""The models import nothing of the viewer (roadmap step 21, the split): no source of app.models or
its card record imports `project`, and the models import, fit and measure in a process where `project` cannot be
imported at all, as in the capstone's CI without the private viewer package. The glue, by contrast, is the part
that imports the viewer."""

import model_support  # noqa: F401  (the helper's package on the path)

import re
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = model_support.MODULE_DIR / "app"
MODEL_SOURCES = sorted((PACKAGE / "models").glob("*.py")) + [PACKAGE / "cards.py", PACKAGE / "__init__.py"]
GLUE_SOURCES = sorted((PACKAGE / "glue").glob("*.py"))
VIEWER_IMPORT = re.compile(r"^\s*(from|import)\s+project\b", re.MULTILINE)
TESTS_DIR = Path(__file__).resolve().parent
# A process in which the viewer cannot be imported: every model module imported, the course model fitted on the
# table fixture, the difficulty model measured, and no `project` module loaded afterwards
PROBE = """
import sys
sys.modules["project"] = None  # `import project` raises ImportError from here on
sys.path.insert(0, {tests!r})
import model_support
from app.models import clustering, course_model, difficulty_model, evaluation, layout, similar, topic_model
from app.cards import CardRecord
model = model_support.course_model()
assert set(model.deck_points) == {{"01", "02", "03"}} and model.card_points and model.deck_links
assert model_support.difficulty_report().measured
loaded = sorted(name for name, module in sys.modules.items() if module is not None and name.split(".")[0] == "project")
assert not loaded, loaded
print("ok")
"""


class ModelsWithoutTheViewerTest(unittest.TestCase):
    def test_no_model_source_imports_the_viewer(self):
        self.assertGreaterEqual(len(MODEL_SOURCES), 9)
        for path in MODEL_SOURCES:
            with self.subTest(file=path.name):
                self.assertIsNone(VIEWER_IMPORT.search(path.read_text(encoding="utf-8")), f"{path.name} imports the viewer")

    def test_the_glue_is_the_part_that_imports_the_viewer(self):
        importing = [path.name for path in GLUE_SOURCES if VIEWER_IMPORT.search(path.read_text(encoding="utf-8"))]
        self.assertIn("composition.py", importing)
        self.assertIn("course.py", importing)
        self.assertNotIn("difficulty_metrics.py", importing)  # Dash and Plotly, but nothing of the viewer's own

    def test_the_models_import_fit_and_measure_where_the_viewer_cannot_be_imported(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", PROBE.format(tests=str(TESTS_DIR))], capture_output=True, text=True,
            cwd=str(model_support.MODULE_DIR), timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
