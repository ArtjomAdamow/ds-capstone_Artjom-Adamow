"""The results files (app.results; the csv module, nothing of Dash or the viewer), read as the dashboard and the
models view read them: the current course's rows, the newest run, the transposed tables with their floor, the chosen
variant and the app's pick marked, the glossary, and the hints without a file or a folder. The small results files
the dashboard's and the models view's tests read are written here too (write_results, PROMPT_ROWS)."""

import model_support

import tempfile
import unittest
from pathlib import Path

from app.headlines import headlines
from app.panels import d2_label, s2_label
from app.results import NO_FOLDER, Results, course_note, read_glossary


def write_results(folder: Path, note: str) -> None:
    """Three small results files: an older run without a run column, a 48-deck row without the note, and the newest
    run, with a deck's and a topic's name in its cells, the floors and the hand columns the headlines read."""
    import csv

    folder.mkdir(parents=True, exist_ok=True)

    def write(name, columns, rows):
        with (folder / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})

    columns = ["date", "run", "model", "variant", "params", "seed", "precision_at_5", "kept_precision", "hand_precision", "hand_p1",
               "hand_diff", "hand_diff_low", "hand_diff_high", "judged_cards", "trust_decks", "trust_cards", "overlap_k2", "lift_k2", "seconds", "notes"]
    write("results.csv", columns, [
        {"date": "2026-10-22", "model": "S1", "variant": "chosen: old", "precision_at_5": "0.1", "notes": "the 48-deck course"},
        {"date": "2026-10-26", "model": "S1", "variant": "unigrams", "precision_at_5": "0.4", "notes": f"x; {note}"},
        {"date": "2026-10-26", "model": "S1", "variant": "chosen: uni+bigrams", "precision_at_5": "0.5", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "B0", "variant": "random positions", "seed": "42", "precision_at_5": "0.2", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "B0", "variant": "random positions", "seed": "1", "precision_at_5": "0.4", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S1", "variant": "unigrams", "params": "a", "precision_at_5": "0.5", "kept_precision": "0.55", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S1", "variant": "chosen: uni+bigrams", "params": "b", "precision_at_5": "0.6", "kept_precision": "0.65",
         "hand_precision": "0.68", "hand_p1": "0.92", "notes": f"x; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "C1", "variant": "model", "params": "k=3", "trust_decks": "0.8", "trust_cards": "0.9", "overlap_k2": "0.7", "lift_k2": "0.5", "notes": f"deck Alpha and Lists; {note}"},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "C1", "variant": "random", "params": "k=3", "seed": "42", "trust_decks": "0.4", "trust_cards": "0.5", "overlap_k2": "0.3", "lift_k2": "0.000", "notes": f"deck Alpha and Lists; {note}"},
    ])
    columns = ["date", "run", "model", "scope", "cards", "topics", "agreement", "kept_share", "moved", "notes"]
    write("topics.csv", columns, [
        {"date": "2026-10-26", "model": "T1", "scope": "all", "cards": "13", "agreement": "0.9", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "all", "cards": "13", "topics": "7", "agreement": "0.640", "moved": "0", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "chance", "scope": "all", "cards": "13", "topics": "7", "agreement": "0.220", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "1-NN", "scope": "all", "cards": "13", "topics": "7", "agreement": "0.620", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "focus 25 %", "cards": "3", "agreement": "0.890", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "deck 01 Alpha", "cards": "5", "topics": "3", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "deck 01", "cards": "5", "topics": "3", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "chance", "scope": "deck 01", "cards": "5", "topics": "3", "agreement": "0.300", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "stack 1", "cards": "6", "agreement": "1.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "T1", "scope": "margin >= 0.05", "cards": "12", "kept_share": "0.923", "agreement": "1.000", "notes": note},
    ])
    columns = ["date", "run", "model", "variant", "scope", "params", "features", "label_rule", "folds", "repeats", "seed", "macro_f1", "macro_f1_sd",
               "kappa", "accuracy", "rated", "unrated", "hand_judged", "hand_within_deck", "hand_within_pairs", "hand_diff", "hand_diff_low", "hand_diff_high", "notes"]
    write("difficulty.csv", columns, [
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "scope": "all", "features": "text", "label_rule": "deck shares",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.593", "macro_f1_sd": "0.018", "kappa": "0.413", "accuracy": "0.7", "hand_within_deck": "0.75", "hand_within_pairs": "77",
         "hand_diff": "0.000", "hand_diff_low": "0.000", "hand_diff_high": "0.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "scope": "deck 01", "features": "text", "label_rule": "deck shares",
         "folds": "5", "repeats": "3", "seed": "42", "rated": "12", "unrated": "7", "hand_judged": "7", "hand_within_deck": "0.80", "hand_within_pairs": "16",
         "hand_diff": "0.000", "hand_diff_low": "0.000", "hand_diff_high": "0.000", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text+options", "scope": "deck 01", "features": "text+options", "label_rule": "deck shares",
         "folds": "5", "repeats": "3", "seed": "42", "rated": "12", "unrated": "7", "hand_judged": "7", "hand_within_deck": "0.85", "hand_within_pairs": "16",
         "hand_diff": "0.050", "hand_diff_low": "-0.100", "hand_diff_high": "0.200", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion, base", "scope": "all", "params": "model=x", "features": "premise", "label_rule": "deck shares",
         "folds": "0", "repeats": "0", "macro_f1": "0.410", "macro_f1_sd": "0.000", "kappa": "0.01", "accuracy": "0.5", "hand_within_deck": "0.80", "hand_within_pairs": "77",
         "hand_diff": "0.052", "hand_diff_low": "-0.020", "hand_diff_high": "0.120", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion, large", "scope": "all", "params": "model=y", "features": "premise", "label_rule": "deck shares",
         "folds": "0", "repeats": "0", "macro_f1": "0.420", "macro_f1_sd": "0.000", "kappa": "0.018", "accuracy": "0.55", "hand_within_deck": "0.688", "hand_within_pairs": "77",
         "hand_diff": "-0.065", "hand_diff_low": "-0.358", "hand_diff_high": "0.224", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion, base", "scope": "all", "params": "model=x", "features": "premise", "label_rule": "most likely class",
         "folds": "0", "repeats": "0", "macro_f1": "0.380", "macro_f1_sd": "0.000", "accuracy": "0.45", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D2", "variant": "criterion, base", "scope": "deck 01", "params": "model=x", "features": "premise", "label_rule": "deck shares",
         "folds": "0", "repeats": "0", "rated": "12", "unrated": "7", "hand_judged": "7", "hand_within_deck": "0.90", "hand_within_pairs": "16",
         "hand_diff": "0.100", "hand_diff_low": "-0.100", "hand_diff_high": "0.300", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "features": "text", "label_rule": "most likely class",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.530", "macro_f1_sd": "0.017", "accuracy": "0.62", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "baseline", "variant": "always easy", "label_rule": "constant",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.27", "macro_f1_sd": "0.00", "kappa": "0.000", "accuracy": "0.667", "notes": note},
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "hand", "variant": "the criterion against the authored levels", "label_rule": "authored",
         "hand_judged": "24", "notes": note},
    ])


PROMPT_ROWS = (
    "date,run,model,variant,params,self_topic_at1,hand_right,hand_wrong,hand_missed,hand_diff,hand_diff_low,hand_diff_high,judged_requests,ms_per_request,notes\n"
    "2026-11-01,2026-11-01T09:00,P1,TF-IDF: card,routes=card,0.5,0.6,0.3,0.1,0.000,0.000,0.000,4,1.0,decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,TF-IDF: card,routes=card,0.7,0.75,0.2,0.05,0.000,0.000,0.000,4,1.0,decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,TF-IDF: all,routes=card+centroid+text,0.8,0.5,0.4,0.1,-0.250,-0.750,0.000,4,2.0,decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,hybrid-minilm: card,routes=card,0.8,0.76,0.055,0.185,0.010,-0.100,0.120,4,2.0,decks 01 to 03\n"
)


class ResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls._tmp.name)
        cls.note = course_note(model_support.DECK_IDS)
        write_results(cls.folder / "results", cls.note)
        cls.results = Results(cls.folder / "results", cls.note, chosen={"D2": d2_label(), "S2": s2_label()})

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_note_marks_the_course(self):
        self.assertEqual(self.note, "decks 01 to 03")

    def test_s1_stands_next_to_b0s_mean_in_the_newest_run_with_the_chosen_variant_marked(self):
        (table,) = self.results.tables("S1")
        self.assertEqual(table.heads, ["B0 random positions (mean of 2 seeds)", "unigrams", "chosen: uni+bigrams"])
        self.assertEqual(table.chosen, [False, False, True])
        self.assertEqual(table.subtitle, "run 2026-11-01T10:00")
        rows = dict(table.rows)
        self.assertEqual(rows["precision_at_5"], ["0.300", "0.5", "0.6"])
        self.assertEqual(rows["kept_precision"], ["", "0.55", "0.65"])
        self.assertNotIn("hand_p3", rows)  # a measure the run left empty is no row
        self.assertNotIn("old", str(table.heads))  # the 48-deck row and the older run stay out
        self.assertEqual(self.results.run("S1")[0]["run"], "2026-11-01T10:00")
        self.assertEqual(self.results.run("S2"), [])  # the file, but no S2 rows
        self.assertIsNone(Results(self.folder / "nowhere", self.note).run("S1"))

    def test_the_app_pick_is_marked_in_d2s_table_next_to_the_reference(self):
        (tags,) = self.results.tables("D2")
        self.assertEqual(tags.heads, ["D1 text (chosen) · deck shares", "D2 criterion, base · deck shares", "D2 criterion, large · deck shares", "D2 criterion, base · most likely class"])
        self.assertEqual(tags.chosen, [True, False, True, False])  # the reference and the app's pick
        rows = dict(tags.rows)
        self.assertEqual(rows["hand_within_deck (pairs)"], ["0.75 (77)", "0.80 (77)", "0.688 (77)", ""])
        self.assertEqual(rows["hand_diff [low, high]"], ["0.000 [0.000, 0.000]", "0.052 [-0.020, 0.120]", "-0.065 [-0.358, 0.224]", ""])
        _course, deck = self.results.tables("D2", "01")
        self.assertEqual((deck.heads, deck.chosen), (["D1 text (chosen) · deck shares", "D2 criterion, base · deck shares"], [True, False]))

    def test_t1s_tables_are_the_course_or_the_entered_deck_with_the_stacks_the_margins_and_the_focus(self):
        course, stacks, margins, focus = self.results.tables("T1")
        self.assertEqual((course.title, course.heads, course.chosen), ("T1 next to the judges, the course", ["T1", "1-NN", "chance"], [True, False, False]))
        self.assertEqual(dict(course.rows)["agreement"], ["0.640", "0.620", "0.220"])
        self.assertEqual((stacks.heads, dict(stacks.rows)["cards"]), (["stack 1"], ["6"]))
        self.assertEqual((margins.heads, dict(margins.rows)["kept_share"]), (["margin >= 0.05"], ["0.923"]))
        self.assertEqual(focus.heads, ["focus 25 %"])
        deck, _stacks, _margins, _focus = self.results.tables("T1", "01")
        self.assertEqual((deck.title, deck.heads), ("T1 next to the judges, deck 01", ["T1", "chance"]))

    def test_the_glossary_has_a_row_for_every_measure_column_and_none_for_the_bookkeeping(self):
        glossary = read_glossary()
        import evaluate  # next to the package, as the tests run

        bookkeeping = {"date", "run", "model", "variant", "scope", "params", "features", "label_rule", "folds", "repeats", "seed", "seconds", "notes"}
        for columns in (evaluate.COLUMNS, evaluate.TOPICS_COLUMNS, evaluate.DIFFICULTY_COLUMNS, evaluate.PROMPT_COLUMNS):
            for column in columns:
                self.assertEqual(column in glossary, column not in bookkeeping, column)
        self.assertEqual(glossary["lift_k2"], glossary["lift_k3"])  # one row for the columns of one measure
        table = self.results.tables("D1")[0]
        self.assertEqual(table.row_columns[0], ["macro_f1", "macro_f1_sd"])  # the columns behind a row, its description's key

    def test_without_a_file_or_a_folder_the_hint_says_how_to_make_one(self):
        results = Results(self.folder / "nowhere", self.note)
        self.assertIsNone(results.tables("S1"))
        self.assertIn("No results.csv in", results.hint("results.csv"))
        self.assertIn("uv run evaluate.py", results.hint("results.csv"))
        self.assertEqual(Results(None, self.note).hint("topics.csv"), NO_FOLDER)
        self.assertIsNone(headlines(results, "S1"))


if __name__ == "__main__":
    unittest.main()
