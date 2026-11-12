"""The results files (app.results; the csv module, nothing of Dash), read as the dashboard
reads them: the current course's rows, each level's newest run, and the hints without a file, a folder or the
level's rows. The small results files the dashboard's tests read are written here too (write_results, PROMPT_ROWS)."""

import model_support

import tempfile
import unittest
from pathlib import Path

from app.headlines import headlines
from app.panels import s2_label
from app.results import NO_FOLDER, Results, course_note, course_row, newest


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
               "kappa", "accuracy", "rated", "unrated", "hand_judged", "hand_within_deck", "hand_within_pairs", "hand_diff", "hand_diff_low", "hand_diff_high",
               "hand_coin_diff", "hand_coin_diff_low", "hand_coin_diff_high", "notes"]
    write("difficulty.csv", columns, [
        {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "D1", "variant": "text (chosen)", "scope": "all", "features": "text", "label_rule": "deck shares",
         "folds": "5", "repeats": "3", "seed": "42", "macro_f1": "0.593", "macro_f1_sd": "0.018", "kappa": "0.413", "accuracy": "0.7", "hand_within_deck": "0.75", "hand_within_pairs": "77",
         "hand_diff": "0.000", "hand_diff_low": "0.000", "hand_diff_high": "0.000", "hand_coin_diff": "0.253", "hand_coin_diff_low": "0.130",
         "hand_coin_diff_high": "0.370", "notes": note},
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
    "date,run,model,variant,params,self_topic_at1,label_at1,label_at3,hand_right,hand_wrong,hand_missed,hand_diff,hand_diff_low,"
    "hand_diff_high,hand_tfidf_diff,hand_tfidf_diff_low,hand_tfidf_diff_high,hand_tfidf_wrong_diff,hand_tfidf_wrong_diff_low,"
    "hand_tfidf_wrong_diff_high,judged_requests,ms_per_request,notes\n"
    "2026-11-01,2026-11-01T09:00,P1,TF-IDF: card,routes=card,0.5,0.300,0.500,0.6,0.3,0.1,0.000,0.000,0.000,,,,,,,4,1.0,decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,TF-IDF: card,routes=card,0.7,0.410,0.620,0.75,0.2,0.05,0.000,0.000,0.000,,,,,,,4,1.0,decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,TF-IDF: all,routes=card+centroid+text,0.8,0.410,0.620,0.5,0.4,0.1,-0.250,-0.750,0.000,,,,,,,4,2.0,"
    "decks 01 to 03\n"
    "2026-11-02,2026-11-02T09:00,P1,hybrid-minilm: card,routes=card,0.8,0.450,0.660,0.76,0.055,0.185,0.010,-0.100,0.120,0.010,-0.090,"
    "0.110,-0.145,-0.250,-0.040,4,2.0,decks 01 to 03\n"
)


class ResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls._tmp.name)
        cls.note = course_note(model_support.DECK_IDS)
        write_results(cls.folder / "results", cls.note)
        cls.results = Results(cls.folder / "results", cls.note, chosen={"S2": s2_label()})

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_note_marks_the_course(self):
        self.assertEqual(self.note, "decks 01 to 03")

    def test_a_level_shows_its_newest_run_of_the_course(self):
        run = self.results.run("S1")
        self.assertEqual({row["run"] for row in run}, {"2026-11-01T10:00"})  # the newest run, whole
        self.assertEqual([row["variant"] for row in run if row["model"] == "S1"], ["unigrams", "chosen: uni+bigrams"])
        self.assertNotIn("chosen: old", [row["variant"] for row in run])  # the 48-deck row and the older run stay out
        self.assertEqual(len([row for row in run if row["model"] == "B0"]), 2)  # the floor's rows of the same run
        self.assertEqual(self.results.run("S2"), [])  # the file, but no S2 rows
        self.assertIn("No S2 rows of this course in results.csv", self.results.no_rows("S2"))
        self.assertIsNone(Results(self.folder / "nowhere", self.note).run("S1"))
        self.assertEqual(self.results.chosen, {"S2": s2_label()})  # the variant the app runs, where the names do not say

    def test_a_run_is_the_run_column_or_the_date_and_a_models_newest_is_the_newest_that_holds_it(self):
        rows = [{"date": "2026-10-26", "model": "S1"}, {"date": "2026-11-01", "run": "2026-11-01T10:00", "model": "S1"},
                {"date": "2026-11-02", "run": "2026-11-02T09:00", "model": "C1"}]
        self.assertEqual([row["model"] for row in newest(rows)], ["C1"])
        self.assertEqual([row.get("run") for row in newest(rows, "S1")], ["2026-11-01T10:00"])  # a later run without S1 hides none
        self.assertEqual(newest(rows, "T1"), [])
        self.assertEqual(newest(rows[:1])[0]["date"], "2026-10-26")  # a row written before the run column: its date

    def test_a_rows_scope_is_the_course_or_a_deck(self):
        self.assertTrue(course_row({"scope": "all"}))
        self.assertTrue(course_row({}))  # a row written before the column
        self.assertFalse(course_row({"scope": "deck 01"}))
        deck_rows = [row for row in self.results.run("D1") if not course_row(row)]
        self.assertEqual({row["scope"] for row in deck_rows}, {"deck 01"})

    def test_without_a_note_every_row_counts(self):
        every = Results(self.folder / "results", None)
        self.assertEqual(len(every.rows["results.csv"]), 9)  # the 48-deck course's row too
        self.assertEqual(len(self.results.rows["results.csv"]), 8)

    def test_without_a_file_or_a_folder_the_hint_says_how_to_make_one(self):
        results = Results(self.folder / "nowhere", self.note)
        self.assertIsNone(results.run("S1"))
        self.assertIn("No results.csv in", results.hint("results.csv"))
        self.assertIn("uv run evaluate.py", results.hint("results.csv"))
        self.assertEqual(Results(None, self.note).hint("topics.csv"), NO_FOLDER)
        self.assertIsNone(headlines(results, "S1"))


if __name__ == "__main__":
    unittest.main()
