"""The offline evaluation (evaluate.py, next to the package) and its measures (app.models.evaluation), on the
fixtures: a results file whose columns grew; the hand labels of D1 (the labels file, the
judging form, the merge of its marks) and the hand columns of D1's rows and the `hand` row; the hand marks of the
similarity part (the marks file, the judging form, the merge, a rewrite with a variant removed); the measures (the
random draw, the hand precision per card, the paired bootstrap, the fixed-k overlap and the lift on a hand-made
layout); the rows of the similarity part; D2's rows per model and template with the deck rows; P1's part (every
variant's rows, the request file read against T1's membership and protected, the review form and its marks, the file
alone, the paired difference to the card route); and a run on the test course's table, with and without the hand
files, in a process of its own, the demo's deck 00 next to it never read."""

import model_support  # noqa: F401  (the package on the path, and the folder next to it with evaluate.py)

import contextlib
import csv
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate
from app.models import evaluation, similar
from app.models.difficulty_model import LEVELS
from app.models.prompt_model import ENTERED, NO_MATCH, ROUTES, SELECTED, Answer, TopicScore

TODAY = "2026-11-01"
TESTS_DIR = Path(__file__).resolve().parent
TAG = f"  —  {evaluate.NOT_JUDGED}  —  "  # how the hand-check form tags a candidate that is new under a judged card


def mark(text: str, card_id: str, level: str) -> str:
    """The form with the card's level box ticked."""
    head, entry = text.split(f"### {card_id} (", 1)
    return head + f"### {card_id} (" + entry.replace(f"- [ ] {level}", f"- [x] {level}", 1)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class WriteRowsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "results.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_rows_are_appended_and_a_file_with_fewer_columns_is_written_again_with_its_rows(self):
        evaluate.write_rows([{"date": "d1", "model": "m"}], self.path, ["date", "model"])
        evaluate.write_rows([{"date": "d2", "model": "n"}], self.path, ["date", "model"])
        self.assertEqual(read_csv(self.path), [{"date": "d1", "model": "m"}, {"date": "d2", "model": "n"}])
        evaluate.write_rows([{"date": "d3", "model": "o", "extra": "1"}], self.path, ["date", "model", "extra"])
        self.assertEqual(read_csv(self.path), [{"date": "d1", "model": "m", "extra": ""}, {"date": "d2", "model": "n", "extra": ""},
                                               {"date": "d3", "model": "o", "extra": "1"}])
        self.assertEqual(self.path.read_text(encoding="utf-8").count("date,model"), 1)  # one header


class HandLabelsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        folder = Path(self._tmp.name)
        self.form = folder / "hand_labels.md"
        self.file = folder / "hand_labels.csv"
        self.cards = model_support.difficulty_course()
        self.unrated = [card for card in self.cards if card.stack == 2]

    def tearDown(self):
        self._tmp.cleanup()

    def write_form(self, labels=None):
        return evaluate.write_hand_labels_form(self.cards, labels or {}, TODAY, "test", self.form, reference_decks=("Alpha",))

    def test_the_form_lists_the_unrated_quizzes_and_the_reference_decks_rated_ones_blind(self):
        self.assertEqual(self.write_form(), (6, 12))
        text = self.form.read_text(encoding="utf-8")
        self.assertEqual(text.count("\n### "), 18)
        self.assertEqual(text.count("- [ ] easy\n- [ ] medium\n- [ ] hard\n- note:\n"), 18)  # nothing marked
        for entry in text.split("\n### ")[1:]:  # no entry names a level outside its boxes: the authored ones stay blind
            body = [line for line in entry.splitlines() if not line.startswith("- [")]
            self.assertFalse(any(level in line for level in LEVELS for line in body), entry[:30])
        self.assertIn("- (wrong) a  —  why a\n- (correct) b  —  why b\n- (wrong) c\n", text)  # the record's options with their feedback
        self.assertIn("- answer: ", text)  # a rated quiz of the fixture has no options: its answer
        self.assertIn("## The rated quizzes of Alpha, judged blind (12)", text)
        for card in self.unrated:
            self.assertIn(f"\n### {card.id} ({card.deck}): ", text)
            self.assertIn(f"\n> {card.question}\n", text)
        self.assertEqual(evaluate.read_label_form(self.form), {})  # nothing judged yet

    def test_the_marks_of_the_form_are_read_and_a_double_mark_is_skipped(self):
        stored = {"201101": {"level": "easy", "note": "stored note"}}
        self.write_form(stored)
        text = self.form.read_text(encoding="utf-8")
        self.assertIn("- [x] easy\n- [ ] medium\n- [ ] hard\n- note: stored note\n", text)
        text = mark(text, "202101", "hard")
        text = mark(mark(text, "203101", "easy"), "203101", "medium")  # two levels: not a judgement
        self.form.write_text(text, encoding="utf-8")
        self.assertEqual(evaluate.read_label_form(self.form), {
            "201101": {"level": "easy", "note": "stored note"}, "202101": {"level": "hard", "note": ""},
        })

    def test_the_marks_merge_over_the_stored_labels(self):
        stored = {
            "A": {"card": "A", "level": "easy", "judged": "2026-10-01", "judge": "X", "note": "old"},
            "B": {"card": "B", "level": "medium", "judged": "2026-10-01", "judge": "X", "note": "old"},
            "D": {"card": "D", "level": "hard", "judged": "2026-10-01", "judge": "X", "note": ""},
        }
        marks = {"A": {"level": "easy", "note": ""}, "B": {"level": "hard", "note": "changed"}, "C": {"level": "medium", "note": ""}}
        merged = evaluate.merge_labels(stored, marks, TODAY, "Y")
        self.assertEqual(merged["A"], stored["A"])  # the same level: the row stays, its note too
        self.assertEqual(merged["B"], {"card": "B", "level": "hard", "judged": TODAY, "judge": "Y", "note": "changed"})
        self.assertEqual(merged["C"], {"card": "C", "level": "medium", "judged": TODAY, "judge": "Y", "note": ""})
        self.assertEqual(merged["D"], stored["D"])  # not in the form: kept
        self.assertEqual(stored["B"]["level"], "medium")  # the stored rows are not changed in place

    def test_the_labels_file_round_trip_in_id_order_keeps_a_level_outside_the_three(self):
        labels = {
            "B": {"card": "B", "level": "hard", "judged": TODAY, "judge": "Y", "note": "a, note"},
            "A": {"card": "A", "level": "easy", "judged": "2026-10-01", "judge": "X", "note": ""},
        }
        evaluate.write_labels(labels, self.file)
        self.assertEqual([row["card"] for row in read_csv(self.file)], ["A", "B"])
        self.assertEqual(evaluate.read_labels(self.file), labels)
        with self.file.open("a", newline="", encoding="utf-8") as handle:
            handle.write("C,unsure,,,a note by hand\n\n")
        self.assertEqual(evaluate.read_labels(self.file)["C"], {"card": "C", "level": "unsure", "judged": "", "judge": "", "note": "a note by hand"})
        self.assertEqual(evaluate.hand_measures(evaluate.read_labels(self.file), {"C": "easy"}), ({}, 0))  # kept, not scored
        self.assertEqual(evaluate.read_labels(self.file.with_name("none.csv")), {})

    def test_the_within_deck_order_counts_a_decks_pairs_with_different_hand_levels(self):
        labels = {card: {"level": level} for card, level in
                  (("a1", "easy"), ("a2", "medium"), ("a3", "hard"), ("b1", "easy"), ("b2", "medium"), ("b3", "medium"), ("b4", "unsure"))}
        scores = {"a1": 0.1, "a2": 0.3, "a3": 0.5, "b1": 0.9, "b2": 0.2, "b3": 0.9, "b4": 0.0, "c1": 0.4}  # c1 is not judged
        deck_of = {card: card[0] for card in scores}
        # deck a: 3 pairs, all in the hand order; deck b: easy-medium wrong, easy-medium tied (half), medium-medium no pair
        self.assertEqual(evaluate.within_deck_concordance(labels, scores, deck_of), (3.5 / 5, 5))
        value, pairs = evaluate.within_deck_concordance(labels, {"a1": 0.1, "b1": 0.2}, deck_of)  # one judged quiz per deck: no pair
        self.assertTrue(math.isnan(value) and pairs == 0)

    def test_every_deck_shares_row_with_judged_quizzes_gets_its_deck_rows(self):
        labels = {card.id: {"card": card.id, "level": "hard" if card.id.endswith("102") else "easy", "judged": TODAY, "judge": "Y", "note": ""}
                  for card in self.unrated}
        rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test", labels, deck_ids={"Alpha": "01", "Beta": "02"})
        self.assertTrue(all(row["scope"] for row in rows))
        decks = [row for row in rows if row["scope"] != evaluate.COURSE_SCOPE]
        self.assertEqual({row["scope"] for row in decks}, {"deck 01", "deck 02", "deck Gamma"})  # by id where one is given
        self.assertEqual({row["model"] for row in decks}, {"D1"})
        self.assertEqual(len(decks), 12)  # four feature sets, three decks, the deck-shares rule alone
        for row in decks:
            self.assertEqual((row["label_rule"], row["hand_judged"], row["rated"], row["unrated"]), ("deck shares", 2, 12, 2))
            self.assertEqual((row["hand_within_deck"], row["hand_within_pairs"]), ("1.000", 1))  # the deck's long quiz above its short one
            self.assertTrue(float(row["hand_diff_low"]) <= float(row["hand_diff"]) <= float(row["hand_diff_high"]))
            for column in ("hand_accuracy", "hand_kappa", "predicted_easy", "macro_f1"):
                self.assertNotIn(column, row)  # a readout of the order, nothing else
        chosen = [row for row in decks if row["variant"] == "text (chosen)"]
        self.assertEqual({(row["hand_diff"], row["hand_diff_low"], row["hand_diff_high"]) for row in chosen}, {("0.000", "0.000", "0.000")})
        course = [row for row in rows if row["scope"] == evaluate.COURSE_SCOPE and row["model"] == "D1"]
        self.assertEqual(len(course), 12)  # four feature sets, three label rules

    def test_the_hand_columns_of_d1s_rows_and_the_hand_row(self):
        # Every deck's short unrated quiz easy, its long one hard; Alpha's rated quizzes at their authored levels
        labels = {card.id: {"card": card.id, "level": "hard" if card.id.endswith("102") else "easy", "judged": TODAY, "judge": "Y", "note": ""}
                  for card in self.unrated}
        for card in self.cards:
            if card.deck == "Alpha" and card.difficulty in LEVELS:
                labels[card.id] = {"card": card.id, "level": card.difficulty, "judged": TODAY, "judge": "Y", "note": ""}
        labels["302001"] = {"card": "302001", "level": "unsure", "judged": "", "judge": "", "note": "by hand"}  # never scored
        rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test", labels)
        d1 = [row for row in rows if row["model"] == "D1" and row["scope"] == evaluate.COURSE_SCOPE]
        self.assertEqual(len(d1), 12)  # four feature sets, three label rules
        self.assertFalse(any("tags" in row["variant"] for row in d1))  # D2 supplies no feature to D1
        for row in d1:
            if row["label_rule"] == "most likely class":
                self.assertNotIn("hand_judged", row)  # no predictions of the unrated quizzes by that rule
                continue
            self.assertEqual(row["hand_judged"], 6, row["label_rule"])
            for name in ("hand_macro_f1", "hand_kappa", "hand_accuracy", "hand_spearman"):
                self.assertTrue(-1.0 <= float(row[name]) <= 1.0, name)
            # each deck's long quiz (hand hard) scores above its short one (hand easy): every deck's one pair right
            self.assertEqual((row["hand_within_deck"], row["hand_within_pairs"]), ("1.000", 3), row["label_rule"])
            # the paired difference of the order to the chosen row, with its interval: 0 on the chosen row itself
            self.assertEqual((row["hand_diff"], row["hand_diff_low"], row["hand_diff_high"]),
                             ("0.000", "0.000", "0.000") if row["variant"] == "text (chosen)" else (row["hand_diff"], row["hand_diff_low"], row["hand_diff_high"]))
            self.assertTrue(-1.0 <= float(row["hand_diff_low"]) <= float(row["hand_diff"]) <= float(row["hand_diff_high"]) <= 1.0)
        # D1 calls the short quiz easy and the long one medium (test_difficulty_model): half of the hand levels met
        shares = [row for row in d1 if row["label_rule"] == "deck shares"]
        self.assertEqual({row["hand_accuracy"] for row in shares}, {"0.500"})
        easy = next(row for row in rows if row["model"] == "baseline" and row["variant"] == "always easy")
        self.assertEqual((easy["hand_judged"], easy["hand_accuracy"]), (6, "0.500"))
        self.assertNotIn("hand_within_deck", easy)  # a constant has no order
        similarity = next(row for row in rows if row["model"] == "baseline" and row["variant"] == "similarity")
        self.assertNotIn("hand_judged", similarity)
        hand = next(row for row in rows if row["model"] == "hand")
        self.assertEqual((hand["label_rule"], hand["rated"], hand["unrated"], hand["hand_judged"]), ("authored", 12, 6, 12))
        self.assertEqual((hand["hand_accuracy"], hand["hand_kappa"], hand["hand_macro_f1"]), ("1.000", "1.000", "1.000"))
        self.assertEqual((hand["recall_easy"], hand["recall_medium"], hand["recall_hard"]), ("1.000", "1.000", "1.000"))
        self.assertIn("hand levels easy/medium/hard: 8/3/1 of the rated quizzes judged blind (authored 8/3/1), 3/0/3 of the unrated", hand["notes"])

    def test_without_labels_the_rows_have_no_hand_columns(self):
        rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test")
        self.assertFalse(any("hand_judged" in row for row in rows))
        self.assertFalse(any("hand_diff" in row for row in rows))
        self.assertFalse(any(row["model"] == "hand" for row in rows))
        self.assertTrue(all(row["scope"] == evaluate.COURSE_SCOPE for row in rows))  # no deck rows without labels

    def test_the_paired_bootstrap_of_the_within_deck_order(self):
        labels = {card: {"level": level} for card, level in
                  (("a1", "easy"), ("a2", "medium"), ("a3", "hard"), ("b1", "easy"), ("b2", "medium"), ("b3", "hard"))}
        deck_of = {card: card[0] for card in labels}
        right = {"a1": 0.1, "a2": 0.3, "a3": 0.5, "b1": 0.1, "b2": 0.3, "b3": 0.5}  # every pair in the hand order
        half = {"a1": 0.5, "a2": 0.3, "a3": 0.1, "b1": 0.1, "b2": 0.3, "b3": 0.5}  # deck a reversed, deck b right
        diff, low, high = evaluate.paired_order_bootstrap(labels, half, right, deck_of, draws=200)
        self.assertAlmostEqual(diff, 0.5)  # 6 of 6 against 3 of 6
        self.assertTrue(low <= diff <= high)
        self.assertEqual(evaluate.paired_order_bootstrap(labels, right, right, deck_of, draws=50), (0.0, 0.0, 0.0))
        self.assertTrue(all(math.isnan(value) for value in evaluate.paired_order_bootstrap(labels, right, {"a1": 0.1}, deck_of)))
        self.assertTrue(all(math.isnan(value) for value in evaluate.paired_order_bootstrap({}, right, half, deck_of)))

    def test_the_order_per_deck(self):
        labels = {card: {"level": level} for card, level in (("a1", "easy"), ("a2", "hard"), ("b1", "easy"), ("b2", "medium"), ("c1", "easy"))}
        scores = {"a1": 0.1, "a2": 0.5, "b1": 0.9, "b2": 0.2, "c1": 0.3}
        self.assertEqual(evaluate.within_deck_by_deck(labels, scores, {card: card[0] for card in scores}), {"a": (1.0, 1), "b": (0.0, 1)})


class TagsPartTest(unittest.TestCase):
    """D2's rows per model and template, with their deck rows, the fake pipeline standing in for the models."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cache = Path(cls._tmp.name) / "cache"
        cls.cards = model_support.difficulty_course()
        unrated = [card for card in cls.cards if card.stack == 2]
        cls.labels = {card.id: {"card": card.id, "level": "hard" if card.id.endswith("102") else "easy", "judged": TODAY, "judge": "Y", "note": ""}
                      for card in unrated}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cls.tags = evaluate.difficulty_tags(cls.cards, cls.cache, loader=model_support.fake_classifier_loader)
            cls.rows = evaluate.evaluate_difficulty(cls.cards, TODAY, "test", cls.labels, tags=cls.tags)
        cls.output = output.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rows_of(self, model: str, variant: str) -> list[dict]:
        return [row for row in self.rows if row["model"] == model and row["variant"] == variant]

    def test_the_tags_come_per_model_and_template_and_are_cached(self):
        self.assertEqual(list(self.tags), ["criterion, base", "level names, base", "criterion, large", "level names, large"])
        self.assertEqual(len(self.tags["criterion, base"].scores), 42)
        self.assertEqual((self.tags["criterion, base"].model_name, self.tags["criterion, large"].model_name),
                         (evaluate.zero_shot.BASE_MODEL, evaluate.zero_shot.LARGE_MODEL))
        self.assertEqual(len(list((self.cache / evaluate.embeddings.CACHE_FOLDER).glob("*.npy"))), 4)
        self.assertIn("D2 criterion, base: 42 quizzes tagged by MoritzLaurer/ModernBERT-base", self.output)
        self.assertIn("D2 level names, large: 42 quizzes tagged by MoritzLaurer/ModernBERT-large", self.output)

    def test_d1_takes_no_feature_of_d2(self):
        d1 = [row for row in self.rows if row["model"] == "D1" and row["scope"] == evaluate.COURSE_SCOPE]
        self.assertEqual(len(d1), 12)  # four feature sets, three label rules
        self.assertFalse(any("tags" in row["variant"] or "tags" in row["features"] for row in d1))

    def test_d2_alone_gets_two_rows_per_variant_without_folds_and_its_deck_rows(self):
        for name in self.tags:
            rows = self.rows_of("D2", name)
            self.assertEqual([row["label_rule"] for row in rows], ["deck shares", "most likely class"] + ["deck shares"] * 3)
            shares = rows[0]
            self.assertEqual((shares["features"], shares["folds"], shares["repeats"], shares["seed"]), ("premise", 0, 0, ""))
            self.assertEqual((shares["rated"], shares["unrated"], shares["scope"]), (36, 6, evaluate.COURSE_SCOPE))
            self.assertTrue(shares["params"].startswith(f"model={self.tags[name].model_name}, device="))  # the model and what tagged
            self.assertEqual(shares["macro_f1_sd"], "0.000")  # one pass: no spread
            self.assertEqual(shares["predicted_easy"] + shares["predicted_medium"] + shares["predicted_hard"], 6)
            self.assertEqual(shares["hand_judged"], 6)
            for column in ("hand_macro_f1", "hand_within_deck", "hand_diff", "hand_diff_low", "hand_diff_high"):
                self.assertIn(column, shares, column)
            self.assertEqual(shares["hand_within_pairs"], 3)
            self.assertNotIn("hand_judged", rows[1])
            self.assertEqual([row["scope"] for row in rows[2:]], ["deck Alpha", "deck Beta", "deck Gamma"])
            self.assertEqual({(row["params"], row["hand_within_pairs"], row["hand_judged"]) for row in rows[2:]}, {(shares["params"], 1, 2)})
        self.assertIn("D2 criterion, base", self.output)
        self.assertIn("paired difference to text", self.output)
        self.assertIn("within a deck, per deck: Alpha", self.output)

    def test_without_the_tags_d2_is_skipped_with_a_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test", self.labels)
        self.assertFalse(any(row["model"] == "D2" for row in rows))
        self.assertFalse(any("tags" in row["variant"] for row in rows))
        with without_tagger(), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(evaluate.difficulty_tags(self.cards, self.cache), {})
        self.assertEqual(output.getvalue().count("\n"), 1)  # one line
        self.assertIn(evaluate.D2_SKIPPED, output.getvalue())
        self.assertIn("transformers", output.getvalue())

    def test_a_model_that_does_not_load_loses_its_rows_alone(self):
        def loader(model_name, device):
            if model_name == evaluate.zero_shot.LARGE_MODEL:
                raise OSError("not fetched")
            return model_support.fake_classifier_loader(model_name, device)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            tags = evaluate.difficulty_tags(self.cards, Path(self._tmp.name) / "other", loader=loader)
        self.assertEqual(list(tags), ["criterion, base", "level names, base"])
        self.assertIn(f"{evaluate.D2_SKIPPED} for the large model: ", output.getvalue())
        self.assertIn("OSError", output.getvalue())


# --- the measures ----------------------------------------------------------------------------------------------


def grouped_directions(groups: int = 3, size: int = 4) -> np.ndarray:
    """Unit vectors in 3D in tight groups around the three axes, no two alike: the vectors are their own perfect
    layout, since on the sphere the cosine order is the euclidean order."""
    rows = []
    for group in range(groups):
        for member in range(size):
            vector = np.zeros(3)
            vector[group % 3] = 1.0
            vector[(group + 1) % 3] = 0.05 * (member + 1)
            vector[(group + 2) % 3] = 0.02 * (member + 1) ** 2
            rows.append(vector / np.linalg.norm(vector))
    return np.array(rows)


class MeasuresTest(unittest.TestCase):
    def test_the_random_directions_lie_on_the_sphere_and_follow_the_seed(self):
        points = evaluation.random_directions(200, 42)
        self.assertEqual(points.shape, (200, 3))
        self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1.0))
        self.assertTrue(np.array_equal(points, evaluation.random_directions(200, 42)))
        self.assertFalse(np.array_equal(points, evaluation.random_directions(200, 43)))
        self.assertEqual(evaluation.random_directions(0, 42).shape, (0, 3))
        self.assertTrue((np.abs(points.mean(axis=0)) < 0.15).all())  # spread over the sphere, not one-sided

    def test_the_hand_precision_per_judged_card_and_its_mean(self):
        neighbours = {"a": [("a1", 0.9), ("a2", 0.8), ("a3", 0.7)], "b": [("b1", 0.5)], "c": [("c1", 0.9)]}
        marks = {"a": {"a1", "a3"}, "b": set(), "d": {"x"}}  # c is not judged; d is judged and has no neighbours
        self.assertEqual(evaluation.judged_precision(neighbours, marks, 5), {"a": 0.4, "b": 0.0, "d": 0.0})
        self.assertEqual(evaluation.judged_precision(neighbours, marks, 1), {"a": 1.0, "b": 0.0, "d": 0.0})
        self.assertEqual(evaluation.judged_precision(neighbours, marks, 3), {"a": 2 / 3, "b": 0.0, "d": 0.0})
        self.assertAlmostEqual(evaluate.hand_precision(neighbours, marks, 5), 0.4 / 3)
        self.assertAlmostEqual(evaluate.hand_precision(neighbours, marks, 1), 1 / 3)
        self.assertTrue(math.isnan(evaluate.hand_precision(neighbours, {}, 5)))

    def test_the_paired_bootstrap_interval(self):
        self.assertEqual(evaluation.paired_bootstrap([0.0] * 20), (0.0, 0.0, 0.0))  # a variant against itself
        for value in evaluation.paired_bootstrap([0.2] * 20):
            self.assertAlmostEqual(value, 0.2)  # no spread: the interval is the mean
        differences = [-0.2, 0.0, 0.2, 0.4, -0.4, 0.2, 0.0, 0.2]
        mean, low, high = evaluation.paired_bootstrap(differences)
        self.assertAlmostEqual(mean, 0.05)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertGreaterEqual(low, min(differences))
        self.assertLessEqual(high, max(differences))
        self.assertEqual((mean, low, high), evaluation.paired_bootstrap(differences))  # deterministic
        self.assertTrue(all(math.isnan(value) for value in evaluation.paired_bootstrap([])))

    def test_the_fixed_k_overlap_and_the_lift_on_a_hand_made_layout(self):
        vectors = grouped_directions()  # 12 rows in three groups of four
        perfect = vectors
        floor_layout = vectors[[(row * 5) % 12 for row in range(12)]]  # the same points, the groups mixed
        for k in (2, 3):
            self.assertEqual(evaluation.mean_overlap([(vectors, perfect)], k), (1.0, 1))
        floor, count = evaluation.mean_overlap([(vectors, floor_layout)], 2)
        self.assertEqual(count, 1)
        self.assertLess(floor, 1.0)
        self.assertAlmostEqual(evaluation.lift(1.0, floor), 1.0)
        self.assertAlmostEqual(evaluation.lift(floor, floor), 0.0)
        self.assertAlmostEqual(evaluation.lift((1.0 + floor) / 2, floor), 0.5)
        self.assertTrue(math.isnan(evaluation.lift(0.5, 1.0)))
        self.assertTrue(math.isnan(evaluation.lift(math.nan, floor)))
        # A topic with k + 1 rows (every other row a neighbour, the overlap 1 for any layout) is left out
        three, four = vectors[:3], vectors[:4]
        value, count = evaluation.mean_overlap([(three, three)], 2)
        self.assertTrue(math.isnan(value) and count == 0)
        self.assertEqual(evaluation.mean_overlap([(vectors, floor_layout), (three, three), (four, four)], 2), ((floor + 1.0) / 2, 2))
        self.assertEqual(evaluation.mean_overlap([(four, four)], 3)[1], 0)


# --- the hand marks of the similarity part ---------------------------------------------------------------------


class FakeIndex:
    """A variant that lists one candidate under one card."""

    def __init__(self, card_id: str, candidate: str):
        self.listed = {card_id: [(candidate, 0.5)]}

    def related(self, card_id: str):
        return list(self.listed.get(card_id, []))


def mark_candidate(text: str, card_id: str, candidate: str) -> str:
    """The form with the candidate's box under the card ticked."""
    head, entry = text.split(f"\n## {card_id} (", 1)
    return head + f"\n## {card_id} (" + entry.replace(f"- [ ] {candidate} (", f"- [x] {candidate} (", 1)


def mark_none(text: str, card_id: str) -> str:
    """The form with the card's `none` line ticked."""
    head, entry = text.split(f"\n## {card_id} (", 1)
    return head + f"\n## {card_id} (" + entry.replace("- [ ] none of these is relevant", "- [x] none of these is relevant", 1)


def row(card_id: str, candidate: str, relevant: str, judged: str = "2026-10-01", judge: str = "X", note: str = "") -> dict:
    return {"card": card_id, "candidate": candidate, "relevant": relevant, "judged": judged, "judge": judge, "note": note}


class HandMarksTest(unittest.TestCase):
    DECKS = {"Alpha": 2}  # the sample on the test course: 101001, 301003 and 401005

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        folder = Path(self._tmp.name)
        self.form = folder / "hand_check.md"
        self.file = folder / "hand_marks.csv"
        self.cards = model_support.CARDS
        self.by_id = {card.id: card for card in self.cards}
        self.indexes = {
            "a": similar.SimilarityIndex.fit(self.cards),
            "b": similar.SimilarityIndex.fit(self.cards, similar.tfidf_vectoriser(min_df=1), min_score=0.0, with_explain=False),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, marks, indexes=None) -> int:
        return evaluate.write_hand_check(self.cards, self.by_id, indexes or self.indexes, TODAY, marks, "test", self.form, self.DECKS)

    def candidates_of(self, card_id: str) -> list[str]:
        """The candidate ids the form lists under the card, in order."""
        entry = self.form.read_text(encoding="utf-8").split(f"\n## {card_id} (", 1)[1].split("\n## ")[0]
        return [match.group(2) for match in map(evaluate.CANDIDATE.match, entry.splitlines()) if match]

    def test_the_form_lists_the_sample_with_the_candidates_of_every_variant(self):
        self.assertEqual(self.write({}), 3)
        text = self.form.read_text(encoding="utf-8")
        self.assertEqual([line.split(" ")[1] for line in text.splitlines() if line.startswith("## ")], ["101001", "301003", "401005"])
        for card_id in ("101001", "301003", "401005"):
            pooled = {other for index in self.indexes.values() for other, _ in index.related(card_id)}
            listed = self.candidates_of(card_id)
            self.assertEqual(set(listed), pooled, card_id)
            self.assertEqual(len(listed), len(pooled))  # each once
        self.assertEqual(text.count("- [x]"), 0)
        self.assertEqual(text.count("- [ ] none of these is relevant"), 3)
        self.assertEqual(text.count(TAG), 0)  # no card is judged, so nothing is new
        self.assertEqual(evaluate.read_hand_check(self.form), {})  # nothing judged yet
        self.assertEqual(evaluate.read_hand_check(self.form.with_name("none.md")), {})

    def test_the_marks_of_the_form_are_read_with_every_listed_candidate(self):
        self.write({})
        first = self.candidates_of("101001")
        text = mark_none(mark_candidate(self.form.read_text(encoding="utf-8"), "101001", first[0]), "301003")
        self.form.write_text(text, encoding="utf-8")
        form = evaluate.read_hand_check(self.form)
        self.assertEqual(set(form), {"101001", "301003"})  # 401005 has no mark: not judged
        self.assertEqual(form["101001"], {"candidates": first, "relevant": {first[0]}})
        self.assertEqual(form["301003"], {"candidates": self.candidates_of("301003"), "relevant": set()})

    def test_the_marks_merge_over_the_stored_rows(self):
        stored = {"A": {"a1": row("A", "a1", "x"), "a2": row("A", "a2", "", note="why not"), "a9": row("A", "a9", "x")},
                  "B": {"b1": row("B", "b1", "x")}}
        form = {"A": {"candidates": ["a1", "a2", "a3"], "relevant": {"a2", "a3"}}, "C": {"candidates": ["c1", "c2"], "relevant": set()}}
        merged = evaluate.merge_marks(stored, form, TODAY, "Y")
        self.assertEqual(merged["A"], {
            "a1": row("A", "a1", "", TODAY, "Y"),  # unmarked now: a row of today
            "a2": row("A", "a2", "x", TODAY, "Y", "why not"),  # changed: a row of today, the note kept
            "a3": row("A", "a3", "x", TODAY, "Y"),  # new
            "a9": stored["A"]["a9"],  # not listed any more: kept
        })
        self.assertEqual(merged["B"], stored["B"])  # not in the form: kept
        self.assertEqual(merged["C"], {"c1": row("C", "c1", "", TODAY, "Y"), "c2": row("C", "c2", "", TODAY, "Y")})  # judged none
        self.assertEqual(stored["A"]["a1"]["relevant"], "x")  # the stored rows are not changed in place
        same = evaluate.merge_marks(merged, form, "2026-12-01", "Z")
        self.assertEqual(same, merged)  # the same judgements again change nothing

    def test_the_file_round_trip_in_id_order_and_the_relevant_sets(self):
        marks = {"B": {"b2": row("B", "b2", "x", note="a, note"), "b1": row("B", "b1", "")}, "A": {"a1": row("A", "a1", "x")}}
        evaluate.write_marks(marks, self.file)
        self.assertEqual([(line["card"], line["candidate"]) for line in read_csv(self.file)], [("A", "a1"), ("B", "b1"), ("B", "b2")])
        self.assertEqual(evaluate.read_marks(self.file), marks)
        self.assertEqual(evaluate.relevant_of(marks), {"A": {"a1"}, "B": {"b2"}})
        self.assertEqual(evaluate.read_marks(self.file.with_name("none.csv")), {})

    def test_a_rewrite_with_a_variant_removed_keeps_every_mark_and_a_new_candidate_is_tagged(self):
        self.write({})
        first = self.candidates_of("101001")
        only_b = [other for other, _ in self.indexes["b"].related("101001") if other not in dict(self.indexes["a"].related("101001"))]
        self.assertTrue(only_b)  # the second variant lists a candidate the first does not
        text = self.form.read_text(encoding="utf-8")
        text = mark_candidate(mark_candidate(text, "101001", first[0]), "101001", only_b[0])
        self.form.write_text(mark_none(text, "301003"), encoding="utf-8")
        marks = evaluate.merge_marks({}, evaluate.read_hand_check(self.form), TODAY, "Y")
        self.assertEqual(evaluate.relevant_of(marks), {"101001": {first[0], only_b[0]}, "301003": set()})
        self.assertEqual(set(marks["101001"]), set(first))  # every listed candidate has a row
        # Written again with the second variant removed: the marks stay, the form shows what is left
        self.write(marks, {"a": self.indexes["a"]})
        text = self.form.read_text(encoding="utf-8")
        self.assertNotIn(only_b[0], self.candidates_of("101001"))
        self.assertIn(f"- [x] {first[0]} (", text)
        self.assertEqual(text.count(TAG), 0)  # nothing new
        self.assertEqual(text.split("\n## 301003 (", 1)[1].count("- [x] none of these is relevant"), 1)
        self.assertEqual(evaluate.merge_marks(marks, evaluate.read_hand_check(self.form), "2026-12-01", "Z"), marks)
        # A new variant lists a candidate under the judged card: tagged until it is judged; the unjudged card gets no tag
        self.write(marks, {"a": self.indexes["a"], "new": FakeIndex("101001", "103001")})
        text = self.form.read_text(encoding="utf-8")
        self.assertIn(f"- [ ] 103001 (Gamma): {evaluate.short(self.by_id['103001'].question)}{TAG}new: 0.50", text)
        self.assertEqual(text.count(TAG), 1)
        self.write(marks, {"a": self.indexes["a"], "new": FakeIndex("401005", "103001")})
        self.assertEqual(self.form.read_text(encoding="utf-8").count(TAG), 0)


# --- the similarity part and the run ---------------------------------------------------------------------------


def without_encoder():
    """The library as not installed, so the S1-only expectations hold wherever the tests run."""
    return mock.patch.object(evaluate.embeddings, "encoder_available", return_value=False)


def without_tagger():
    """transformers as not installed, so the difficulty part runs without D2 wherever the tests run."""
    return mock.patch.object(evaluate.zero_shot, "tagger_available", return_value=False)


class SimilarityPartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marks = {"101001": {"201002", "101004"}, "301003": set()}
        with without_encoder(), contextlib.redirect_stdout(io.StringIO()) as output:
            cls.rows, cls.indexes = evaluate.evaluate_similarity(
                model_support.CARDS, model_support.SEEDS, model_support.DECK_IDS, TODAY, "test", cls.marks)
        cls.output = output.getvalue()

    def rows_of(self, model: str) -> list[dict]:
        return [row for row in self.rows if row["model"] == model]

    def test_the_rows_of_b0_s1_and_c1(self):
        self.assertEqual([row["model"] for row in self.rows], ["B0"] * 5 + ["S1"] * 7 + ["C1"] * 3)
        self.assertTrue(all(key in evaluate.COLUMNS for row in self.rows for key in row))
        self.assertEqual([row["seed"] for row in self.rows_of("B0")], list(evaluate.SEEDS))
        self.assertTrue(all(0.0 <= float(row["precision_at_5"]) <= 1.0 and row["params"] == "uniform on the sphere" for row in self.rows_of("B0")))
        names = [variant.name for variant in evaluate.variants()]
        self.assertEqual([row["variant"] for row in self.rows_of("S1")], names)
        self.assertEqual(list(self.indexes), names)
        self.assertEqual([row["variant"] for row in self.rows_of("C1")], ["model", "SVD start", "random"])

    def test_every_variant_has_its_own_minimum_score_and_the_hand_columns(self):
        for row in self.rows_of("S1"):
            self.assertTrue(0.0 <= float(row["min_score"]) <= 1.0, row["variant"])
            self.assertIn(f"min_score {row['min_score']}", row["notes"])
            self.assertIn("vocab=", row["params"])
            self.assertEqual(row["judged_cards"], 2)
            for name in ("hand_precision", "hand_p1", "hand_p3"):
                self.assertTrue(0.0 <= float(row[name]) <= 1.0, name)
            self.assertLessEqual(float(row["hand_diff_low"]), float(row["hand_diff"]))
            self.assertLessEqual(float(row["hand_diff"]), float(row["hand_diff_high"]))
        chosen = next(row for row in self.rows_of("S1") if row["variant"] == evaluate.CHOSEN)
        self.assertEqual((chosen["hand_diff"], chosen["hand_diff_low"], chosen["hand_diff_high"]), ("0.000", "0.000", "0.000"))
        self.assertEqual(float(chosen["hand_precision"]), evaluate.hand_precision(self.indexes[evaluate.CHOSEN].neighbours, self.marks))

    def test_the_map_rows_carry_the_fixed_k_columns(self):
        model, start, random = self.rows_of("C1")
        for row in (model, start, random):
            self.assertTrue(0.0 <= float(row["overlap_cards"]) <= 1.0)
            self.assertIn("fixed k over the topics with k+2 cards or more", row["params"])
        # The test course's topics have three cards at most, below k + 2: the fixed-k columns stay empty
        self.assertEqual((model["overlap_k2"], model["lift_k2"], model["overlap_k3"], model["lift_k3"]), ("", "", "", ""))
        self.assertIn("0 at k = 2, 0 at k = 3", model["notes"])
        self.assertEqual(random["seed"], 42)
        self.assertNotEqual(model["seconds"], "")

    def test_without_marks_the_hand_columns_stay_empty(self):
        with without_encoder(), contextlib.redirect_stdout(io.StringIO()):
            rows, _ = evaluate.evaluate_similarity(model_support.CARDS, model_support.SEEDS, model_support.DECK_IDS, TODAY, "test", {})
        for row in rows:
            if row["model"] == "S1":
                self.assertEqual((row["hand_precision"], row["hand_p1"], row["hand_p3"], row["judged_cards"]), ("", "", "", ""))
                self.assertNotIn("hand_diff", row)

    def test_without_the_library_the_s2_rows_are_skipped_with_one_line_and_the_deck_report_prints(self):
        self.assertEqual(self.output.count(evaluate.S2_SKIPPED), 1)
        self.assertNotIn("S2 ", [row["model"] + " " for row in self.rows])
        self.assertIn("hand columns per deck", self.output)
        self.assertIn("deck 01 (2 judged cards)", self.output)  # the two marked cards are Alpha's


class EncoderVariantsTest(unittest.TestCase):
    """S2's rows with the fake encoder standing in for the models (model_support.fake_loader)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cache = Path(cls._tmp.name) / "cache"
        marks = {"101001": {"201002", "101004"}, "301003": set()}
        tagged = {"lexical trap": {("101001", "103001")}, "paraphrase": {("101001", "201002")}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cls.rows, cls.indexes = evaluate.evaluate_similarity(
                model_support.CARDS, model_support.SEEDS, model_support.DECK_IDS, TODAY, "test", marks, cls.cache,
                tagged=tagged, loader=model_support.fake_loader)
        cls.output = output.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_s2_rows_follow_s1s_and_name_the_encoders_and_the_hybrids(self):
        self.assertEqual([row["model"] for row in self.rows], ["B0"] * 5 + ["S1"] * 7 + ["S2"] * 4 + ["C1"] * 3)
        s2 = [row for row in self.rows if row["model"] == "S2"]
        self.assertEqual([row["variant"] for row in s2], ["MiniLM", "hybrid: chosen + MiniLM", "mpnet", "hybrid: chosen + mpnet"])
        self.assertIn(f"model={evaluate.embeddings.MINILM}", s2[0]["params"])
        self.assertNotIn("vocab=", s2[0]["params"])  # an encoder has no vocabulary
        for row in (s2[1], s2[3]):
            self.assertIn("vocab=", row["params"])
            self.assertIn("minima=(", row["params"])
            self.assertIn("weights=(", row["params"])
        for row in s2:
            self.assertTrue(0.0 <= float(row["min_score"]) <= 1.0)
            self.assertEqual(row["judged_cards"], 2)
            self.assertIn(row["variant"], self.indexes)
            self.assertIn("device=", row["params"])  # the device the seconds were measured on, or the cache
        self.assertNotIn(evaluate.S2_SKIPPED, self.output)

    def test_the_cache_holds_one_file_per_model_and_the_hybrids_read_it(self):
        files = sorted(path.name for path in (self.cache / evaluate.embeddings.CACHE_FOLDER).glob("*.npy"))
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].startswith("all-minilm-l6-v2-"))
        self.assertTrue(files[1].startswith("all-mpnet-base-v2-"))
        hybrid = self.indexes["hybrid: chosen + MiniLM"].vectoriser
        self.assertTrue(hybrid.second.cached)  # the encoder's row encoded, the hybrid read the cache

    def test_the_tag_report_counts_the_tagged_pairs_each_variant_lists(self):
        self.assertIn("tagged pairs listed in the top 5 (lexical trap: 1, paraphrase: 1)", self.output)
        self.assertIn("| Variant | lexical trap | paraphrase |", self.output)
        counts = evaluate.tag_report(self.indexes, {"paraphrase": {("101001", "201002")}, "lexical trap": set()})
        for name, index in self.indexes.items():
            listed = "201002" in [other for other, _ in index.related("101001")[:5]]
            self.assertEqual(counts[name], {"paraphrase": int(listed)}, name)
        self.assertEqual(evaluate.tag_report(self.indexes, {}), {})


class TagsTest(unittest.TestCase):
    def test_the_tags_are_read_from_the_notes_as_whole_words_in_any_case(self):
        marks = {
            "a": {"b": row("a", "b", "", note="Lexical trap: shares 'rate'"), "c": row("a", "c", "x", note="paraphrase; code name kept")},
            "d": {"e": row("d", "e", "", note="a generic terminology issue"), "f": row("d", "f", "x", note="")},
        }
        tagged = evaluate.tags_of(marks)
        self.assertEqual(tagged["lexical trap"], {("a", "b")})
        self.assertEqual(tagged["paraphrase"], {("a", "c")})
        self.assertEqual(tagged["code name"], {("a", "c")})
        self.assertEqual(tagged["generic term"], set())  # "generic terminology" is not the tag
        self.assertEqual(tagged["shared dataset"], set())
        self.assertEqual(set(tagged), set(evaluate.TAGS))

    def test_the_sample_is_every_card_of_the_three_decks_and_the_form_names_them(self):
        self.assertEqual(evaluate.HAND_CHECK_DECKS, {"NLP Fundamentals": 1, "LLM Foundations & Embeddings": 1, "AI Agents & RAG Systems": 1})
        self.assertEqual(evaluate.TRANSFER_DECK, "AI Agents & RAG Systems")

    def test_the_default_folders_are_in_the_launchers_data_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            module_dir = Path(folder) / "root" / "package"
            module_dir.mkdir(parents=True)
            (module_dir.parent / "data").mkdir()
            with mock.patch.object(evaluate, "MODULE_DIR", module_dir):
                # data/ in the folder above the package's, where there is none next to it
                self.assertEqual(evaluate.default_cache_dir(), module_dir.parent / "data" / "cache")
                self.assertEqual(evaluate.default_table_dir(), module_dir.parent / "data" / "course")
                (module_dir / "data").mkdir()  # data/ next to the package (the capstone) comes first
                self.assertEqual(evaluate.default_cache_dir(), module_dir / "data" / "cache")
                self.assertEqual(evaluate.default_table_dir(), module_dir / "data" / "course")


REQUESTS = (
    "request,kind,deck_id,topic_id,card,also,judged,judge,note\n"
    "null hypothesis,concept,03,02,103003,,2026-11-01,Y,the definition\n"
    "the history of a repository,paraphrase,02,01,,,2026-11-01,Y,\"Commits' heading, no one card\"\n"
    "how to bake bread,off-course,,,,,2026-11-01,Y,no deck teaches it\n"
    "python dictionary keys,concept,01,02,301003,401005,2026-11-01,Y,\n"
    "a request without a target,concept,,,,,,,\n"
    "a transfer request,concept,18,01,,,2026-11-01,Y,read once after the choice\n"
)


class PromptPartTest(unittest.TestCase):
    """P1's part: the rows of every variant, the request file read against T1's membership, the review form and
    its marks, the file alone in the capstone, no hand side without the file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.table = model_support.write_test_table(self.folder / "course")
        self.results = self.folder / "results"
        self.files = evaluate.result_files(self.results)
        self._without = without_encoder()
        self._without.start()

    def tearDown(self):
        self._without.stop()
        self._tmp.cleanup()

    def run_prompt(self, judge: str = "") -> str:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            evaluate.run(self.table, self.results, "prompt", judge, TODAY)
        return output.getvalue()

    def test_the_rows_of_every_variant_and_without_the_request_file_the_self_set_alone(self):
        output = self.run_prompt()
        with self.files.prompt.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(next(csv.reader(handle)), evaluate.PROMPT_COLUMNS)
        rows = read_csv(self.files.prompt)
        self.assertEqual([row["variant"] for row in rows],
                         [f"TF-IDF: {name}" for name in ROUTES] + [f"{evaluate.CHAR_NGRAMS}: card"])  # S1's, no encoder here
        self.assertTrue(all(row["model"] == "P1" and row["run"] for row in rows))
        for row in rows:
            for column in ("self_topic_at1", "self_topic_at3", "self_navigated", "min_card", "min_centroid",
                           "min_text", "ms_per_request", "seconds"):
                self.assertNotEqual(row[column], "", column)
            self.assertLessEqual(float(row["self_topic_at1"]), float(row["self_topic_at3"]))
            hand = [column for column in evaluate.PROMPT_COLUMNS if column.startswith("hand") or column == "judged_requests"]
            self.assertTrue(all(row[column] == "" for column in hand))
        self.assertEqual(len({(row["min_card"], row["min_centroid"]) for row in rows[:5]}), 1)  # one model per vectoriser
        self.assertNotIn("card_bar", evaluate.PROMPT_COLUMNS)  # no second threshold for the card
        self.assertIn("routes=card+text, gate=card", rows[3]["params"])
        self.assertIn("prompt_requests.md is not written", output)
        self.assertFalse(self.files.prompt_form.exists())  # no form without the file
        self.assertFalse(self.files.prompt_requests.exists())

    def test_the_request_file_is_protected_and_the_form_written(self):
        self.results.mkdir()
        self.files.prompt_requests.write_text(REQUESTS, encoding="utf-8")
        self.files.prompt_form.write_text("", encoding="utf-8")  # the hand side is there: an empty form
        self.run_prompt("Z")
        self.assertEqual(self.files.prompt_requests.read_text(encoding="utf-8"), REQUESTS)  # nothing marked: not written
        form = self.files.prompt_form.read_text(encoding="utf-8")
        for number, text in enumerate(["null hypothesis", "the history of a repository", "how to bake bread"], 1):
            self.assertIn(f"\n## {number}. {text}\n", form)
        self.assertIn("- [x] 103003 card (03:02): What is the null hypothesis?  —  intended  —  TF-IDF: card ", form)
        self.assertIn("- [x] 03:02 topic: Tests  —  implied", form)
        self.assertIn("- [x] 401005 card (01:02): ", form)
        rows = read_csv(self.files.prompt)
        self.assertTrue(all(row["judged_requests"] == "4" for row in rows))  # neither the untargeted nor the transfer request

        # A further acceptable target marked in the form goes into `also`, with the day and the judge
        entry = form.split("\n## 2. the history of a repository\n")[1].split("\n## ")[0]
        candidate = next(match.group(2) for match in map(evaluate.PROMPT_CANDIDATE.match, entry.splitlines())
                         if match and match.group(1) == " ")
        self.files.prompt_form.write_text(form.replace(f"- [ ] {candidate} ", f"- [x] {candidate} ", 1), encoding="utf-8")
        output = self.run_prompt("Z")
        self.assertIn("1 changed by the form", output)
        stored = read_csv(self.files.prompt_requests)
        self.assertEqual(len(stored), 6)
        self.assertEqual((stored[1]["also"], stored[1]["judged"], stored[1]["judge"]), (candidate, TODAY, "Z"))
        self.assertEqual(stored[1]["note"], "Commits' heading, no one card")  # the rest of the row kept
        self.assertEqual(stored[3]["also"], "401005")
        self.assertIn(f"- [x] {candidate} ", self.files.prompt_form.read_text(encoding="utf-8"))

    def test_with_the_file_alone_the_hand_columns_are_computed_and_nothing_is_written_back(self):
        """The capstone: the requests without their notes, and no form."""
        self.results.mkdir()
        rows = list(csv.DictReader(io.StringIO(REQUESTS)))
        columns = [column for column in evaluate.REQUEST_COLUMNS if column != "note"]
        evaluate.write_requests(columns, rows, self.files.prompt_requests)  # as the capstone holds it, without the notes
        without_notes = self.files.prompt_requests.read_text(encoding="utf-8")
        self.assertNotIn("no deck teaches it", without_notes)
        output = self.run_prompt()
        self.assertEqual(self.files.prompt_requests.read_text(encoding="utf-8"), without_notes)
        self.assertFalse(self.files.prompt_form.exists())
        self.assertIn("nothing written back", output)
        rows = read_csv(self.files.prompt)
        self.assertTrue(all(row["judged_requests"] == "4" and row["hand_right"] != "" for row in rows))

    def test_the_paired_difference_is_against_the_vectorisers_card_route(self):
        self.results.mkdir()
        self.files.prompt_requests.write_text(REQUESTS, encoding="utf-8")
        output = self.run_prompt()
        rows = {row["variant"]: row for row in read_csv(self.files.prompt)}
        reference = rows["TF-IDF: card"]
        self.assertEqual((reference["hand_diff"], reference["hand_diff_low"], reference["hand_diff_high"]), ("0.000",) * 3)
        for label, row in rows.items():  # paired over the same requests: the difference of the shares right
            self.assertAlmostEqual(float(row["hand_diff"]), float(row["hand_right"]) - float(reference["hand_right"]), places=2)
            self.assertLessEqual(float(row["hand_diff_low"]), float(row["hand_diff"]))
            self.assertGreaterEqual(float(row["hand_diff_high"]), float(row["hand_diff"]))
            shares = [float(row[column]) for column in ("hand_right", "hand_wrong", "hand_missed")]
            self.assertAlmostEqual(sum(shares), 1.0, places=2)
            # the same outcomes counted on and off the course: three in-course requests, one off-course, never missed
            course = [int(row[f"hand_course_{outcome}"]) for outcome in ("right", "wrong", "missed")]
            off = [int(row[f"hand_off_{outcome}"]) for outcome in ("right", "wrong")]
            self.assertEqual((sum(course), sum(off)), (3, 1))
            self.assertAlmostEqual((course[0] + off[0]) / 4, shares[0], places=2)
            self.assertAlmostEqual((course[1] + off[1]) / 4, shares[1], places=2)
            self.assertAlmostEqual(course[2] / 4, shares[2], places=2)
        char_line = next(line for line in output.splitlines() if line.strip().startswith(f"{evaluate.CHAR_NGRAMS}: card"))
        self.assertIn("against TF-IDF: card", char_line)  # the character n-grams against S1's card route
        self.assertIn("hand_right by kind", output)

    def test_the_transfer_decks_numbers_are_printed_apart_and_never_written(self):
        self.results.mkdir()
        self.files.prompt_requests.write_text(REQUESTS, encoding="utf-8")
        with mock.patch.object(evaluate, "PROMPT_TRANSFER_DECKS", ("03",)):  # the null hypothesis request, with its card
            output = self.run_prompt()
        rows = read_csv(self.files.prompt)
        self.assertTrue(all(row["judged_requests"] == "3" for row in rows))  # the transfer request out of the hand columns
        self.assertFalse(any(column.startswith("transfer") for column in evaluate.PROMPT_COLUMNS))  # printed, never written
        self.assertIn("1 of the transfer deck printed apart", output)
        lines = [line.strip() for line in output.splitlines()]
        start = next(number for number, line in enumerate(lines) if line.startswith("the transfer deck, 1 judged request of deck 03"))
        self.assertIn("| Variant | right | wrong | missed | card@1 (1) |", lines[start])
        table = lines[start + 1:start + 1 + len(rows)]
        self.assertEqual([line.strip("|").split("|")[0].strip() for line in table], [row["variant"] for row in rows])
        for line in table:
            cells = [cell.strip() for cell in line.strip("|").split("|")][1:]
            right, wrong, missed = (float(cell) for cell in cells[:3])
            self.assertAlmostEqual(right + wrong + missed, 1.0, places=2)
            self.assertIn(cells[3], ("0.00", "1.00"))  # one request with a card: its card first or not


class RequestTest(unittest.TestCase):
    """A request read against the course and T1's membership, its outcome, and the form's marks merged."""

    BY_ID = {card.id: card for card in model_support.CARDS}
    DECK_NAME = {deck_id: name for name, deck_id in model_support.DECK_IDS.items()}
    TOPIC_OF = {**{card_id: key[1] for key, ids in model_support.SCRIPTED.items() for card_id in ids}, "101004": "02"}

    def request(self, **row):
        return evaluate.request_of({"request": "r", "kind": "concept", **row}, self.BY_ID, self.DECK_NAME, self.TOPIC_OF)

    def test_the_card_names_its_topic_under_the_membership_in_force(self):
        request = self.request(deck_id="01", topic_id="01", card="101004", also="02:02 103001")
        self.assertEqual(request.topics, {("Alpha", "02"), ("Beta", "02"), ("Gamma", "01")})  # not the row's 01:01
        self.assertEqual((request.card, request.cards), ("101004", {"101004", "103001"}))
        self.assertTrue(request.judged)
        topic_only = self.request(deck_id="02", topic_id="01")
        self.assertEqual((topic_only.topics, topic_only.card), ({("Beta", "01")}, ""))
        self.assertFalse(self.request().judged)  # no target, not off-course
        self.assertFalse(self.request(card="999999").judged)  # a card not in the course
        off = self.request(kind=evaluate.OFF_COURSE)
        self.assertTrue(off.judged and off.off_course and not off.topics)
        self.assertTrue(self.request(deck_id="18", topic_id="01").transfer)

    def test_the_outcome_of_an_answer(self):
        def answer(decision, key=("Alpha", "02")):
            return Answer("r", ("card",), decision, (TopicScore(key, 0.0, {}, None),))

        request = self.request(card="101004")
        self.assertEqual(evaluate.request_outcome(request, answer(ENTERED)), (True, False, False))
        self.assertEqual(evaluate.request_outcome(request, answer(SELECTED, ("Alpha", "01"))), (False, True, False))
        self.assertEqual(evaluate.request_outcome(request, answer(NO_MATCH)), (False, False, True))
        off = self.request(kind=evaluate.OFF_COURSE)
        self.assertEqual(evaluate.request_outcome(off, answer(NO_MATCH)), (True, False, False))
        self.assertEqual(evaluate.request_outcome(off, answer(ENTERED)), (False, True, False))

    def test_the_forms_marks_go_into_also_and_the_intended_and_implied_never(self):
        rows = [
            {"request": "a", "kind": "concept", "deck_id": "01", "topic_id": "02", "card": "301003", "also": "401005 02:01",
             "judged": "d", "judge": "X"},
            {"request": "b", "kind": evaluate.OFF_COURSE, "also": "", "judged": "d", "judge": "X"},
            {"request": "c", "kind": "concept", "deck_id": "03", "topic_id": "01", "card": "", "also": "", "judged": "d", "judge": "X"},
        ]
        form_text = "\n".join([
            "## 1. a", "", "- [x] 301003 card", "- [ ] 01:02 topic", "- [x] 101001 card", "- [ ] 02:01 topic", "- [x] 401005 card",
            "## 2. b", "", "- [x] 01:01 topic",
            "## 3. c", "", "- [x] 03:01 topic", "- [ ] 103002 card",
        ])
        path = Path(tempfile.mkdtemp()) / "prompt_requests.md"
        path.write_text(form_text, encoding="utf-8")
        form = evaluate.read_prompt_form(path)
        self.assertEqual(form["a"]["candidates"], ["301003", "01:02", "101001", "02:01", "401005"])
        self.assertEqual(form["a"]["marked"], {"301003", "101001", "401005"})
        changed = evaluate.merge_requests(rows, form, TODAY, "Z", self.BY_ID, model_support.DECK_IDS, self.TOPIC_OF)
        self.assertEqual(changed, 1)
        self.assertEqual((rows[0]["also"], rows[0]["judged"], rows[0]["judge"]), ("401005 101001", TODAY, "Z"))  # 02:01 unticked
        self.assertEqual(rows[1]["also"], "")  # an off-course request's marks are not read
        self.assertEqual((rows[2]["also"], rows[2]["judged"]), ("", "d"))  # the intended topic is never written


PROBE = """
import sys
sys.path.insert(0, {tests!r})
import model_support
from pathlib import Path
import evaluate
evaluate.embeddings.encoder_available = lambda *arguments, **options: False  # the S1 rows alone, as without torch
evaluate.zero_shot.tagger_available = lambda *arguments, **options: False
folder = Path({folder!r})
evaluate.run(model_support.write_test_table(folder / "course"), folder / "results", "similarity", "", "2026-11-01")
print("ok")
"""


class RunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.table = model_support.write_test_table(self.folder / "course")
        self.results = self.folder / "results"
        self.files = evaluate.result_files(self.results)
        self._without = without_encoder()
        self._without.start()
        self._without_tagger = without_tagger()
        self._without_tagger.start()

    def tearDown(self):
        self._without_tagger.stop()
        self._without.stop()
        self._tmp.cleanup()

    def test_the_demos_deck_00_is_never_read(self):
        self.assertEqual(evaluate.default_table_dir().name, "course")
        self.assertEqual(evaluate.default_table_dir(), model_support.COURSE_TABLE_DIR)
        model_support.write_demo_table(self.folder / "demo" / "dashboard")  # next to the course table, as in data/
        opened = []
        real_open = io.open

        def recording_open(file, *arguments, **options):
            opened.append(str(file))
            return real_open(file, *arguments, **options)

        with mock.patch("io.open", recording_open), mock.patch("builtins.open", recording_open), contextlib.redirect_stdout(io.StringIO()):
            evaluate.run(self.table, self.results, "topics", today=TODAY)
        self.assertTrue(any(name.endswith("cards.csv") for name in opened))
        self.assertFalse([name for name in opened if "demo" in Path(name).parts])

    def test_without_the_hand_files_the_derived_rows_are_written_and_no_form(self):
        evaluate.run(self.table, self.results, "similarity", today=TODAY)
        rows = read_csv(self.files.results)
        self.assertEqual(list(rows[0]), evaluate.COLUMNS)
        self.assertEqual(len(rows), 15)
        hand_columns = [column for column in evaluate.COLUMNS if column.startswith("hand") or column == "judged_cards"]
        self.assertTrue(all(row[column] == "" for row in rows for column in hand_columns))
        self.assertEqual(len({row["run"] for row in rows}), 1)  # every row of the run carries its key
        self.assertRegex(rows[0]["run"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")  # the start time to the minute
        self.assertFalse(self.files.hand_check.exists())
        self.assertFalse(self.files.hand_marks.exists())
        evaluate.run(self.table, self.results, "difficulty", today=TODAY, run_key="2026-11-02T09:00")
        self.assertTrue(all(row["hand_judged"] == "" and row["run"] == "2026-11-02T09:00" for row in read_csv(self.files.difficulty)))
        self.assertFalse(self.files.hand_labels_form.exists())
        self.assertFalse(self.files.hand_labels.exists())
        with self.files.difficulty.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(next(csv.reader(handle)), evaluate.DIFFICULTY_COLUMNS)  # the header, with the hand_diff columns
        self.assertEqual(read_csv(self.files.difficulty), [])  # one rated card on this course: nothing measured, no row
        with contextlib.redirect_stdout(io.StringIO()) as output:
            evaluate.run(self.table, self.results, "difficulty", today=TODAY, cache_dir=self.folder / "cache",
                         tagger_loader=model_support.fake_classifier_loader)
        self.assertIn("D2 criterion, base: 2 quizzes tagged by", output.getvalue())  # the tagger ran on the two quizzes
        self.assertIn("D2 level names, large: 2 quizzes tagged by", output.getvalue())
        self.assertIn("D2 criterion, base: Too few rated quizzes", output.getvalue())  # and D2's judge has nothing to measure
        self.assertEqual(len(list((self.folder / "cache" / evaluate.embeddings.CACHE_FOLDER).glob("*.npy"))), 4)
        evaluate.run(self.table, self.results, "topics", today=TODAY)
        topics = read_csv(self.files.topics)
        self.assertEqual(topics[0]["model"], "T1")
        self.assertEqual({row["scope"] for row in topics if row["scope"].startswith("deck ")}, {"deck 01", "deck 02", "deck 03"})  # by id
        focus = [row for row in topics if row["scope"].startswith("focus ")]  # the share scale's marks
        self.assertEqual([row["scope"] for row in focus], ["focus 100 %", "focus 75 %", "focus 50 %", "focus 25 %", "focus 1 %"])
        self.assertEqual((focus[0]["margin"], focus[0]["kept_share"], focus[0]["cards"]), ("0.000", "1.000", "13"))
        self.assertEqual(focus[-1]["cards"], "0")  # 1 % of 13 cards rounds to none
        self.assertGreater(float(focus[1]["margin"]), 0.0)
        self.assertEqual(next(row for row in topics if row["scope"] == "margin >= 0.05")["margin"], "0.050")
        self.assertTrue(all(name not in row["scope"] for row in topics for name in model_support.DECK_IDS))
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), ["difficulty.csv", "results.csv", "topics.csv"])

    def test_with_the_marks_alone_the_hand_columns_are_computed_and_nothing_is_written_back(self):
        """The capstone: the marks and the labels without their notes, and no forms."""
        self.results.mkdir()
        marks = "card,candidate,relevant,judged,judge\n101001,201002,x,2026-10-30,Y\n101001,101004,,2026-10-30,Y\n"
        labels = "card,level,judged,judge\n201002,medium,2026-10-31,Y\n"
        self.files.hand_marks.write_text(marks, encoding="utf-8")
        self.files.hand_labels.write_text(labels, encoding="utf-8")
        evaluate.run(self.table, self.results, "all", "Z", TODAY)
        rows = [row for row in read_csv(self.files.results) if row["model"] == "S1"]
        self.assertEqual([row["judged_cards"] for row in rows], ["1"] * 7)
        self.assertTrue(all(row["hand_precision"] != "" and row["hand_diff_low"] != "" for row in rows))
        self.assertEqual(self.files.hand_marks.read_text(encoding="utf-8"), marks)  # never written back
        self.assertEqual(self.files.hand_labels.read_text(encoding="utf-8"), labels)
        self.assertFalse(self.files.hand_check.exists())  # no form
        self.assertFalse(self.files.hand_labels_form.exists())

    def test_with_the_forms_the_marks_and_the_labels_are_kept_in_their_files(self):
        self.results.mkdir()
        self.files.hand_check.write_text("", encoding="utf-8")  # the hand side is there: an empty form
        self.files.hand_labels_form.write_text("", encoding="utf-8")
        with mock.patch.object(evaluate, "HAND_CHECK_DECKS", {"Alpha": 2}):
            evaluate.run(self.table, self.results, "all", "Y", TODAY)
            self.assertEqual(read_csv(self.files.hand_marks), [])  # written, nothing judged
            self.assertEqual(read_csv(self.files.hand_labels), [])
            text = self.files.hand_check.read_text(encoding="utf-8")
            self.assertIn("\n## 101001 (Alpha): ", text)
            candidate = next(match.group(2) for match in map(evaluate.CANDIDATE.match, text.split("\n## 101001 (")[1].splitlines()) if match)
            self.files.hand_check.write_text(mark_candidate(text, "101001", candidate), encoding="utf-8")
            form = self.files.hand_labels_form.read_text(encoding="utf-8")
            self.assertIn("- (correct) list[start:stop]  —  ", form)  # the table's options
            self.files.hand_labels_form.write_text(mark(form, "201002", "medium"), encoding="utf-8")
            evaluate.run(self.table, self.results, "all", "Y", TODAY)
        marks = read_csv(self.files.hand_marks)
        self.assertEqual({(line["card"], line["judged"], line["judge"]) for line in marks}, {("101001", TODAY, "Y")})
        self.assertEqual([line["candidate"] for line in marks if line["relevant"]], [candidate])
        self.assertGreater(len(marks), 1)  # the other listed candidates: judged, not relevant
        self.assertEqual(read_csv(self.files.hand_labels), [{"card": "201002", "level": "medium", "judged": TODAY, "judge": "Y", "note": ""}])
        rows = [row for row in read_csv(self.files.results) if row["model"] == "S1"]
        self.assertEqual([row["judged_cards"] for row in rows], [""] * 7 + ["1"] * 7)  # the first run had no marks
        self.assertEqual(rows[-1]["hand_diff"], "0.000")
        self.assertIn(f"- [x] {candidate} (", self.files.hand_check.read_text(encoding="utf-8"))

    def test_the_evaluation_runs_in_a_process_of_its_own(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", PROBE.format(tests=str(TESTS_DIR), folder=str(self.folder / "probe"))],
            capture_output=True, text=True, cwd=str(model_support.MODULE_DIR), timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok", result.stdout)
        self.assertIn("hand_check.md is not written", result.stdout)
        self.assertEqual(len(read_csv(self.folder / "probe" / "results" / "results.csv")), 15)


if __name__ == "__main__":
    unittest.main()
