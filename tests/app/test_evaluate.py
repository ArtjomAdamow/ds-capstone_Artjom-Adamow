"""The offline evaluation (evaluate.py, next to the package) and its measures (app.models.evaluation), on the
fixtures and nothing of the viewer: a results file whose columns grew; the hand labels of D1 (the labels file, the
judging form, the merge of its marks) and the hand columns of D1's rows and the `hand` row; the hand marks of the
similarity part (the marks file, the judging form, the merge, a rewrite with a variant removed); the measures (the
random draw, the hand precision per card, the paired bootstrap, the fixed-k overlap and the lift on a hand-made
layout); the rows of the similarity part; and a run on the test course's table, with and without the hand files,
in a process where the viewer cannot be imported."""

import model_support  # noqa: F401  (the helper's package on the path, and the folder next to it with evaluate.py)

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

    def test_the_hand_columns_of_d1s_rows_and_the_hand_row(self):
        # Every deck's short unrated quiz easy, its long one hard; Alpha's rated quizzes at their authored levels
        labels = {card.id: {"card": card.id, "level": "hard" if card.id.endswith("102") else "easy", "judged": TODAY, "judge": "Y", "note": ""}
                  for card in self.unrated}
        for card in self.cards:
            if card.deck == "Alpha" and card.difficulty in LEVELS:
                labels[card.id] = {"card": card.id, "level": card.difficulty, "judged": TODAY, "judge": "Y", "note": ""}
        labels["302001"] = {"card": "302001", "level": "unsure", "judged": "", "judge": "", "note": "by hand"}  # never scored
        rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test", labels)
        d1 = [row for row in rows if row["model"] == "D1"]
        self.assertEqual(len(d1), 9)  # three feature sets, three label rules
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
    """D2's rows and the feature set text+tags with the fake pipeline standing in for the model."""

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

    def test_the_tags_come_per_template_and_are_cached(self):
        self.assertEqual(list(self.tags), ["criterion", "level names"])
        self.assertEqual(len(self.tags["criterion"].scores), 42)
        self.assertEqual(len(list((self.cache / evaluate.embeddings.CACHE_FOLDER).glob("*.npy"))), 2)
        self.assertIn("D2 criterion: 42 quizzes tagged by", self.output)

    def test_the_feature_set_with_the_tags_joins_d1s_rows(self):
        rows = self.rows_of("D1", "text+tags")
        self.assertEqual([row["label_rule"] for row in rows], ["deck shares", "most likely class", "cut points of the rated cards"])
        self.assertEqual((rows[0]["features"], rows[0]["folds"], rows[0]["seed"]), ("text+tags", 3, 42))
        self.assertEqual(rows[0]["hand_judged"], 6)
        self.assertTrue(rows[0]["hand_diff_low"] <= rows[0]["hand_diff"] <= rows[0]["hand_diff_high"])
        self.assertEqual(len([row for row in self.rows if row["model"] == "D1"]), 12)  # four feature sets, three label rules

    def test_d2_alone_gets_two_rows_per_template_without_folds(self):
        for name in ("criterion", "level names"):
            rows = self.rows_of("D2", name)
            self.assertEqual([row["label_rule"] for row in rows], ["deck shares", "most likely class"])
            shares = rows[0]
            self.assertEqual((shares["features"], shares["folds"], shares["repeats"], shares["seed"]), ("premise", 0, 0, ""))
            self.assertEqual((shares["rated"], shares["unrated"]), (36, 6))
            self.assertEqual(shares["macro_f1_sd"], "0.000")  # one pass: no spread
            self.assertEqual(shares["predicted_easy"] + shares["predicted_medium"] + shares["predicted_hard"], 6)
            self.assertEqual(shares["hand_judged"], 6)
            for column in ("hand_macro_f1", "hand_within_deck", "hand_diff", "hand_diff_low", "hand_diff_high"):
                self.assertIn(column, shares, column)
            self.assertEqual(shares["hand_within_pairs"], 3)
            self.assertNotIn("hand_judged", rows[1])
        self.assertIn("D2 criterion", self.output)
        self.assertIn("paired difference to text", self.output)
        self.assertIn("within a deck, per deck: Alpha", self.output)

    def test_without_the_tags_both_are_skipped_with_a_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rows = evaluate.evaluate_difficulty(self.cards, TODAY, "test", self.labels)
        self.assertFalse(any(row["model"] == "D2" for row in rows))
        self.assertFalse(any(row["variant"] == "text+tags" for row in rows))
        self.assertIn("D1 text+tags: skipped without D2's tags", output.getvalue())
        with without_tagger(), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(evaluate.difficulty_tags(self.cards, self.cache), {})
        self.assertIn(evaluate.D2_SKIPPED, output.getvalue())
        self.assertIn("transformers", output.getvalue())


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

    def test_the_default_cache_folder_is_data_cache_next_to_the_package_or_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(evaluate, "CACHE_DIRS", (root / "data" / "cache", root / "temp")):
                self.assertEqual(evaluate.default_cache_dir(), root / "temp")  # no data/: the module's temp/
                (root / "data").mkdir()
                self.assertEqual(evaluate.default_cache_dir(), root / "data" / "cache")


PROBE = """
import sys
sys.modules["project"] = None  # `import project` raises ImportError from here on
sys.path.insert(0, {tests!r})
import model_support
from pathlib import Path
import evaluate
evaluate.embeddings.encoder_available = lambda *arguments, **options: False  # the S1 rows alone, as without torch
evaluate.zero_shot.tagger_available = lambda *arguments, **options: False
folder = Path({folder!r})
evaluate.run(model_support.write_test_table(folder / "course"), folder / "results", "similarity", "", "2026-11-01")
loaded = sorted(name for name, module in sys.modules.items() if module is not None and name.split(".")[0] == "project")
assert not loaded, loaded
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
        self.assertIn("D2 criterion: 2 quizzes tagged by", output.getvalue())  # the tagger ran on the two quizzes
        self.assertIn("D2 level names: 2 quizzes tagged by", output.getvalue())
        self.assertIn("D2 criterion: Too few rated quizzes", output.getvalue())  # and D2's judge has nothing to measure
        self.assertEqual(len(list((self.folder / "cache" / evaluate.embeddings.CACHE_FOLDER).glob("*.npy"))), 2)
        evaluate.run(self.table, self.results, "topics", today=TODAY)
        topics = read_csv(self.files.topics)
        self.assertEqual(topics[0]["model"], "T1")
        self.assertEqual({row["scope"] for row in topics if row["scope"].startswith("deck ")}, {"deck 01", "deck 02", "deck 03"})  # by id
        self.assertTrue(all(name not in row["scope"] for row in topics for name in model_support.DECK_IDS))
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), ["difficulty.csv", "results.csv", "topics.csv"])

    def test_with_the_marks_alone_the_hand_columns_are_computed_and_nothing_is_written_back(self):
        """The capstone: the marks and the labels travel without their notes, the forms stay behind."""
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
            self.assertIn("- (correct) list[start:stop]  —  ", form)  # the table's options, no viewer needed
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

    def test_the_evaluation_runs_where_the_viewer_cannot_be_imported(self):
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
