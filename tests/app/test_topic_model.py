"""The topic model T1: the centroids of the scripted topics, the leave-one-out scoring, the margin, the coherence,
the edge cases (a two-card topic, an empty topic, a one-card topic, a single centroid, a deck with one topic),
determinism, the agreement measure, and the judges of the measurement (the neighbour rule, logistic regression,
chance). Nothing of the viewer: the scripted topics read from the course file and the derived course file are the
glue's (test_course.py, test_derived.py)."""

import model_support  # noqa: F401  (the helper's package on the path)

import math
import unittest

import numpy as np

from app.cards import CardRecord
from app.models import evaluation, similar
from app.models.topic_model import FOCUS_MARGIN, FULL_MARGIN, TopicModel, TopicSeed, unit


class FixedRows:
    """A stand-in vectoriser: the rows are given, one per card in card order (not unit length: the index normalises)."""

    def __init__(self, rows):
        self.rows = np.array(rows, dtype=float)

    def fit_transform(self, texts):
        assert len(texts) == len(self.rows)
        return self.rows


def card(card_id: str, deck: str) -> CardRecord:
    return CardRecord(card_id, deck, f"Question {card_id}?", f"Answer {card_id}.")


def cosine(left, right) -> float:
    return float(unit(np.array(left, dtype=float)) @ unit(np.array(right, dtype=float)))


# Deck "Clear": P = {p1, p2, m}, Q = {q1, q2}, R = {}; m sits between P and Q, a little nearer to Q
P1, P2, Q1, Q2 = (1, 0, 0, 0), (1, 0.2, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0.2)
M = (0.95, 0, 1.0, 0)
CLEAR = [card("p1", "Clear"), card("p2", "Clear"), card("m", "Clear"), card("q1", "Clear"), card("q2", "Clear")]
CLEAR_ROWS = [P1, P2, M, Q1, Q2]
CLEAR_SEEDS = [
    TopicSeed("Clear", "P", "First", False, ("p1", "p2", "m")),
    TopicSeed("Clear", "Q", "Second", False, ("q1", "q2")),
    TopicSeed("Clear", "R", "Empty", False, ()),
]
# Deck "Loo": A = {a1, a2} at right angles, B = {b1, b2} between them: with its own vote every A card would stay
A1, A2, B = (1, 0, 0, 0), (0, 1, 0, 0), (0.6, 0.8, 0, 0)
LOO = [card("a1", "Loo"), card("a2", "Loo"), card("b1", "Loo"), card("b2", "Loo")]
LOO_ROWS = [A1, A2, B, B]
LOO_SEEDS = [TopicSeed("Loo", "A", "Apart", False, ("a1", "a2")), TopicSeed("Loo", "B", "Between", False, ("b1", "b2"))]
# Deck "Single": X = {x1} (one card), Y = {y1, y2}; deck "Alone": its one topic Z; deck "Lonely": one topic, one card
SINGLE = [card("x1", "Single"), card("y1", "Single"), card("y2", "Single")]
SINGLE_ROWS = [(1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0.2, 1)]
SINGLE_SEEDS = [TopicSeed("Single", "X", "One", False, ("x1",)), TopicSeed("Single", "Y", "Two", False, ("y1", "y2"))]
ALONE = [card("z1", "Alone"), card("z2", "Alone"), card("w1", "Lonely")]
ALONE_ROWS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
ALONE_SEEDS = [TopicSeed("Alone", "Z", "Only", False, ("z1", "z2")), TopicSeed("Lonely", "W", "Single", False, ("w1",))]


def fit(cards, rows, seeds, margin=FOCUS_MARGIN) -> TopicModel:
    return TopicModel(similar.SimilarityIndex.fit(cards, FixedRows(rows)), seeds, margin)


class NearestCentroidTest(unittest.TestCase):
    def setUp(self):
        self.model = fit(CLEAR, CLEAR_ROWS, CLEAR_SEEDS)

    def test_a_card_takes_the_nearest_centroid_and_its_margin_is_the_best_cosine_minus_the_second(self):
        model = self.model
        self.assertEqual(model.topic_of["p1"], "P")
        own = cosine(P1, unit(np.array(P2, dtype=float)) + unit(np.array(M, dtype=float)))  # P without p1: p2 and m
        other = cosine(P1, unit(np.array(Q1, dtype=float)) + unit(np.array(Q2, dtype=float)))  # Q: the mean of its unit rows
        self.assertAlmostEqual(model.margin_of["p1"], own - other)
        self.assertAlmostEqual(other, 0.0)
        self.assertGreater(own, 0.9)

    def test_the_centroid_is_the_normalised_mean_of_the_topics_unit_rows(self):
        # q1 against P: all three of P's rows (q1 is not in P), normalised after summing the unit rows
        centroid = unit(np.array(P1, dtype=float)) + unit(np.array(P2, dtype=float)) + unit(np.array(M, dtype=float))
        own = cosine(Q1, Q2)
        self.assertAlmostEqual(self.model.margin_of["q1"], own - cosine(Q1, centroid))

    def test_a_card_between_two_topics_moves_to_the_nearer_one_with_a_small_margin_and_is_hidden(self):
        model = self.model
        self.assertEqual(model.topic_of["m"], "Q")
        p_centroid = unit(np.array(P1, dtype=float)) + unit(np.array(P2, dtype=float))  # without m itself
        q_centroid = unit(np.array(Q1, dtype=float)) + unit(np.array(Q2, dtype=float))
        margin = cosine(M, q_centroid) - cosine(M, p_centroid)
        self.assertAlmostEqual(model.margin_of["m"], margin)
        self.assertLess(margin, FOCUS_MARGIN)
        self.assertEqual(model.hidden, {"m"})
        self.assertEqual(model.members[("Clear", "P")], ["p1", "p2"])
        self.assertEqual(model.members[("Clear", "Q")], ["m", "q1", "q2"])  # deck order
        self.assertEqual(model.moved, 1)

    def test_an_empty_topic_has_no_centroid_no_members_and_no_coherence(self):
        model = self.model
        self.assertEqual(model.members[("Clear", "R")], [])
        self.assertTrue(math.isnan(model.coherence[("Clear", "R")]))
        self.assertNotIn("R", model.topic_of.values())
        self.assertEqual(model.emptied, [])  # it never had cards: not emptied by the model

    def test_a_two_card_topics_cards_are_each_others_centroid(self):
        model = self.model
        self.assertEqual((model.topic_of["q1"], model.topic_of["q2"]), ("Q", "Q"))
        p_centroid = sum(unit(np.array(row, dtype=float)) for row in (P1, P2, M))
        self.assertAlmostEqual(model.margin_of["q1"], cosine(Q1, Q2) - cosine(Q1, p_centroid))  # Q without q1: q2 alone
        self.assertAlmostEqual(model.margin_of["q2"], cosine(Q2, Q1) - cosine(Q2, p_centroid))  # and the other way round

    def test_the_coherence_is_the_share_of_a_topics_scripted_cards_nearest_to_itself(self):
        self.assertAlmostEqual(self.model.coherence[("Clear", "P")], 2 / 3)
        self.assertAlmostEqual(self.model.coherence[("Clear", "Q")], 1.0)

    def test_the_agreement_and_the_margin_table(self):
        model = self.model
        self.assertAlmostEqual(model.agreement, 0.8)
        self.assertAlmostEqual(model.agreement_of(["p1", "m"]), 0.5)
        self.assertTrue(math.isnan(model.agreement_of([])))
        share, agreement = model.kept(0.0)
        self.assertEqual((share, agreement), (1.0, 0.8))
        share, agreement = model.kept(0.5)
        self.assertEqual((share, agreement), (0.8, 1.0))  # m is the only card below 0.5

    def test_the_threshold_is_a_parameter(self):
        strict = fit(CLEAR, CLEAR_ROWS, CLEAR_SEEDS, margin=0.9)
        self.assertEqual(strict.hidden, {card_id for card_id, margin in strict.margin_of.items() if margin < 0.9})
        self.assertIn("q1", strict.hidden)
        self.assertEqual(strict.topic_of, self.model.topic_of)  # the threshold hides, it never moves a card

    def test_ties_go_to_the_topic_that_comes_first_in_the_file(self):
        # A card at right angles to every centroid scores 0 everywhere: the first topic wins with margin 0
        cards = CLEAR + [card("far", "Clear")]
        seeds = [TopicSeed("Clear", "P", "First", False, ("p1", "p2", "m", "far"))] + CLEAR_SEEDS[1:]
        model = fit(cards, CLEAR_ROWS + [(0, 0, 0, 0)], seeds)
        self.assertEqual((model.topic_of["far"], model.margin_of["far"]), ("P", 0.0))
        self.assertIn("far", model.hidden)


class LeaveOneOutTest(unittest.TestCase):
    def test_a_card_never_votes_for_itself(self):
        model = fit(LOO, LOO_ROWS, LOO_SEEDS)
        # a1 against A without itself is a2 alone (cosine 0); with its own vote A's centroid would be nearer than B
        self.assertGreater(cosine(A1, np.add(A1, A2)), cosine(A1, B))
        self.assertEqual(model.topic_of["a1"], "B")
        self.assertAlmostEqual(model.margin_of["a1"], cosine(A1, B) - cosine(A1, A2))
        self.assertEqual(model.topic_of["a2"], "B")
        self.assertEqual(model.members[("Loo", "A")], [])
        self.assertEqual(model.emptied, [("Loo", "A")])  # it had cards and keeps none: it keeps its dot
        self.assertEqual(model.coherence[("Loo", "A")], 0.0)
        self.assertEqual(model.coherence[("Loo", "B")], 1.0)

    def test_a_one_card_topic_has_no_centroid_so_its_card_goes_to_the_nearest_other_topic(self):
        model = fit(SINGLE, SINGLE_ROWS, SINGLE_SEEDS)
        self.assertEqual(model.topic_of["x1"], "Y")
        self.assertEqual(model.margin_of["x1"], FULL_MARGIN)  # Y was the only centroid to compare with
        self.assertNotIn("x1", model.hidden)
        self.assertEqual(model.emptied, [("Single", "X")])
        self.assertEqual(model.coherence[("Single", "X")], 0.0)
        self.assertEqual(model.members[("Single", "Y")], ["x1", "y1", "y2"])

    def test_a_deck_with_one_topic_keeps_its_cards_with_the_full_margin(self):
        model = fit(ALONE, ALONE_ROWS, ALONE_SEEDS)
        self.assertEqual((model.topic_of["z1"], model.topic_of["z2"]), ("Z", "Z"))
        self.assertEqual((model.margin_of["z1"], model.margin_of["z2"]), (FULL_MARGIN, FULL_MARGIN))
        self.assertEqual((model.topic_of["w1"], model.margin_of["w1"]), ("W", FULL_MARGIN))  # nothing to compare with
        self.assertEqual(model.hidden, set())
        self.assertEqual(model.agreement, 1.0)

    def test_the_fit_is_deterministic(self):
        first = fit(CLEAR + LOO, CLEAR_ROWS + LOO_ROWS, CLEAR_SEEDS + LOO_SEEDS)
        second = fit(CLEAR + LOO, CLEAR_ROWS + LOO_ROWS, CLEAR_SEEDS + LOO_SEEDS)
        self.assertEqual(first.topic_of, second.topic_of)
        self.assertEqual(first.margin_of, second.margin_of)
        self.assertEqual(first.members, second.members)


class TestCourseTest(unittest.TestCase):
    """T1 on the models' table fixture: the scripted topics are kept, one card sits between two topics."""

    def test_the_model_keeps_the_scripted_topics_and_hides_the_card_between_two_topics(self):
        model = model_support.course_model().topic_model
        self.assertEqual(model.moved, 0)
        self.assertEqual(model.hidden, {model_support.HIDDEN})
        self.assertEqual(model.emptied, [])
        self.assertEqual(model.members[("Alpha", "02")], model_support.SCRIPTED[("01", "02")])

    def test_the_moving_course_moves_one_card(self):
        model = model_support.course_model(model_support.moving_cards(), model_support.seeds_of(model_support.MOVING_SCRIPTED)).topic_model
        self.assertEqual((model.topic_of[model_support.MOVED], model.moved), ("02", 1))
        self.assertEqual(model.members[("Gamma", "02")], [model_support.MOVED, "103003", "103005"])


class TopicJudgesTest(unittest.TestCase):
    """The judges of the T1 measurement in models.evaluation."""

    def test_the_neighbour_rule_takes_the_label_of_the_most_similar_other_card(self):
        rows = [(1, 0), (0.9, 0.1), (0, 1), (0.1, 0.9)]
        self.assertEqual(evaluation.nearest_labels(rows, ["a", "b", "c", "d"]), ["b", "a", "d", "c"])
        self.assertEqual(evaluation.nearest_labels([(1, 0)], ["a"]), ["a"])

    def test_logistic_regression_leave_one_out_predicts_separable_topics_and_a_lone_class(self):
        rows = [(1, 0, 0), (0.9, 0.1, 0), (0.8, 0, 0.2), (0, 1, 0), (0.1, 0.9, 0), (0, 0.8, 0.2)]
        labels = ["x", "x", "x", "y", "y", "y"]
        self.assertEqual(evaluation.leave_one_out_logistic(rows, labels), labels)
        self.assertEqual(evaluation.leave_one_out_logistic(rows[:3], ["x", "x", "x"]), ["x", "x", "x"])

    def test_agreement_and_chance_by_sizes(self):
        self.assertEqual(evaluation.agreement(["a", "b", "b"], ["a", "a", "b"]), 2 / 3)
        self.assertTrue(math.isnan(evaluation.agreement([], [])))
        self.assertAlmostEqual(evaluation.chance_by_sizes(["a", "a", "b", "b"]), 0.5)
        self.assertAlmostEqual(evaluation.chance_by_sizes(["a"] * 3 + ["b"]), 0.625)

    def test_the_random_floor_takes_given_points(self):
        ids = ["a1", "a2", "b1", "b2"]
        deck_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
        grouped = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0]]
        self.assertAlmostEqual(evaluation.random_floor(ids, deck_of, grouped, k=1), 1.0)
        swapped = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0], [0.1, 0.9, 0]]
        self.assertAlmostEqual(evaluation.random_floor(ids, deck_of, swapped, k=1), 0.0)


if __name__ == "__main__":
    unittest.main()
