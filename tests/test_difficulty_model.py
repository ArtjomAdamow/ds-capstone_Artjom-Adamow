"""The difficulty model D1: the split, features that ignore the course order, the deck shares, the out-of-fold metrics
with their spread, the predictions of the unrated quizzes, the feature set with the answer options' statistics, and
D2 alone judged the same way without folds, on D1's frame and supplying no feature."""

import model_support  # noqa: F401  (the package on the path)

import random
import unittest

import numpy as np

from models.cards import CardRecord
from models import similar
from models.difficulty_model import (
    COURSE_SHARES, FEATURE_SETS, LEVELS, OPTION_FEATURES, TEXT_FEATURES, band, band_labels, features, fit_difficulty, judge_tags,
    option_statistics, share_labels, split,
)

TOPICS = model_support.TOPICS
# 3 decks: 12 rated quizzes (8 / 3 / 1), 2 unrated quizzes, 3 plain cards and a flashcard of the quiz file (stack 4)
course = model_support.difficulty_course


def shuffled_and_renamed(cards: list[CardRecord]) -> list[CardRecord]:
    """The same cards in another order and with other ids: nothing a card's level may depend on."""
    copies = [CardRecord(f"x{number:04d}", card.deck, card.question, card.answer, card.stack, card.difficulty, card.explain,
                         card.options, card.answers, card.correct)
              for number, card in enumerate(cards)]
    random.Random(7).shuffle(copies)
    return copies


def by_question(cards, mapping):
    return {card.question: mapping[card.id] for card in cards if card.id in mapping}


def fake_tags(cards) -> dict[str, tuple[float, float, float]]:
    """Stand-in tags of D2 by card id, a function of the question alone (so a shuffled, renamed course gets the
    same): the hard probability grows with the question's length, as the course's hard quizzes are the long ones."""
    tags = {}
    for card in cards:
        hard = min(0.8, len(card.question.split()) / 50)
        medium = min(0.3, 1.0 - hard)
        tags[card.id] = (1.0 - hard - medium, medium, hard)
    return tags


class SplitAndFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.cards = course()
        self.index = similar.SimilarityIndex.fit(self.cards)

    def test_the_rated_quizzes_are_trained_on_and_only_the_unrated_quizzes_are_predicted(self):
        rated, unrated = split(self.cards)
        self.assertEqual(len(rated), 36)
        self.assertEqual({card.stack for card in rated}, {3})
        self.assertEqual(len(unrated), 6)
        self.assertEqual({card.stack for card in unrated}, {2})
        # The flashcards are in neither: Q_and_A.json's (stack 1) and the quiz file's (stack 4) alike
        flashcards = {card.id for card in self.cards if card.stack in (1, 4)}
        self.assertEqual(len(flashcards), 12)
        self.assertEqual({card.stack for card in self.cards if card.id in flashcards}, {1, 4})
        self.assertFalse(flashcards & {card.id for card in rated + unrated})

    def test_no_feature_depends_on_the_course_order_or_the_ids(self):
        other = shuffled_and_renamed(self.cards)
        other_index = similar.SimilarityIndex.fit(other)
        for feature_set in FEATURE_SETS:
            mine = features(self.cards, self.index, feature_set)
            theirs = features(other, other_index, feature_set)
            row_of = {card.question: row for row, card in enumerate(other)}
            order = [row_of[card.question] for card in self.cards]
            np.testing.assert_allclose(mine.dense, theirs.dense[order], atol=1e-12, err_msg=feature_set)
            if mine.rows is not None:
                np.testing.assert_allclose(mine.rows.toarray(), theirs.rows[order].toarray(), atol=1e-12)

    def test_an_unknown_feature_set_is_refused(self):
        with self.assertRaises(ValueError):
            features(self.cards, self.index, "card order")

    def test_the_feature_set_with_the_options_puts_their_statistics_after_the_text_ones_and_d2_supplies_none(self):
        self.assertEqual(FEATURE_SETS, ("text", "text+course", "text+course+tfidf", "text+options"))  # no set reads D2's tags
        data = features(self.cards, self.index, "text+options")
        self.assertEqual(data.dense.shape, (len(self.cards), len(TEXT_FEATURES) + len(OPTION_FEATURES)))
        self.assertIsNone(data.rows)
        np.testing.assert_allclose(data.dense[:, :len(TEXT_FEATURES)], features(self.cards, self.index, "text").dense)
        statistics = option_statistics(self.cards, self.index)
        np.testing.assert_allclose(data.dense[:, len(TEXT_FEATURES):], statistics)
        without = [position for position, card in enumerate(self.cards) if not card.options]
        self.assertEqual(len(without), len(self.cards) - 6)  # the rated quizzes and the flashcards of the test course carry no options
        np.testing.assert_array_equal(statistics[without], 0.0)
        # the names: nothing counts the options or the correct ones, and no feedback is read
        self.assertEqual(OPTION_FEATURES, ("code in the options", "digits in the options", "option length spread (log)", "closest distractor"))

    def test_the_option_statistics_read_code_digits_the_spread_and_the_closest_distractor_never_the_counts(self):
        def quiz(options, correct):
            return CardRecord("q", "Alpha", "What is a slice?", "a", stack=2, options=options, correct=correct, answers=tuple("" for _ in options))

        plain = option_statistics([quiz(("a slice of a list", "a loop in python", "a git branch", "a sample"), (0,))], self.index)[0]
        self.assertEqual(list(plain[:2]), [0.0, 0.0])
        self.assertAlmostEqual(plain[2], np.log1p(np.std([5, 4, 3, 2])))
        coded = option_statistics([quiz(("`list[1:3]`", "a loop", "4 items"), (0,))], self.index)[0]
        self.assertEqual(list(coded[:2]), [1.0, 1.0])
        # the closest distractor: a wrong option that shares the correct one's words lies nearer than one that does not
        near = option_statistics([quiz(("a slice of a list", "a slice of a list in python", "a git branch"), (0,))], self.index)[0][3]
        far = option_statistics([quiz(("a slice of a list", "a git branch", "a sample"), (0,))], self.index)[0][3]
        self.assertGreater(near, 0.5)
        self.assertGreater(near, far)
        # a many-choice quiz with every option correct has no distractor; one without options has nothing
        self.assertEqual(option_statistics([quiz(("a slice", "a list"), (0, 1))], self.index)[0][3], 0.0)
        np.testing.assert_array_equal(option_statistics([CardRecord("p", "Alpha", "Q?", "A.")], self.index)[0], 0.0)
        # two quizzes of the same option texts, one single-choice and one many-choice, differ in no count: the counts are no feature
        single = option_statistics([quiz(("a slice", "a list", "a loop"), (0,))], self.index)[0]
        many = option_statistics([quiz(("a slice", "a list", "a loop"), (0, 1))], self.index)[0]
        np.testing.assert_array_equal(single[:3], many[:3])


class DeckSharesTest(unittest.TestCase):
    def test_the_bands_of_a_deck_of_n_test_quizzes(self):
        expected = {1: (1, 0, 0), 2: (1, 1, 0), 3: (2, 1, 0), 4: (3, 1, 0), 5: (3, 2, 0), 6: (4, 2, 0),
                    7: (5, 1, 1), 8: (5, 2, 1), 9: (6, 2, 1), 10: (7, 2, 1), 11: (7, 3, 1), 12: (8, 3, 1)}
        for n, counts in expected.items():
            levels = [band(rank, n, COURSE_SHARES) for rank in range(1, n + 1)]
            self.assertEqual(tuple(levels.count(number) for number in range(3)), counts, n)
            self.assertEqual(levels, sorted(levels, reverse=True), "the highest scores are the hardest")

    def test_the_cross_validation_cuts_each_deck_by_its_own_counts(self):
        cards = [CardRecord(f"{deck}{number}", deck, f"{deck} {number}?", "a") for deck in "AB" for number in range(6)]
        scores = np.array([0.1, 1.9, 0.5, 1.2, 0.2, 0.3] + [0.4, 0.3, 1.5, 0.2, 1.1, 0.9])
        labels = share_labels(cards, scores, {"A": (4, 1, 1), "B": (3, 2, 1)})
        self.assertEqual(list(labels[:6]), [0, 2, 0, 1, 0, 0])
        self.assertEqual(list(labels[6:]), [0, 0, 2, 0, 1, 1])

    def test_a_deck_without_rated_quizzes_takes_the_course_shares(self):
        cards = [CardRecord(f"N{number}", "New", f"q{number}?", "a") for number in range(12)]
        labels = band_labels(cards, np.arange(12, dtype=float), {})
        self.assertEqual([list(labels).count(number) for number in range(3)], [8, 3, 1])
        self.assertEqual(labels[11], 2)  # the highest score is the hard one


class FitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = course()
        cls.index = similar.SimilarityIndex.fit(cls.cards)
        cls.report = fit_difficulty(cls.cards, cls.index)

    def test_the_metrics_are_out_of_fold_with_their_spread_next_to_the_baselines(self):
        report = self.report
        self.assertTrue(report.measured)
        self.assertEqual((report.rated_counts, report.folds, report.repeats), ((24, 9, 3), 3, 3))  # 3 hard: 3 folds
        for name in ("macro_f1", "balanced_accuracy", "kappa", "accuracy", "recall_hard"):
            mean, sd = report.metrics[name]
            self.assertTrue(0.0 <= mean <= 1.0 and sd >= 0.0, name)
        self.assertEqual(set(report.baselines), {"always easy", "similarity"})
        self.assertAlmostEqual(report.baselines["always easy"]["accuracy"][0], 24 / 36)
        self.assertAlmostEqual(report.baselines["always easy"]["macro_f1"][0], (2 * (24 / 36) / (1 + 24 / 36)) / 3)
        self.assertGreater(report.metrics["macro_f1"][0], report.baselines["always easy"]["macro_f1"][0])
        self.assertGreater(report.metrics["kappa"][0], report.baselines["always easy"]["kappa"][0])
        self.assertEqual(report.confusion.shape, (3, 3))
        np.testing.assert_allclose(report.confusion.sum(axis=1), 1.0)

    def test_every_unrated_quiz_gets_a_level_by_its_decks_shares_and_no_plain_card_does(self):
        report = self.report
        unrated = [card for card in self.cards if card.stack == 2]
        self.assertEqual(set(report.predictions), {card.id for card in unrated})
        self.assertEqual(set(report.cut_point_predictions), set(report.predictions))
        self.assertTrue(set(report.predictions.values()) <= set(LEVELS))
        for deck in TOPICS:
            pair = [card for card in unrated if card.deck == deck]
            self.assertEqual(sorted(report.predictions[card.id] for card in pair), ["easy", "medium"])  # 2 of a deck
            longer = max(pair, key=lambda card: len(card.question))
            self.assertEqual(report.predictions[longer.id], "medium", "the long scenario is the harder one")
        self.assertEqual(report.predicted_counts, (3, 3, 0))

    def test_the_predictions_and_the_metrics_do_not_depend_on_the_course_order(self):
        other = shuffled_and_renamed(self.cards)
        theirs = fit_difficulty(other, similar.SimilarityIndex.fit(other))
        self.assertEqual(by_question(other, theirs.predictions), by_question(self.cards, self.report.predictions))
        mine_scores, their_scores = by_question(self.cards, self.report.scores), by_question(other, theirs.scores)
        for question, score in mine_scores.items():
            self.assertAlmostEqual(their_scores[question], score, places=6)
        for name, (mean, sd) in self.report.metrics.items():
            self.assertAlmostEqual(theirs.metrics[name][0], mean, places=9, msg=name)

    def test_every_deck_has_its_own_part_of_the_cross_validation_and_of_the_predictions(self):
        report = self.report
        self.assertEqual(set(report.decks), set(TOPICS))
        for deck, part in report.decks.items():
            self.assertEqual(part.scope, deck)
            self.assertEqual(part.rated_counts, (8, 3, 1))
            self.assertEqual((part.folds, part.repeats), (report.folds, report.repeats))
            self.assertTrue(part.measured)
            self.assertEqual(set(part.baselines), {"always easy", "similarity"})
            self.assertAlmostEqual(part.baselines["always easy"]["accuracy"][0], 8 / 12)
            self.assertEqual(part.confusion.shape, (3, 3))
            unrated = {card.id for card in self.cards if card.deck == deck and card.stack == 2}
            self.assertEqual(set(part.predictions), unrated)
            self.assertEqual(part.unrated, 2)
        self.assertEqual(sum(len(part.predictions) for part in report.decks.values()), len(report.predictions))
        # the decks' accuracies average to the course's: every deck has 12 of the 36 rated quizzes
        mean_accuracy = np.mean([part.metrics["accuracy"][0] for part in report.decks.values()])
        self.assertAlmostEqual(mean_accuracy, report.metrics["accuracy"][0], places=9)

    def test_a_topics_part_counts_the_topics_quizzes_with_the_decks_metrics(self):
        deck = self.report.for_deck("Alpha")
        self.assertEqual(set(deck.levels), {card.id for card in self.cards if card.deck == "Alpha" and card.stack == 3})
        self.assertEqual(set(deck.unrated_ids), {card.id for card in self.cards if card.deck == "Alpha" and card.stack == 2})
        ids = ["301001", "301009", "301012", "201101", "101200"]  # an easy, a medium and the hard quiz, an unrated one, a flashcard
        part = deck.for_topic("Some topic", ids)
        self.assertEqual((part.scope, part.rated_counts, part.unrated), ("Some topic", (1, 1, 1), 1))
        self.assertEqual(set(part.levels), {"301001", "301009", "301012"})
        self.assertEqual(set(part.predictions), {"201101"})
        self.assertEqual(part.predictions["201101"], deck.predictions["201101"])
        self.assertEqual((part.metrics, part.baselines), (deck.metrics, deck.baselines))  # the deck's numbers
        np.testing.assert_array_equal(part.confusion, deck.confusion)
        self.assertIs(part.parent, deck)
        self.assertEqual(part.decks, {})
        empty = deck.for_topic("Nothing", ["101201"])
        self.assertEqual((empty.rated, empty.unrated, empty.predictions), (0, 0, {}))
        self.assertTrue(empty.measured)  # the deck's metrics still show

    def test_a_deck_without_rated_quizzes_has_predictions_but_no_metrics(self):
        cards = course() + [
            CardRecord("204101", "Delta", *model_support.quiz("Alpha", 30, "easy"), stack=2),
            CardRecord("204102", "Delta", *model_support.quiz("Alpha", 31, "hard"), stack=2),
        ]
        report = fit_difficulty(cards, similar.SimilarityIndex.fit(cards))
        part = report.for_deck("Delta")
        self.assertFalse(part.measured)
        self.assertTrue(part.note)
        self.assertEqual(sorted(part.predictions.values()), ["easy", "medium"])  # the course's shares, 2 quizzes
        empty = report.for_deck("Nowhere")
        self.assertEqual((empty.rated, empty.unrated, empty.predictions), (0, 0, {}))
        self.assertTrue(empty.note)

    def test_too_few_rated_quizzes_give_no_model(self):
        cards = model_support.CARDS  # one rated card
        report = fit_difficulty(cards, similar.SimilarityIndex.fit(cards))
        self.assertFalse(report.measured)
        self.assertEqual(report.predictions, {})
        self.assertTrue(report.note)

    def test_the_feature_set_with_the_options_is_measured_out_of_fold_like_the_others(self):
        report = fit_difficulty(self.cards, self.index, "text+options")
        self.assertEqual((report.feature_set, report.folds, report.repeats), ("text+options", 3, 3))
        self.assertTrue(report.measured)
        self.assertEqual(set(report.predictions), set(self.report.predictions))
        self.assertEqual(report.predicted_counts, (3, 3, 0))  # two per deck: an easy and a medium
        other = shuffled_and_renamed(self.cards)
        theirs = fit_difficulty(other, similar.SimilarityIndex.fit(other), "text+options")
        self.assertEqual(by_question(other, theirs.predictions), by_question(self.cards, report.predictions))


class JudgeTagsTest(unittest.TestCase):
    """D2 alone: one pass on the expected level of the tags, no folds, the same label rules and measures."""

    @classmethod
    def setUpClass(cls):
        cls.cards = course()
        cls.tags = fake_tags(cls.cards)
        cls.report = judge_tags(cls.cards, cls.tags)

    def test_the_rated_quizzes_are_measured_in_one_pass_without_folds(self):
        report = self.report
        self.assertEqual((report.feature_set, report.rated_counts, report.folds, report.repeats), ("premise", (24, 9, 3), 0, 0))
        self.assertTrue(report.measured)
        for name in ("macro_f1", "kappa", "accuracy", "recall_hard"):
            mean, sd = report.metrics[name]
            self.assertTrue(0.0 <= mean <= 1.0 and sd == 0.0, name)  # one pass: no spread
        self.assertEqual(set(report.most_likely), set(report.metrics))
        self.assertEqual(set(report.baselines), {"always easy"})
        self.assertAlmostEqual(report.baselines["always easy"]["accuracy"][0], 24 / 36)
        self.assertEqual(report.confusion.shape, (3, 3))
        self.assertGreater(report.metrics["spearman"][0], 0.5)  # the fake tags grow with the question's length, as the levels do
        self.assertEqual(report.model, "D2")

    def test_every_deck_has_its_part_of_the_one_pass_and_of_the_predictions(self):
        report = self.report
        self.assertEqual(set(report.decks), set(TOPICS))
        for deck in TOPICS:
            part = report.for_deck(deck)
            self.assertEqual((part.model, part.scope, part.rated_counts, part.unrated, part.folds), ("D2", deck, (8, 3, 1), 2, 0))
            self.assertTrue(part.measured)
            self.assertEqual(set(part.baselines), {"always easy"})  # no similarity baseline: nothing of the index
            self.assertAlmostEqual(part.baselines["always easy"]["accuracy"][0], 8 / 12)
            self.assertEqual(part.confusion.shape, (3, 3))
            self.assertEqual(set(part.predictions), {card.id for card in self.cards if card.deck == deck and card.stack == 2})
        self.assertEqual(report.for_deck("nowhere").model, "D2")  # a deck without quizzes says so under D2's name

    def test_every_unrated_quiz_gets_a_level_by_its_decks_shares_and_its_score_is_the_expected_level(self):
        report = self.report
        unrated = [card for card in self.cards if card.stack == 2]
        self.assertEqual(set(report.predictions), {card.id for card in unrated})
        self.assertEqual(set(report.cut_point_predictions), set(report.predictions))
        for card in unrated:
            easy, medium, hard = self.tags[card.id]
            self.assertAlmostEqual(report.scores[card.id], medium + 2 * hard)
        for deck in TOPICS:
            pair = [card for card in unrated if card.deck == deck]
            self.assertEqual(sorted(report.predictions[card.id] for card in pair), ["easy", "medium"])
            longer = max(pair, key=lambda card: len(card.question))
            self.assertEqual(report.predictions[longer.id], "medium")

    def test_the_judgement_does_not_depend_on_the_course_order(self):
        other = shuffled_and_renamed(self.cards)
        theirs = judge_tags(other, fake_tags(other))
        self.assertEqual(by_question(other, theirs.predictions), by_question(self.cards, self.report.predictions))
        for name, (mean, _sd) in self.report.metrics.items():
            self.assertAlmostEqual(theirs.metrics[name][0], mean, places=9, msg=name)

    def test_with_one_rated_level_nothing_is_measured_but_the_unrated_quizzes_get_their_bands(self):
        cards = model_support.CARDS  # one rated card, one unrated quiz
        report = judge_tags(cards, fake_tags(cards))
        self.assertFalse(report.measured)
        self.assertTrue(report.note)
        self.assertEqual(report.predictions, {"201002": "easy"})  # one extra quiz of a deck: easy


if __name__ == "__main__":
    unittest.main()
