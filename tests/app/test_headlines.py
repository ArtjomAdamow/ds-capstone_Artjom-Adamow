"""The section's numbers (app.headlines; nothing of Dash or the viewer), as the dashboard's tiles and the standalone
page's section show them: the verdict rule on its own cases, never a hand-written word; each generation's two numbers
from the newest run with their contrast, verdict and explanation; Claude named as the judge, never a hand."""

import model_support

import tempfile
import unittest
from pathlib import Path

from app import headlines as hl
from app.panels import d2_label, s2_label
from app.results import Results, course_note
from test_results import PROMPT_ROWS, write_results


class VerdictRuleTest(unittest.TestCase):
    """The verdict rule on its own cases, never a hand-written word."""

    def test_an_interval_by_whether_it_holds_zero(self):
        self.assertEqual(hl.interval_verdict(0.018, 0.093), "better")
        self.assertEqual(hl.interval_verdict(-0.358, 0.224), "within noise")
        self.assertEqual(hl.interval_verdict(-0.3, -0.1), "worse")
        self.assertEqual(hl.interval_verdict(0.0, 0.1), "within noise")  # 0 held

    def test_kappa_by_landis_and_kochs_words(self):
        for kappa, word in ((-0.1, "poor"), (0.0, "slight"), (0.2, "slight"), (0.21, "fair"), (0.40, "fair"), (0.41, "moderate"),
                            (0.6, "moderate"), (0.61, "substantial"), (0.8, "substantial"), (0.81, "almost perfect"), (1.0, "almost perfect")):
            self.assertEqual(hl.kappa_verdict(kappa), word, kappa)

    def test_a_floor_by_the_distance_above_it_at_each_bands_edge_and_below_the_floor(self):
        self.assertAlmostEqual(hl.distance(0.686, 0.04), (0.686 - 0.04) / 0.96)
        floor = 0.2
        for value, word in ((0.1, "below the floor"), (0.2, "near the floor"), (0.39, "near the floor"), (0.4, "above the floor"),
                            (0.59, "above the floor"), (0.6, "well above the floor"), (0.79, "well above the floor"), (0.8, "far above the floor"),
                            (1.0, "far above the floor")):
            self.assertEqual(hl.floor_verdict(value, floor), f"{word} by generic criteria", value)
        self.assertEqual(hl.floor_verdict(0.887, 0.638, "all cards"), "well above all cards by generic criteria")
        self.assertTrue(hl.floor_verdict(0.5, 0.2).endswith(hl.GENERIC))  # the generic criteria named

    def test_the_explanation_says_the_share_of_the_way_with_the_floor_and_the_top_named(self):
        text = hl.floor_explanation(0.686, 0.04, "random positions", "all five nearest in the card's deck", "4 %")
        self.assertEqual(text, "about two thirds of the way from random positions (4 %) to all five nearest in the card's deck")
        self.assertEqual(hl.floor_explanation(0.868, 0.51, "a random layout", "the top", "0.51"), "about three quarters of the way from a random layout (0.51) to the top")
        self.assertEqual(hl.floor_explanation(0.05, 0.04, "random positions", "the top", "4 %"), "at the level of random positions (4 %)")
        self.assertEqual(hl.floor_explanation(0.03, 0.04, "random positions", "the top", "4 %"), "below random positions (4 %)")
        self.assertEqual(hl.floor_explanation(0.99, 0.04, "random positions", "the top", "4 %"), "all the way from random positions (4 %) to the top")
        self.assertNotRegex(text, r"\d\.\d")  # never the distance as a number
        for share, word in ((0.1, "a tenth"), (0.2, "a fifth"), (0.25, "a quarter"), (0.33, "a third"), (0.5, "half"), (0.67, "two thirds"),
                            (0.75, "three quarters"), (0.8, "four fifths"), (0.9, "nine tenths"), (0.96, "all")):
            self.assertEqual(hl.share_words(share), word, share)

    def test_a_hit_rates_bands_and_a_lower_better_shares_mirrored_ones(self):
        for share, word in ((0.49, "weak"), (0.5, "fair"), (0.74, "fair"), (0.75, "good"), (0.89, "good"), (0.9, "very good")):
            self.assertEqual(hl.rate_verdict(share), f"{word} by generic criteria", share)
        for share, word in ((0.1, "very good"), (0.11, "good"), (0.25, "good"), (0.26, "fair"), (0.5, "fair"), (0.51, "weak")):
            self.assertEqual(hl.rate_verdict(share, lower_better=True), f"{word} by generic criteria", share)

    def test_the_format_shares_as_percentages_a_difference_in_points(self):
        self.assertEqual(hl.shown(0.686, True), "69 %")
        self.assertEqual(hl.shown(0.868, False), "0.87")
        self.assertEqual(hl.shown(None, True), "–")
        gain = hl.Headline("x", 0.734, hl.INTERVAL, "s", "S1", 0.68, 0.054, 0.018, 0.093)
        self.assertEqual(gain.gain, "+5.4 points (+1.8 to +9.3)")
        self.assertEqual(gain.verdict, "better")
        self.assertEqual(hl.Headline("x", 0.955, hl.RATE, "s", "S1", 0.92).gain, "+3.5 points")
        self.assertEqual(hl.Headline("x", 0.013, hl.KAPPA, "s", "D1", 0.413, percent=False).gain, "−0.40")
        self.assertEqual(hl.Headline("x", 0.75, hl.FLOOR, "s", "a coin", 0.5).gain, "")  # a floor is no gain
        self.assertEqual(hl.Headline("x", 0.75, hl.FLOOR, "s").verdict, hl.NO_CONTRAST)
        self.assertEqual(hl.Headline("x", 0.75, hl.RATE, "s").explanation, "")  # a hit rate needs none
        self.assertEqual(hl.plain_words("hybrid: chosen + MiniLM"), "the sentence encoder MiniLM next to the words")
        self.assertEqual(hl.plain_words("unknown"), "unknown")
        self.assertTrue(all(len(text.split()) <= 12 for text in hl.WHAT.values()))  # at most twelve words


class HeadlinesTest(unittest.TestCase):
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

    def test_s1s_numbers_are_the_same_deck_precision_against_b0_and_the_hand_precision_at_1(self):
        precision, p1 = hl.headlines(self.results, "S1")
        self.assertEqual((precision.label, precision.shown, precision.contrast_shown), ("same-deck precision@5", "60 %", "random positions 30 %"))
        self.assertEqual(precision.verdict, "above the floor by generic criteria")  # (0.6 − 0.3) / 0.7 = 0.43
        self.assertEqual(precision.explanation, "about half of the way from random positions (30 %) to all five nearest in the card's deck")  # 0.43: nearer half
        self.assertEqual((p1.label, p1.shown, p1.contrast, p1.verdict), ("judged precision@1", "92 %", None, "very good by generic criteria"))
        self.assertEqual(hl.chosen_label(self.results, "S1"), "chosen: uni+bigrams")

    def test_c1s_numbers_are_the_two_trustworthinesses_against_the_random_layout(self):
        cards, decks = hl.headlines(self.results, "C1")
        self.assertEqual((cards.label, cards.shown, cards.contrast_shown), ("trustworthiness of the card layouts", "0.90", "a random layout 0.50"))
        self.assertEqual(cards.verdict, "far above the floor by generic criteria")  # 0.8 of the way
        self.assertEqual(cards.explanation, "about four fifths of the way from a random layout (0.50) to no stranger by text among a card's nearest on the map")
        self.assertEqual((decks.shown, decks.contrast_shown, decks.verdict), ("0.80", "a random layout 0.40", "well above the floor by generic criteria"))
        self.assertEqual(hl.chosen_label(self.results, "C1"), "model")

    def test_t1s_numbers_are_the_agreement_against_chance_and_the_most_confident_quarter(self):
        agreement, confident = hl.headlines(self.results, "T1")
        self.assertEqual((agreement.shown, agreement.contrast_shown), ("64 %", "chance 22 %"))
        self.assertIn("1-NN gets 62 %", agreement.sentence)
        self.assertEqual(agreement.verdict, "well above the floor by generic criteria")  # (0.64 − 0.22) / 0.78 = 0.54
        self.assertEqual(agreement.explanation, "about half of the way from chance by the topic sizes (22 %) to every card in its scripted topic")
        self.assertEqual((confident.label, confident.shown, confident.contrast_shown), ("agreement of the 25 % most confident", "89 %", "all cards 64 %"))
        self.assertEqual(confident.verdict, "well above all cards by generic criteria")  # in the floor's place
        self.assertEqual(confident.explanation, "about two thirds of the way from all cards (64 %) to every card in its scripted topic")

    def test_d1s_numbers_are_the_within_deck_order_against_a_coin_and_the_weighted_kappa(self):
        order, kappa = hl.headlines(self.results, "D1")
        self.assertEqual((order.shown, order.contrast_shown, order.of), ("75 %", "a coin 50 %", "of 77 pairs"))
        self.assertEqual(order.verdict, "well above the floor by generic criteria")
        self.assertEqual(order.explanation, "about half of the way from a coin (50 %) to every pair in order")
        self.assertEqual((kappa.label, kappa.shown, kappa.contrast_shown, kappa.verdict), ("weighted kappa", "0.41", "always easy 0.00", "moderate"))
        self.assertIn("never accuracy", kappa.sentence)
        self.assertEqual(hl.chosen_label(self.results, "D1"), "text (chosen)")

    def test_d2s_numbers_are_its_gain_over_d1_within_noise_and_its_kappa(self):
        order, kappa = hl.headlines(self.results, "D2")
        self.assertEqual((order.shown, order.contrast_shown, order.gain), ("69 %", "D1 75 %", "−6.5 points (−35.8 to +22.4)"))
        self.assertEqual(order.verdict, "within noise")
        self.assertEqual((kappa.shown, kappa.contrast_shown, kappa.gain, kappa.verdict), ("0.02", "D1 0.41", "−0.39", "slight"))
        self.assertEqual(hl.chosen_label(self.results, "D2"), d2_label())
        self.assertEqual(d2_label(), "criterion, large")  # the app's D2: the criterion template on the large model

    def test_claude_is_the_judge_never_a_hand(self):
        """The slide brief's rule: the relevance marks, the quizzes' levels and the requests were judged by Claude."""
        for level in ("S1", "S2", "C1", "T1", "D1", "D2", "P1"):
            for headline in hl.headlines(self.results, level) or []:
                for words in (headline.label, headline.sentence, headline.explanation):
                    self.assertNotIn("hand", words.lower(), (level, words))
                    self.assertNotIn("authored", words.lower(), (level, words))
        self.assertEqual(hl.headlines(self.results, "D1")[0].label, "within-deck order on Claude's levels")
        self.assertFalse(any("hand" in words for words in hl.PLAIN.values()))


class PromptHeadlinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        folder = Path(cls._tmp.name) / "results"
        folder.mkdir()
        (folder / "prompt.csv").write_text(PROMPT_ROWS, encoding="utf-8")
        cls.results = Results(folder, course_note(model_support.DECK_IDS))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_hybrids_numbers_stand_against_the_tfidf_card_route(self):
        results = Results(self.results.folder, self.results.note, chosen={"P1": "hybrid-minilm: card"})
        right, wrong = hl.headlines(results, "P1")
        self.assertEqual((right.shown, right.contrast_shown, right.gain, right.verdict), ("76 %", "TF-IDF 75 %", "+1.0 points", "good by generic criteria"))
        self.assertEqual((wrong.shown, wrong.contrast_shown, wrong.verdict), ("6 %", "TF-IDF 20 %", "very good by generic criteria"))
        self.assertEqual(hl.chosen_label(results, "P1"), "hybrid-minilm: card")


if __name__ == "__main__":
    unittest.main()
