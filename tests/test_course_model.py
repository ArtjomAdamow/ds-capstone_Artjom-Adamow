"""The course model: deck points on the unit sphere, the cards of every model topic on the surface of the topic's
sphere, and the connections of the decks and of every topic's cards."""

import model_support  # noqa: F401  (the package on the path, the test course as card records)

import math
import unittest

import numpy as np

from models.course_model import CourseModel

SCRIPTED = model_support.SCRIPTED
NAME_OF = model_support.NAME_OF


class CourseModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model_support.course_model()
        cls.cards = model_support.CARDS

    def test_deck_points_lie_on_the_unit_sphere_keyed_by_deck_id_in_id_order(self):
        self.assertEqual(list(self.model.deck_points), ["01", "02", "03"])
        self.assertEqual(self.model.deck_names, ["Alpha", "Beta", "Gamma"])
        for point in self.model.deck_points.values():
            self.assertAlmostEqual(float(np.linalg.norm(point)), 1.0)
        self.assertEqual((self.model.deck_of["101001"], self.model.deck_of["103004"]), ("01", "03"))  # a card's deck, by id

    def test_the_topics_are_the_models_membership_and_every_card_of_a_topic_with_two_cards_or_more_sits_on_the_surface(self):
        model = self.model
        self.assertEqual(set(model.topics), set(SCRIPTED))
        for (deck_id, topic_id), placed in model.topics.items():
            self.assertEqual(placed.ids, model.topic_model.members[(NAME_OF[deck_id], topic_id)])
            self.assertEqual(placed.ids, SCRIPTED[(deck_id, topic_id)])  # on this course the model keeps the script
            if len(placed.ids) >= 2:
                self.assertEqual(sorted(placed.positions), sorted(placed.ids))
                for point in placed.positions.values():
                    self.assertAlmostEqual(float(np.linalg.norm(point)), 1.0)
            else:
                self.assertEqual((placed.positions, placed.links), ({}, []))
        self.assertEqual(set(model.card_points), {card.id for card in self.cards})  # no one-card topic here
        self.assertEqual([placed.name for placed in model.topics_of_deck("01")], ["Lists", "Dictionaries", "Empty"])

    def test_a_one_card_topic_gets_no_layout_and_no_point(self):
        # Without 401005 the scripted Dictionaries has one card: under leave-one-out it has no centroid, so 301003
        # goes to Lists, and the card between the two topics (101004) then sits nearest to Dictionaries, alone
        cards = [card for card in self.cards if card.id != "401005"]
        model = model_support.course_model(cards)
        placed = model.topics[("01", "02")]
        self.assertEqual(model.topic_model.topic_of["301003"], "01")
        self.assertEqual(len(placed.ids), 1)
        self.assertEqual((placed.positions, placed.links), ({}, []))
        self.assertTrue(math.isnan(placed.overlap))
        (alone,) = placed.ids
        self.assertNotIn(alone, model.card_points)

    def test_each_topic_gets_one_connection_per_merge_within_the_topic_and_none_below_two_cards(self):
        for (deck_id, topic_id), expected in SCRIPTED.items():
            pairs = self.model.topics[(deck_id, topic_id)].links
            self.assertEqual(len(pairs), max(0, len(expected) - 1), (deck_id, topic_id))
            self.assertTrue(all(source in expected and target in expected for source, target in pairs), (deck_id, topic_id))
            self.assertEqual(len({frozenset(pair) for pair in pairs}), len(pairs))
        self.assertEqual([frozenset(pair) for pair in self.model.topics[("01", "02")].links], [frozenset(("301003", "401005"))])
        self.assertEqual(self.model.topics[("01", "03")].links, [])  # the empty topic

    def test_the_decks_are_connected_by_deck_id(self):
        self.assertEqual(len(self.model.deck_links), 2)
        for source, target in self.model.deck_links:
            self.assertIn(source, {"01", "02", "03"})
            self.assertIn(target, {"01", "02", "03"})

    def test_related_cards_come_from_the_index(self):
        related = self.model.related("102001")  # git commit
        self.assertTrue(related)
        self.assertIn(related[0][0], model_support.BETA_CARDS)
        self.assertEqual(self.model.related("unknown"), [])

    def test_the_measures_are_recorded(self):
        self.assertTrue(math.isnan(self.model.deck_trustworthiness))  # three decks are too few for k = 3
        self.assertFalse(math.isnan(self.model.deck_overlap))
        self.assertFalse(math.isnan(self.model.topics[("01", "01")].overlap))
        self.assertTrue(math.isnan(self.model.topics[("01", "03")].overlap))

    def test_the_hidden_cards_are_the_topic_models(self):
        self.assertEqual(self.model.hidden, self.model.topic_model.hidden)
        self.assertEqual(self.model.hidden, {model_support.HIDDEN})

    def test_a_deck_without_an_id_keeps_its_name(self):
        model = CourseModel(self.cards, {"Alpha": "01"}, model_support.SEEDS)
        self.assertEqual(sorted(model.deck_points), ["01", "Beta", "Gamma"])
        self.assertIn(("Beta", "01"), model.topics)
        self.assertEqual(model.topics[("Beta", "01")].links, self.model.topics[("02", "01")].links)

    def test_the_model_is_the_same_at_every_fit(self):
        again = model_support.course_model()
        self.assertEqual(again.deck_points, self.model.deck_points)
        self.assertEqual(again.card_points, self.model.card_points)
        self.assertEqual(again.deck_links, self.model.deck_links)
        self.assertEqual(again.topic_model.topic_of, self.model.topic_model.topic_of)


if __name__ == "__main__":
    unittest.main()
