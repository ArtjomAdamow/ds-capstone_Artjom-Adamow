"""The similarity pipeline: card text, neighbours (order, minimum score, top k), a stand-in vectoriser and deck
vectors, on the models' table fixture (nothing of the viewer; the real course's index is in test_course.py)."""

import model_support  # noqa: F401  (the helper's package on the path, the test course as card records)

import unittest

import numpy as np

from app.cards import CardRecord
from app.models import similar


class CardTextTest(unittest.TestCase):
    def test_the_text_is_question_answer_and_by_default_the_explanation(self):
        card = CardRecord("101001", "Alpha", "What is `x_y`?", "**Bold** answer", explain="Wikipedia's article on x.")
        self.assertEqual(similar.card_text(card, with_explain=False), "What is `x_y`? **Bold** answer")
        self.assertEqual(similar.card_text(card), "What is `x_y`? **Bold** answer Wikipedia's article on x.")

    def test_markdown_does_not_reach_the_vocabulary_and_code_names_stay_one_token(self):
        vectoriser = similar.tfidf_vectoriser(min_df=1)
        vectoriser.fit_transform(["Use `word_tokenize` and **bold** words", "word_tokenize splits bold text"])
        vocabulary = set(vectoriser.vocabulary_)
        self.assertIn("word_tokenize", vocabulary)
        self.assertNotIn("`word_tokenize`", vocabulary)
        self.assertNotIn("**bold**", vocabulary)


class NeighboursTest(unittest.TestCase):
    def setUp(self):
        self.cards = model_support.CARDS
        self.deck_of = {card.id: card.deck for card in self.cards}

    def test_every_card_has_cosine_one_with_itself(self):
        index = similar.SimilarityIndex.fit(self.cards)
        cosines = similar.cosine_similarities(index.vectors)
        self.assertTrue(np.allclose(np.diag(cosines), 1.0))
        self.assertTrue(np.allclose(cosines, cosines.T))

    def test_neighbours_come_best_first_and_from_the_same_topic(self):
        index = similar.SimilarityIndex.fit(self.cards, min_score=0.0)
        related = index.related("101001")  # the list comprehension card
        self.assertTrue(related)
        scores = [score for _, score in related]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(self.deck_of[related[0][0]], "Alpha")
        self.assertNotIn("101001", [other for other, _ in related])
        self.assertLessEqual(len(related), similar.TOP_K)

    def test_the_minimum_score_drops_weak_neighbours(self):
        scores = np.array([[1.0, 0.9, 0.3, 0.05], [0.9, 1.0, 0.2, 0.0], [0.3, 0.2, 1.0, 0.5], [0.05, 0.0, 0.5, 1.0]])
        ids = ["a", "b", "c", "d"]
        self.assertEqual(similar.top_neighbours(ids, scores, k=3, min_score=0.0)["a"], [("b", 0.9), ("c", 0.3), ("d", 0.05)])
        self.assertEqual(similar.top_neighbours(ids, scores, k=3, min_score=0.1)["a"], [("b", 0.9), ("c", 0.3)])
        self.assertEqual(similar.top_neighbours(ids, scores, k=3, min_score=0.95)["a"], [])

    def test_top_k_limits_the_list_and_ties_keep_the_id_order(self):
        scores = np.array([[1.0, 0.5, 0.5, 0.5], [0.5, 1.0, 0.1, 0.1], [0.5, 0.1, 1.0, 0.1], [0.5, 0.1, 0.1, 1.0]])
        neighbours = similar.top_neighbours(["a", "b", "c", "d"], scores, k=2, min_score=0.0)
        self.assertEqual(neighbours["a"], [("b", 0.5), ("c", 0.5)])
        self.assertEqual(neighbours["b"], [("a", 0.5), ("c", 0.1)])

    def test_a_single_card_has_no_neighbours(self):
        index = similar.SimilarityIndex.fit(self.cards[:1])
        self.assertEqual(index.related(self.cards[0].id), [])

    def test_a_stand_in_vectoriser_plugs_in(self):
        decks = sorted({card.deck for card in self.cards})

        class OneHotByDeck:
            def fit_transform(self, texts):
                rows = np.zeros((len(texts), len(decks)))
                for row, card in enumerate(self.cards):
                    rows[row, decks.index(card.deck)] = 3.0  # not unit length: the pipeline normalises
                return rows

        vectoriser = OneHotByDeck()
        vectoriser.cards = self.cards
        index = similar.SimilarityIndex.fit(self.cards, vectoriser, min_score=0.5)
        self.assertTrue(np.allclose(np.asarray(index.vectors.sum(axis=1)).ravel(), 1.0))
        for card in self.cards:
            others = [self.deck_of[other] for other, _ in index.related(card.id)]
            self.assertEqual(set(others), {card.deck})

    def test_deck_vectors_are_unit_means_in_first_seen_order(self):
        vectors = similar.unit_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 2.0]]))
        names, means = similar.deck_vectors(["Beta", "Alpha", "Beta", "Alpha"], vectors)
        self.assertEqual(names, ["Beta", "Alpha"])
        self.assertTrue(np.allclose(means, [[1.0, 0.0], [0.0, 1.0]]))
        self.assertTrue(np.allclose(np.linalg.norm(means, axis=1), 1.0))

    def test_rows_of_deck(self):
        index = similar.SimilarityIndex.fit(self.cards)
        self.assertEqual([index.ids[row] for row in index.rows_of_deck("Beta")], model_support.BETA_CARDS)


if __name__ == "__main__":
    unittest.main()
