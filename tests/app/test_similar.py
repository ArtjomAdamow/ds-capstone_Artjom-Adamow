"""The similarity pipeline: card text, neighbours (order, minimum score, top k), a stand-in vectoriser, deck
vectors and a request's vector (query), on the models' table fixture (nothing of the viewer; the real course's index
is in test_course.py)."""

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



class OwnMinimumTest(unittest.TestCase):
    """S2's rule for every vectoriser: the index's own minimum, the 10th percentile of the fifth neighbour's cosine."""

    def test_neighbour_scores_and_the_own_minimum(self):
        scores = np.array([[1.0, 0.9, 0.3, 0.05], [0.9, 1.0, 0.2, 0.0], [0.3, 0.2, 1.0, 0.5], [0.05, 0.0, 0.5, 1.0]])
        self.assertEqual(list(similar.neighbour_scores(scores, 1)), [0.9, 0.9, 0.5, 0.5])
        self.assertEqual(list(similar.neighbour_scores(scores, 3)), [0.05, 0.0, 0.2, 0.0])
        self.assertEqual(list(similar.neighbour_scores(scores, 7)), [0.05, 0.0, 0.2, 0.0])  # the last other card's
        self.assertEqual(similar.neighbour_scores(np.ones((1, 1)), 1).size, 0)
        self.assertAlmostEqual(similar.own_minimum(scores, k=1, percentile=50), 0.7)  # the median of the first neighbour
        self.assertEqual(similar.own_minimum(np.ones((1, 1))), similar.MIN_SCORE)
        self.assertEqual(similar.OWN_PERCENTILE, 10)

    def test_the_index_takes_its_own_minimum_with_none_and_keeps_the_minimum_it_used(self):
        fixed = similar.SimilarityIndex.fit(model_support.CARDS)
        self.assertEqual(fixed.min_score, similar.MIN_SCORE)
        own = similar.SimilarityIndex.fit(model_support.CARDS, min_score=None)
        expected = similar.own_minimum(similar.cosine_similarities(own.vectors))
        self.assertEqual(own.min_score, expected)
        self.assertTrue(all(score >= expected for items in own.neighbours.values() for _other, score in items))
        self.assertEqual(similar.SimilarityIndex.fit(model_support.CARDS[:1], min_score=None).min_score, similar.MIN_SCORE)

    def test_tfidf_transforms_another_text_into_the_indexs_space(self):
        index = similar.SimilarityIndex.fit(model_support.CARDS)
        row = similar.unit_rows(index.vectoriser.transform([similar.card_text(model_support.CARDS[0])]))
        self.assertEqual(row.shape[1], index.vectors.shape[1])
        self.assertAlmostEqual(float((row @ index.vectors[0].T).todense()[0, 0]), 1.0)


class QueryTest(unittest.TestCase):
    """P1's request vector: the fitted vectoriser's transform, at unit length, in the space of the cards' rows."""

    def setUp(self):
        self.index = similar.SimilarityIndex.fit(model_support.CARDS)

    def test_a_request_is_a_unit_vector_in_the_indexs_space(self):
        vector = self.index.query("null hypothesis")
        self.assertEqual(vector.shape, (self.index.vectors.shape[1],))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)
        cosines = self.index.vectors @ vector
        self.assertEqual(self.index.ids[int(np.argmax(cosines))], "103003")  # "What is the null hypothesis?"

    def test_a_cards_own_text_is_nearest_to_itself(self):
        for row, card in enumerate(model_support.CARDS):
            cosines = self.index.vectors @ self.index.query(similar.card_text(card))
            self.assertAlmostEqual(float(cosines[row]), 1.0)
            self.assertEqual(int(np.argmax(cosines)), row)

    def test_words_outside_the_vocabulary_give_a_zero_vector(self):
        vector = self.index.query("bake bread")
        self.assertEqual(float(np.abs(vector).sum()), 0.0)
        self.assertEqual(self.index.query("").shape, (self.index.vectors.shape[1],))

    def test_queries_transform_many_requests_at_once_as_query_does_one(self):
        texts = ["git merge", "null hypothesis", "bake bread"]
        rows = self.index.queries(texts)
        self.assertEqual(rows.shape, (3, self.index.vectors.shape[1]))
        for number, text in enumerate(texts):
            self.assertTrue(np.allclose(np.asarray(rows[number].todense())[0], self.index.query(text)))
        self.assertEqual(self.index.queries([]).shape, (0, self.index.vectors.shape[1]))

    def test_an_index_without_cards_answers_with_nothing(self):
        empty = similar.SimilarityIndex.fit([])
        self.assertEqual(empty.query("git merge").size, 0)
        self.assertEqual(empty.queries(["a", "b"]).shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
