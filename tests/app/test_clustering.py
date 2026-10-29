"""Connections from the clustering: one per merge between the closest members, in merge order (nothing of the
viewer; the proof that the viewer's dendrogram replays them to scipy's tree is in test_cutoff.py)."""

import model_support  # noqa: F401  (the helper's package on the path)

import unittest

import numpy as np

from app.models import clustering


class CosineDistancesTest(unittest.TestCase):
    def test_distances_are_one_minus_cosine_with_zero_rows_far_from_everything(self):
        distances = clustering.cosine_distances(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        self.assertTrue(np.allclose(np.diag(distances), 0.0))
        self.assertAlmostEqual(float(distances[0, 1]), 0.0)
        self.assertAlmostEqual(float(distances[0, 2]), 1.0)
        self.assertAlmostEqual(float(distances[0, 3]), 1.0)
        self.assertTrue(np.allclose(distances, distances.T))


class MergeLinksTest(unittest.TestCase):
    def test_each_merge_is_one_connection_between_the_closest_members(self):
        vectors = np.array([[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]])
        pairs = clustering.merge_links(["a", "b", "c", "d"], vectors)
        self.assertEqual(len(pairs), 3)
        self.assertEqual({frozenset(pairs[0]), frozenset(pairs[1])}, {frozenset("ab"), frozenset("cd")})
        self.assertEqual(frozenset(pairs[2]), frozenset("bd"))  # the closest pair across the two clusters

    def test_fewer_than_two_ids_give_no_connection(self):
        self.assertEqual(clustering.merge_links([], np.zeros((0, 2))), [])
        self.assertEqual(clustering.merge_links(["a"], np.array([[1.0, 0.0]])), [])

    def test_one_connection_per_merge_in_merge_order(self):
        generator = np.random.RandomState(0)
        vectors = generator.rand(12, 6)
        ids = [f"card-{index}" for index in range(12)]
        pairs = clustering.merge_links(ids, vectors)
        self.assertEqual(len(pairs), 11)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 11)
        self.assertTrue(all(source != target and source in ids and target in ids for source, target in pairs))

    def test_sparse_vectors_are_accepted_and_the_result_is_deterministic(self):
        from scipy import sparse

        vectors = np.random.RandomState(1).rand(8, 5)
        dense_pairs = clustering.merge_links(list("abcdefgh"), vectors)
        sparse_pairs = clustering.merge_links(list("abcdefgh"), sparse.csr_matrix(vectors))
        self.assertEqual(dense_pairs, sparse_pairs)
        self.assertEqual(dense_pairs, clustering.merge_links(list("abcdefgh"), vectors))


if __name__ == "__main__":
    unittest.main()
