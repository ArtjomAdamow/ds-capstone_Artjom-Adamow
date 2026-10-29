"""The layout step: the even spread, the SVD start, the force layout on the sphere, and the two measures (nothing
of the viewer)."""

import model_support  # noqa: F401  (the helper's package on the path)

import unittest

import numpy as np

from app.models import layout


def clustered_vectors(groups: int = 3, size: int = 5, dimensions: int = 12) -> np.ndarray:
    """Points in a few tight groups, built without randomness: each group around its own axis."""
    rows = []
    for group in range(groups):
        for member in range(size):
            vector = np.zeros(dimensions)
            vector[group] = 1.0
            vector[groups + (group * size + member) % (dimensions - groups)] = 0.35
            vector[groups + (member + 1) % (dimensions - groups)] = 0.15
            rows.append(vector)
    return np.array(rows)


def norms(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1)


class FibonacciSphereTest(unittest.TestCase):
    def test_points_lie_on_the_sphere_and_are_spread(self):
        points = layout.fibonacci_sphere(50)
        self.assertTrue(np.allclose(norms(points), 1.0))
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 0.2)

    def test_the_empty_case(self):
        self.assertEqual(layout.fibonacci_sphere(0).shape, (0, 3))


class SvdStartTest(unittest.TestCase):
    def test_the_start_is_on_the_sphere_and_deterministic(self):
        vectors = clustered_vectors()
        first = layout.svd_start(vectors)
        second = layout.svd_start(vectors)
        self.assertEqual(first.shape, (15, 3))
        self.assertTrue(np.allclose(norms(first), 1.0))
        self.assertTrue(np.array_equal(first, second))

    def test_coinciding_vectors_are_told_apart(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        start = layout.svd_start(vectors)
        self.assertTrue(np.allclose(norms(start), 1.0))
        self.assertGreater(np.linalg.norm(start[0] - start[1]), 0.0)

    def test_a_single_vector_gets_a_direction(self):
        start = layout.svd_start(np.array([[0.3, 0.4]]))
        self.assertEqual(start.shape, (1, 3))
        self.assertAlmostEqual(float(norms(start)[0]), 1.0)

    def test_sparse_input_is_accepted(self):
        from scipy import sparse

        dense = clustered_vectors()
        self.assertTrue(np.allclose(layout.svd_start(sparse.csr_matrix(dense)), layout.svd_start(dense)))


class ForceLayoutTest(unittest.TestCase):
    def test_neighbour_weights_are_the_top_k_cosines(self):
        vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
        index, weights = layout.neighbour_weights(vectors, 2)
        self.assertEqual(list(index[0]), [1, 2])
        self.assertAlmostEqual(float(weights[0, 0]), 0.9 / np.hypot(0.9, 0.1))
        self.assertEqual(float(weights[0, 1]), 0.0)  # orthogonal
        self.assertEqual(index.shape, (4, 2))
        self.assertEqual(layout.neighbour_weights(vectors[:1], 2)[0].shape, (1, 0))

    def test_the_layout_keeps_the_points_on_the_sphere_and_brings_neighbours_together(self):
        vectors = clustered_vectors()
        points, start = layout.sphere_layout(vectors, k=4)
        self.assertTrue(np.allclose(norms(points), 1.0))
        self.assertGreaterEqual(layout.neighbour_overlap(vectors, points, 4), 0.9)
        self.assertGreaterEqual(layout.neighbour_overlap(vectors, points, 4), layout.neighbour_overlap(vectors, start, 4))
        # The three groups end up apart: members are closer to their own group than to the others
        for group in range(3):
            own = points[group * 5:(group + 1) * 5]
            others = np.delete(points, slice(group * 5, (group + 1) * 5), axis=0)
            own_spread = np.linalg.norm(own[:, None, :] - own[None, :, :], axis=2).max()
            gap = np.linalg.norm(own[:, None, :] - others[None, :, :], axis=2).min()
            self.assertLess(own_spread, gap * 2.5)

    def test_the_layout_is_deterministic(self):
        vectors = clustered_vectors()
        first, _ = layout.sphere_layout(vectors, k=3)
        second, _ = layout.sphere_layout(vectors, k=3)
        self.assertTrue(np.array_equal(first, second))

    def test_tiny_inputs(self):
        one, _ = layout.sphere_layout(np.array([[1.0, 2.0]]), k=3)
        self.assertEqual(one.shape, (1, 3))
        two, _ = layout.sphere_layout(np.array([[1.0, 0.0], [0.0, 1.0]]), k=3)
        self.assertTrue(np.allclose(norms(two), 1.0))
        self.assertGreater(np.linalg.norm(two[0] - two[1]), 0.5)
        self.assertEqual(layout.sphere_layout(np.zeros((0, 4)), k=3)[0].shape, (0, 3))


class MeasuresTest(unittest.TestCase):
    def test_overlap_is_one_when_the_map_keeps_the_neighbours(self):
        vectors = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]])
        points = vectors / norms(vectors)[:, None]
        self.assertAlmostEqual(layout.neighbour_overlap(vectors, points, 1), 1.0)

    def test_overlap_is_zero_when_the_map_swaps_them(self):
        vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        self.assertAlmostEqual(layout.neighbour_overlap(vectors, points, 1), 0.0)

    def test_trustworthiness_needs_enough_points(self):
        vectors = clustered_vectors()
        points, _ = layout.sphere_layout(vectors, k=4)
        score = layout.trustworthiness_score(vectors, points, 4)
        self.assertGreater(score, 0.8)
        self.assertTrue(np.isnan(layout.trustworthiness_score(vectors[:4], points[:4], 4)))
        self.assertTrue(np.isnan(layout.neighbour_overlap(vectors[:1], points[:1], 3)))


if __name__ == "__main__":
    unittest.main()
