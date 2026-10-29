"""The layout step (no Dash, no state, nothing of the viewer): points of a vector space onto the unit sphere in 3D.

* Start: the first three principal directions of the centred vectors (an SVD), every row scaled to length 1, so
  the points lie on the unit sphere. Rows that end up at the origin, and rows that coincide, are told apart by a
  tiny deterministic offset from an even spread of points (fibonacci_sphere).
* Refine: a force layout on the sphere. Each point is pulled toward its top-k neighbours in the vector space,
  weighted by their cosine, and pushed away from all other points (a 1/distance kernel, scaled by the number of
  points), and after each step is scaled back to length 1. The step size cools down linearly to zero.
* Measure: scikit-learn's trustworthiness (neighbours in the vector space by cosine), and the share of each point's
  top-k neighbours in the vector space that are also among its k nearest points in 3D.

Nothing here draws random numbers: the same vectors give the same layout.
"""

import numpy as np
from scipy import sparse
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

DIMENSIONS = 3
ITERATIONS = 300
STEP = 0.1  # the first step size; it cools down linearly to zero
REPULSION = 1.0  # strength of the push between all points, relative to the pull toward neighbours
EPSILON = 1e-4  # keeps the push finite for coinciding points
TIE_OFFSET = 1e-3  # offset that tells coinciding start points apart


def as_dense(vectors) -> np.ndarray:
    """A dense float matrix, whatever the vectoriser returned."""
    if sparse.issparse(vectors):
        return np.asarray(vectors.todense(), dtype=float)
    return np.asarray(vectors, dtype=float)


def fibonacci_sphere(count: int) -> np.ndarray:
    """`count` points spread evenly over the unit sphere, in a fixed order."""
    points = np.zeros((count, DIMENSIONS))
    if count == 0:
        return points
    golden = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    angle = golden * index
    points[:, 0] = radius * np.cos(angle)
    points[:, 1] = radius * np.sin(angle)
    points[:, 2] = z
    return points


def to_sphere(points: np.ndarray) -> np.ndarray:
    """Every row scaled to length 1. A row at the origin gets an even-spread direction of its own index."""
    points = np.array(points, dtype=float, copy=True)
    lengths = np.linalg.norm(points, axis=1)
    missing = lengths <= 1e-12
    if missing.any():
        points[missing] = fibonacci_sphere(len(points))[missing]
        lengths = np.linalg.norm(points, axis=1)
    return points / lengths[:, None]


def svd_start(vectors, centre: bool = True) -> np.ndarray:
    """The deterministic start: the first three principal directions on the unit sphere, with a tiny offset per
    point so that coinciding points can be pushed apart."""
    matrix = as_dense(vectors)
    count = len(matrix)
    if count == 0:
        return np.zeros((0, DIMENSIONS))
    centred = matrix - matrix.mean(axis=0) if centre else matrix
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    components = min(DIMENSIONS, singular.size)
    points = np.zeros((count, DIMENSIONS))
    points[:, :components] = left[:, :components] * singular[:components]
    # The sign of a singular vector is arbitrary: make the largest entry of each component positive
    for column in range(components):
        largest = np.argmax(np.abs(points[:, column]))
        if points[largest, column] < 0:
            points[:, column] *= -1
    points += TIE_OFFSET * fibonacci_sphere(count)
    return to_sphere(points)


def neighbour_weights(vectors, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every row's top-k neighbours by cosine (as column indices) with their cosines as weights (clipped at 0)."""
    matrix = as_dense(vectors)
    count = len(matrix)
    limit = max(0, min(k, count - 1))
    if limit == 0:
        return np.zeros((count, 0), dtype=int), np.zeros((count, 0))
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    units = np.divide(matrix, lengths, out=np.zeros_like(matrix), where=lengths > 0)
    cosines = units @ units.T
    np.fill_diagonal(cosines, -np.inf)
    index = np.argsort(-cosines, axis=1, kind="stable")[:, :limit]
    weights = np.clip(np.take_along_axis(cosines, index, axis=1), 0.0, None)
    return index, weights


def force_layout(
    points: np.ndarray,
    index: np.ndarray,
    weights: np.ndarray,
    iterations: int = ITERATIONS,
    step: float = STEP,
    repulsion: float = REPULSION,
) -> np.ndarray:
    """Refine points on the unit sphere: pull toward the weighted neighbours, push away from everyone, cool down."""
    points = to_sphere(points)
    count = len(points)
    if count < 2 or iterations <= 0:
        return points
    for iteration in range(iterations):
        rate = step * (1.0 - iteration / iterations)
        if index.shape[1]:
            pull = (weights[:, :, None] * (points[index] - points[:, None, :])).sum(axis=1)
        else:
            pull = 0.0
        difference = points[:, None, :] - points[None, :, :]
        squared = (difference * difference).sum(axis=2) + EPSILON
        np.fill_diagonal(squared, np.inf)
        push = (difference / squared[:, :, None]).sum(axis=1) * (repulsion / count)
        points = to_sphere(points + rate * (pull + push))
    return points


def sphere_layout(vectors, k: int, **options) -> tuple[np.ndarray, np.ndarray]:
    """The whole layout step: the SVD start refined by the force layout. Returns (points, start)."""
    matrix = as_dense(vectors)
    start = svd_start(matrix)
    index, weights = neighbour_weights(matrix, k)
    return force_layout(start, index, weights, **options), start


def nearest_in_3d(points: np.ndarray, k: int) -> np.ndarray:
    """Every point's k nearest other points (as row indices) by euclidean distance."""
    count = len(points)
    limit = max(0, min(k, count - 1))
    if limit == 0:
        return np.zeros((count, 0), dtype=int)
    difference = points[:, None, :] - points[None, :, :]
    distances = (difference * difference).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :limit]


def neighbour_overlap(vectors, points: np.ndarray, k: int) -> float:
    """The mean share of a point's top-k vector-space neighbours that are also among its k nearest points in 3D.
    NaN with fewer than two points."""
    index, _ = neighbour_weights(vectors, k)
    nearest = nearest_in_3d(np.asarray(points, dtype=float), k)
    if index.shape[1] == 0:
        return float("nan")
    shares = [len(set(index[row]) & set(nearest[row])) / index.shape[1] for row in range(len(index))]
    return float(np.mean(shares))


def trustworthiness_score(vectors, points: np.ndarray, k: int) -> float:
    """scikit-learn's trustworthiness of the 3D points for the vector space (cosine); NaN when there are too few
    points for k (it needs k below half of the points)."""
    matrix = as_dense(vectors)
    count = len(matrix)
    if count < 2 or k < 1 or k >= count / 2:
        return float("nan")
    return float(sklearn_trustworthiness(matrix, np.asarray(points, dtype=float), n_neighbors=k, metric="cosine"))
