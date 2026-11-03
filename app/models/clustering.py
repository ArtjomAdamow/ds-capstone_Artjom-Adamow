"""Connections from a hierarchical clustering (no Dash, no state, nothing of the viewer).

scipy's average-linkage clustering on the cosine distances of a set of unit vectors gives a tree of merges. Each
merge becomes one connection between the two closest members of the two merged clusters, in merge order. The
viewer's dendrogram replays stored connections in that order (project.services.dendrogram.build_forest): a connection
joins the two trees its ends belong to, which at that point are exactly the two clusters scipy merged. So the
connections give the same tree as scipy's, with the merge ranks as heights, and the project needs no scipy itself.

The cut-off preset (preset_rank): where a cut of those connections leaves parts small enough to read. The glue's
preset of the viewer's cut-off line and the standalone page both measure it over the model's pairs alone.
"""

from typing import Iterable, Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .layout import as_dense

METHOD = "average"


def cosine_distances(vectors) -> np.ndarray:
    """The dense matrix of cosine distances (1 - cosine) between the rows; a row of zeros is at distance 1 from
    everything. Symmetric, zero on the diagonal, values in [0, 2]."""
    matrix = as_dense(vectors)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    units = np.divide(matrix, lengths, out=np.zeros_like(matrix), where=lengths > 0)
    distances = np.clip(1.0 - units @ units.T, 0.0, 2.0)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    return distances


def merge_links(ids: Sequence[str], vectors, method: str = METHOD) -> list[tuple[str, str]]:
    """One connection per merge of the clustering, in merge order: the closest pair of the two merged clusters.
    Ties go to the lowest row indices, so the result is deterministic. Fewer than two ids give no connection."""
    ids = list(ids)
    count = len(ids)
    if count < 2:
        return []
    distances = cosine_distances(vectors)
    merges = linkage(squareform(distances, checks=False), method=method)
    clusters: dict[int, list[int]] = {row: [row] for row in range(count)}
    links: list[tuple[str, str]] = []
    for step, (left, right, _distance, _size) in enumerate(merges):
        first, second = clusters.pop(int(left)), clusters.pop(int(right))
        block = distances[np.ix_(first, second)]
        row, column = np.unravel_index(int(np.argmin(block)), block.shape)
        links.append((ids[first[row]], ids[second[column]]))
        clusters[count + step] = first + second
    return links


def preset_rank(pairs: Iterable[tuple[str, str]], size: int) -> int:
    """The last merge rank whose parts hold at most `size` dots: the pairs, in merge order, join their parts one by
    one (a union-find over their ends) until the next would make a part larger. A pair within one part (a cycle) is
    no merge and takes no rank; the model's pairs, a tree or a part of one, have none. 0 when the first merge is too
    large already or there is none."""
    parent: dict[str, str] = {}
    members: dict[str, int] = {}

    def find(dot: str) -> str:
        parent.setdefault(dot, dot)
        members.setdefault(dot, 1)
        while parent[dot] != dot:
            parent[dot] = parent[parent[dot]]
            dot = parent[dot]
        return dot

    rank = 0
    for source, target in pairs:
        first, second = find(source), find(target)
        if first == second:
            continue
        if members[first] + members[second] > size:
            break
        parent[second] = first
        members[first] += members.pop(second)
        rank += 1
    return rank
