"""The similarity pipeline (no Dash, no state, nothing of the viewer): card text, a swappable vectoriser, unit
vectors, neighbours and deck vectors.

    card text → vectoriser (TF-IDF by default) → unit rows ─┬→ neighbours: cosine = dot product, top 5 above MIN_SCORE
                                                           ├→ deck vectors: the mean of a deck's unit rows, unit length again
                                                           └→ positions and connections (models.layout, models.clustering)

The pipeline is fitted once on all cards of the course (SimilarityIndex.fit), so term weights and vectors are the
same in every view. A vectoriser is anything with fit_transform(texts) that returns one row per text, dense or
sparse: the TF-IDF baseline, a scikit-learn Pipeline (TF-IDF plus TruncatedSVD) or a sentence embedding wrapper
plug in at the same place.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..cards import Card

TOP_K = 5
# A neighbour below this cosine is dropped: a card without a good match gets fewer (or no) neighbours rather than
# weak ones. Chosen from the score distribution of the course: about the 10th percentile of the fifth neighbour's
# score (docs/draft/helper/concept.md, "Baseline numbers").
MIN_SCORE = 0.1
# The chosen settings of the baseline (the offline evaluation, docs/draft/helper/concept.md): unigrams plus bigrams,
# terms in at least two cards, sublinear term frequency, and the explanation in the text.
NGRAM_RANGE = (1, 2)
MIN_DF = 2
SUBLINEAR_TF = True
WITH_EXPLAIN = True


class Vectoriser(Protocol):
    def fit_transform(self, texts: Sequence[str]): ...


def card_text(card: Card, with_explain: bool = WITH_EXPLAIN) -> str:
    """The document of a card: its question, its answer (the cleaned feedback) and, by default, its explanation,
    which names the card's concept (mostly as the title of its web link).

    The markdown left by the parser (backticks, bold) does not matter: the default token pattern keeps only letters,
    digits and underscores, so code names such as word_tokenize stay one token.
    """
    parts = [card.question, card.answer]
    if with_explain and card.explain:
        parts.append(card.explain)
    return " ".join(parts)


def tfidf_vectoriser(
    ngram_range=NGRAM_RANGE, min_df: int = MIN_DF, sublinear_tf: bool = SUBLINEAR_TF, **options
) -> TfidfVectorizer:
    """The baseline vectoriser: English stop words, unigrams plus bigrams, terms in at least two cards, and a
    sublinear term frequency (1 + log tf), so a term repeated in a long answer does not dominate."""
    return TfidfVectorizer(
        stop_words="english", ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf, **options
    )


def unit_rows(vectors) -> sparse.csr_matrix:
    """Every row scaled to length 1 (a row of zeros stays zero), as a CSR matrix whatever the vectoriser returned."""
    matrix = vectors.tocsr() if sparse.issparse(vectors) else sparse.csr_matrix(np.asarray(vectors, dtype=float))
    return normalize(matrix, norm="l2", copy=True)


def cosine_similarities(vectors: sparse.csr_matrix) -> np.ndarray:
    """All pairwise cosines of unit rows: the dense n × n matrix of dot products (1,064² values fit in memory)."""
    return np.asarray((vectors @ vectors.T).todense(), dtype=float)


def top_neighbours(
    ids: Sequence[str], similarities: np.ndarray, k: int = TOP_K, min_score: float = MIN_SCORE
) -> dict[str, list[tuple[str, float]]]:
    """For every row its k most similar other rows with a cosine of at least min_score, best first.

    Ties keep the order of the ids, so the result is deterministic.
    """
    scores = np.array(similarities, dtype=float, copy=True)
    np.fill_diagonal(scores, -np.inf)
    limit = min(k, len(ids) - 1)
    neighbours: dict[str, list[tuple[str, float]]] = {}
    for row, card_id in enumerate(ids):
        if limit <= 0:
            neighbours[card_id] = []
            continue
        values = scores[row]
        candidates = np.argpartition(-values, limit - 1)[:limit]
        ordered = sorted(candidates, key=lambda column: (-values[column], column))
        neighbours[card_id] = [(ids[column], float(values[column])) for column in ordered if values[column] >= min_score]
    return neighbours


def deck_vectors(decks: Sequence[str], vectors: sparse.csr_matrix) -> tuple[list[str], np.ndarray]:
    """One unit vector per deck, in the order the decks first appear: the mean of the deck's unit rows, scaled to
    length 1 again. It is the deck's centroid direction in the full vector space."""
    names = list(dict.fromkeys(decks))
    rows = []
    for name in names:
        members = [row for row, deck in enumerate(decks) if deck == name]
        mean = np.asarray(vectors[members].mean(axis=0), dtype=float).ravel()
        length = float(np.linalg.norm(mean))
        rows.append(mean / length if length > 0 else mean)
    return names, np.vstack(rows) if rows else np.zeros((0, vectors.shape[1]))


@dataclass
class SimilarityIndex:
    """The fitted pipeline: one unit row per card, and every card's neighbours."""

    ids: list[str]
    decks: list[str]  # the deck name of every row
    vectors: sparse.csr_matrix  # unit rows, in the order of ids
    neighbours: dict[str, list[tuple[str, float]]]
    vectoriser: object

    @classmethod
    def fit(
        cls,
        cards: Sequence[Card],
        vectoriser: Optional[Vectoriser] = None,
        k: int = TOP_K,
        min_score: float = MIN_SCORE,
        with_explain: bool = WITH_EXPLAIN,
    ) -> "SimilarityIndex":
        """Fit the vectoriser on the texts of all cards and keep the unit vectors and the neighbours."""
        texts = [card_text(card, with_explain) for card in cards]
        if vectoriser is None:
            # The baseline; a corpus smaller than MIN_DF (tests, tiny courses) keeps every term
            vectoriser = tfidf_vectoriser(min_df=min(MIN_DF, max(1, len(texts))))
        vectors = unit_rows(vectoriser.fit_transform(texts)) if texts else sparse.csr_matrix((0, 0))
        ids = [card.id for card in cards]
        neighbours = top_neighbours(ids, cosine_similarities(vectors), k, min_score) if ids else {}
        return cls(ids=ids, decks=[card.deck for card in cards], vectors=vectors, neighbours=neighbours, vectoriser=vectoriser)

    def row(self, card_id: str) -> int:
        return self.ids.index(card_id)

    def related(self, card_id: str) -> list[tuple[str, float]]:
        """The neighbours of a card, best first; empty for an unknown card."""
        return list(self.neighbours.get(card_id, []))

    def rows_of_deck(self, deck: str) -> list[int]:
        return [row for row, name in enumerate(self.decks) if name == deck]
