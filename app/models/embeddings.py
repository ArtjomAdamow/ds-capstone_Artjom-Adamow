"""S2: a sentence encoder as a vectoriser of the similarity pipeline, the hybrid with TF-IDF, and the cache of the
embeddings (docs/models/similarity.md, "S2").

    card text ──┬→ TF-IDF → unit rows · √(1/(1 − m₁)) ─┐
                │                                       ├→ the blocks side by side → SimilarityIndex (unit rows, cosine)
                └→ encoder → unit rows · √(1/(1 − m₂)) ─┘
    m₁, m₂: each half's own minimum, the 10th percentile of its fifth neighbour's cosine over the cards

* `SentenceEncoder` encodes texts with a sentence-transformers model (nothing is fitted: fit_transform and
  transform both encode), unit length from the model, in batches, on the device `resolve_device` picks: APP_DEVICE,
  the GPU where torch sees one, else the CPU; the library is imported when the model is loaded, so this module
  imports without torch. The embeddings of a course are cached (`EmbeddingCache`): one .npy
  per model and course, keyed by the model's name and a hash of the texts in order, in a folder the launcher names.
* `HybridVectoriser` fits two vectorisers on the same texts and returns their unit rows side by side, each block
  scaled by √(1/(1 − m)) with the block's own minimum m, so the index's cosine of two cards is the weighted sum
  (w₁ s₁ + w₂ s₂) / (w₁ + w₂) with wᵢ = 1/(1 − mᵢ): the ranking of the sum of the two scores each normalised at its
  own minimum. The index, C1, T1 and D1 read the rows as they read TF-IDF's.
* `app_vectoriser` is the vectoriser the app runs: APP_VECTORISER, set by the hand check's verdict, or TF-IDF with
  one printed line when the library or the model is not there. The launchers and evaluate.py pass it explicitly;
  CourseModel(vectoriser=None) stays TF-IDF, so the fixture tests run without torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import sparse

from . import similar

MINILM = "all-MiniLM-L6-v2"  # 384 dimensions, 91 MB: the first model
MPNET = "all-mpnet-base-v2"  # 768 dimensions, 438 MB: the quality check
TFIDF = "tfidf"
# The names app_vectoriser knows: TF-IDF, an encoder alone, the hybrid of TF-IDF's chosen settings and an encoder
ENCODERS = {"minilm": MINILM, "mpnet": MPNET}
VECTORISERS = (TFIDF,) + tuple(ENCODERS) + tuple(f"hybrid-{name}" for name in ENCODERS)
# The vectoriser the app runs, set by the verdict of the hand check (similarity.md, "S2", "The verdict rule" and "The
# numbers"): the hybrid of TF-IDF's chosen settings and MiniLM, +0.07 of hand p@5 over S1 on decks 17 and 24 with an
# interval that excludes zero, the first neighbour kept, deck 18 not reversing it
APP_VECTORISER = "hybrid-minilm"
BATCH_SIZE = 64
CACHE_FOLDER = "embeddings"  # the cache's folder name under the launchers' cache folder (data/cache/)
LIBRARY = "sentence_transformers"
FETCH_COMMAND = 'python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer({name!r})"'
# The device of S2's encoder and D2's model (gen3.md): "auto", the GPU where torch sees one (the embeddings-gpu
# extra), else the CPU; "cpu" or "cuda" to choose. fp32 on the GPU gives the CPU's vectors (cosine 1.000000, the
# same five neighbours), so the cache's key has no device.
DEVICE_VARIABLE = "APP_DEVICE"
DEVICES = ("auto", "cpu", "cuda")


def encoder_available(find_spec: Callable = importlib.util.find_spec) -> bool:
    """Whether sentence-transformers imports here (nothing is imported)."""
    try:
        return find_spec(LIBRARY) is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """Whether torch sees a CUDA GPU (torch imported here; False without it or with its CPU build)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(setting: Optional[str] = None, cuda: Optional[Callable[[], bool]] = None) -> str:
    """The device the models run on: `setting`, else APP_DEVICE, else "auto"; auto, and "cuda" without a GPU, is the
    CPU unless `cuda()` (cuda_available, looked up at the call) says a GPU is there."""
    setting = (setting or os.environ.get(DEVICE_VARIABLE) or "auto").strip().lower()
    if setting not in DEVICES:
        raise ValueError(f"unknown device {setting!r} in {DEVICE_VARIABLE}; one of {', '.join(DEVICES)}")
    if setting == "cpu":
        return "cpu"
    return "cuda" if (cuda or cuda_available)() else "cpu"


def slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()


class EmbeddingCache:
    """The embeddings of a course, one .npy per model and course in `folder`: the file is named by the model and a
    hash of the texts in order (a changed card, a reordered course or another model misses it); writing a model's
    file removes the model's older ones. Nothing of it is state: a missing file costs the encoding once."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    @staticmethod
    def key(texts: Sequence[str]) -> str:
        digest = hashlib.blake2s(digest_size=8)
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x1f")
        return digest.hexdigest()

    def path(self, model_name: str, texts: Sequence[str]) -> Path:
        return self.folder / f"{slug(model_name)}-{self.key(texts)}.npy"

    def read(self, model_name: str, texts: Sequence[str]) -> Optional[np.ndarray]:
        path = self.path(model_name, texts)
        if not path.is_file():
            return None
        try:
            vectors = np.load(path)
        except (OSError, ValueError):
            return None
        return vectors if vectors.ndim == 2 and len(vectors) == len(texts) else None

    def write(self, model_name: str, texts: Sequence[str], vectors: np.ndarray) -> Path:
        path = self.path(model_name, texts)
        self.folder.mkdir(parents=True, exist_ok=True)
        for older in self.folder.glob(f"{slug(model_name)}-*.npy"):
            if older != path:
                older.unlink(missing_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(vectors, dtype=np.float32))
        os.replace(temporary, path)
        return path


class SentenceEncoder:
    """A sentence-transformers model as a vectoriser: fit_transform and transform both encode the texts (unit length
    from the model, in batches on `device`, resolve_device's when None) and return a dense array, one row per text.
    `cache` keeps a course's embeddings between starts; `loader(model_name, device)` makes the model
    (sentence_transformers.SentenceTransformer by default)."""

    def __init__(self, model_name: str = MINILM, batch_size: int = BATCH_SIZE, cache: Optional[EmbeddingCache] = None,
                 loader: Optional[Callable] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = cache
        self._loader = loader
        self._model = None
        self.device = device  # resolved at the load: "cpu" or "cuda"
        self.seconds = 0.0  # the last encoding's time (0 for a cache hit)
        self.cached = False  # whether the last fit_transform came from the cache
        self.dimensions: Optional[int] = None

    @property
    def name(self) -> str:
        return self.model_name

    def load(self):
        """The model, loaded once (the library imported here, so the package imports without torch)."""
        if self._model is None:
            self.device = resolve_device(self.device)
            if self._loader is not None:
                self._model = self._loader(self.model_name, self.device)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self.load()
        started = time.perf_counter()
        vectors = model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
        self.seconds = time.perf_counter() - started
        return np.asarray(vectors, dtype=np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """The course's embeddings: from the cache where it holds them, else encoded and cached."""
        texts = list(texts)
        vectors = self.cache.read(self.model_name, texts) if self.cache is not None else None
        self.cached = vectors is not None
        if vectors is None:
            vectors = self.encode(texts) if texts else np.zeros((0, self.dimensions or 0), dtype=np.float32)
            if self.cache is not None and texts:
                self.cache.write(self.model_name, texts, vectors)
        else:
            self.seconds = 0.0
        self.dimensions = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[1] else self.dimensions
        return vectors

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Other texts (a request): encoded, never cached."""
        texts = list(texts)
        return self.encode(texts) if texts else np.zeros((0, self.dimensions or 0), dtype=np.float32)


class HybridVectoriser:
    """Two vectorisers on the same texts, their unit rows side by side, each block scaled by √(1/(1 − m)) with the
    block's own minimum m (the 10th percentile of its fifth neighbour's cosine), so the cosine of the joined rows
    ranks pairs as the sum of the two scores each normalised at its own minimum."""

    def __init__(self, first, second, k: int = similar.TOP_K, percentile: float = similar.OWN_PERCENTILE) -> None:
        self.first = first
        self.second = second
        self.k = k
        self.percentile = percentile
        self.minima: tuple[float, float] = (0.0, 0.0)
        self.weights: tuple[float, float] = (1.0, 1.0)

    @property
    def halves(self):
        return (self.first, self.second)

    @property
    def vocabulary_(self):
        """The TF-IDF half's vocabulary (the readouts and the evaluation's params read it); None without one."""
        return getattr(self.first, "vocabulary_", None)

    def _blocks(self, texts: Sequence[str], fit: bool) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        texts = list(texts)
        first = self.first.fit_transform(texts) if fit else self.first.transform(texts)
        second = self.second.fit_transform(texts) if fit else self.second.transform(texts)
        return similar.unit_rows(first), similar.unit_rows(second)

    def fit_transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        a, b = self._blocks(texts, fit=True)
        self.minima = (
            similar.own_minimum(similar.cosine_similarities(a), self.k, self.percentile),
            similar.own_minimum(similar.cosine_similarities(b), self.k, self.percentile),
        )
        self.weights = tuple(1.0 / max(1.0 - minimum, 1e-9) for minimum in self.minima)
        return self._join(a, b)

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        a, b = self._blocks(texts, fit=False)
        return self._join(a, b)

    def _join(self, a: sparse.csr_matrix, b: sparse.csr_matrix) -> sparse.csr_matrix:
        scaled = [block * float(np.sqrt(weight)) for block, weight in zip((a, b), self.weights)]
        return sparse.hstack(scaled, format="csr")


def app_vectoriser(cache_dir: Optional[Path] = None, name: Optional[str] = None, output: Callable = print,
                   loader: Optional[Callable] = None):
    """The vectoriser the app runs (`name`, APP_VECTORISER by default): TF-IDF's chosen settings, an encoder, or the
    hybrid of both, with the course's embeddings cached in `cache_dir`. Without the library or the model it is
    TF-IDF, with one line that says so and how to get the encoder."""
    name = name or APP_VECTORISER
    if name not in VECTORISERS:
        raise ValueError(f"unknown vectoriser {name!r}; one of {', '.join(VECTORISERS)}")
    if name == TFIDF:
        return similar.tfidf_vectoriser()
    encoder_name = ENCODERS[name.removeprefix("hybrid-")]
    if loader is None and not encoder_available():
        output(f"S2's encoder {encoder_name} needs sentence-transformers (the embeddings group); the index runs on TF-IDF.")
        return similar.tfidf_vectoriser()
    cache = EmbeddingCache(Path(cache_dir) / CACHE_FOLDER) if cache_dir is not None else None
    encoder = SentenceEncoder(encoder_name, cache=cache, loader=loader)
    try:
        encoder.load()
    except Exception as error:  # noqa: BLE001  (a missing model, an offline hub: the app still starts)
        output(f"S2's encoder {encoder_name} could not be loaded ({error.__class__.__name__}); the index runs on TF-IDF. "
               f"Fetch it once with: {FETCH_COMMAND.format(name=encoder_name)}")
        return similar.tfidf_vectoriser()
    if name.startswith("hybrid-"):
        return HybridVectoriser(similar.tfidf_vectoriser(), encoder)
    return encoder


def describe(vectoriser) -> str:
    """One line naming a vectoriser for the readouts: "TF-IDF", "MiniLM (384 dimensions)", "hybrid: TF-IDF + mpnet"."""
    if isinstance(vectoriser, HybridVectoriser):
        return f"hybrid: {describe(vectoriser.first)} + {describe(vectoriser.second)}"
    if isinstance(vectoriser, SentenceEncoder):
        short = {MINILM: "MiniLM", MPNET: "mpnet"}.get(vectoriser.model_name, vectoriser.model_name)
        return f"{short} ({vectoriser.dimensions} dimensions)" if vectoriser.dimensions else short
    return "TF-IDF"
