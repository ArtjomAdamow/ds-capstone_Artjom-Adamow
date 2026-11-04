"""S2 (app.models.embeddings) on the fixtures, with a fake encoder in place of the model: the embeddings' cache, the
encoder as a vectoriser, the device, the hybrid's blocks and weights, the app's vectoriser by name, and the tests on
the real encoder, which skip without the library or the cached model, the GPU's without CUDA."""

import model_support  # noqa: F401  (the helper's package on the path, the test course as card records)

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.models import embeddings, similar

CARDS = model_support.CARDS
TEXTS = [similar.card_text(card) for card in CARDS]


class CacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = embeddings.EmbeddingCache(Path(self._tmp.name) / "embeddings")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_key_follows_the_texts_in_order_and_the_file_the_model(self):
        key = embeddings.EmbeddingCache.key(TEXTS)
        self.assertEqual(key, embeddings.EmbeddingCache.key(list(TEXTS)))
        self.assertNotEqual(key, embeddings.EmbeddingCache.key(TEXTS[::-1]))
        self.assertNotEqual(key, embeddings.EmbeddingCache.key(TEXTS[:-1] + [TEXTS[-1] + " changed"]))
        self.assertNotEqual(embeddings.EmbeddingCache.key(["ab", "c"]), embeddings.EmbeddingCache.key(["a", "bc"]))
        self.assertEqual(self.cache.path(embeddings.MINILM, TEXTS).name, f"all-minilm-l6-v2-{key}.npy")
        self.assertNotEqual(self.cache.path(embeddings.MINILM, TEXTS), self.cache.path(embeddings.MPNET, TEXTS))

    def test_a_round_trip_and_a_miss_on_other_texts_or_another_model(self):
        self.assertIsNone(self.cache.read(embeddings.MINILM, TEXTS))  # no folder yet
        vectors = np.arange(len(TEXTS) * 4, dtype=np.float32).reshape(len(TEXTS), 4)
        path = self.cache.write(embeddings.MINILM, TEXTS, vectors)
        self.assertTrue(path.is_file())
        self.assertFalse(list(self.cache.folder.glob("*.tmp")))
        np.testing.assert_array_equal(self.cache.read(embeddings.MINILM, TEXTS), vectors)
        self.assertIsNone(self.cache.read(embeddings.MPNET, TEXTS))
        self.assertIsNone(self.cache.read(embeddings.MINILM, TEXTS[:-1]))

    def test_writing_a_models_file_removes_its_older_ones_and_keeps_another_models(self):
        other = self.cache.write(embeddings.MPNET, TEXTS, np.zeros((len(TEXTS), 2), dtype=np.float32))
        older = self.cache.write(embeddings.MINILM, TEXTS[:-1], np.zeros((len(TEXTS) - 1, 2), dtype=np.float32))
        newer = self.cache.write(embeddings.MINILM, TEXTS, np.ones((len(TEXTS), 2), dtype=np.float32))
        self.assertFalse(older.exists())
        self.assertTrue(newer.exists())
        self.assertTrue(other.exists())

    def test_a_broken_or_mismatched_file_reads_as_a_miss(self):
        path = self.cache.path(embeddings.MINILM, TEXTS)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not an array")
        self.assertIsNone(self.cache.read(embeddings.MINILM, TEXTS))
        self.cache.write(embeddings.MINILM, TEXTS, np.zeros((len(TEXTS) + 1, 3), dtype=np.float32))
        self.assertIsNone(self.cache.read(embeddings.MINILM, TEXTS))  # the wrong number of rows


class EncoderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = embeddings.EmbeddingCache(Path(self._tmp.name) / "embeddings")

    def tearDown(self):
        self._tmp.cleanup()

    def test_fit_transform_encodes_once_and_the_cache_serves_the_next_start(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, cache=self.cache, loader=model_support.fake_loader)
        first = encoder.fit_transform(TEXTS)
        self.assertEqual(first.shape, (len(TEXTS), 16))
        self.assertTrue(np.allclose(np.linalg.norm(first, axis=1), 1.0))
        self.assertFalse(encoder.cached)
        self.assertGreaterEqual(encoder.seconds, 0.0)
        self.assertEqual(encoder.dimensions, 16)
        self.assertEqual(encoder.load().calls, 1)
        again = embeddings.SentenceEncoder(embeddings.MINILM, cache=self.cache, loader=model_support.fake_loader)
        second = again.fit_transform(TEXTS)
        np.testing.assert_array_equal(second, first)
        self.assertTrue(again.cached)
        self.assertEqual(again.seconds, 0.0)
        self.assertEqual(again.dimensions, 16)
        self.assertIsNone(again._model)  # the cache served the start without loading the model

    def test_transform_encodes_and_is_never_cached(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, cache=self.cache, loader=model_support.fake_loader)
        encoder.fit_transform(TEXTS)
        vectors = encoder.transform(["what does tf-idf weight"])
        self.assertEqual(vectors.shape, (1, 16))
        self.assertEqual(len(list(self.cache.folder.glob("*.npy"))), 1)  # the course's file alone
        self.assertEqual(encoder.transform([]).shape, (0, 16))

    def test_without_a_cache_every_start_encodes(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader)
        encoder.fit_transform(TEXTS)
        encoder.fit_transform(TEXTS)
        self.assertEqual(encoder.load().calls, 2)
        self.assertFalse(encoder.cached)

    def test_the_encoder_plugs_into_the_index_and_its_text_is_s1s(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader)
        index = similar.SimilarityIndex.fit(CARDS, encoder, min_score=0.0)
        self.assertEqual(index.vectors.shape, (len(CARDS), 16))
        self.assertTrue(index.related("101001"))
        self.assertTrue(all(0.0 <= score <= 1.0 + 1e-9 for items in index.neighbours.values() for _other, score in items))
        self.assertEqual(embeddings.describe(encoder), "MiniLM (16 dimensions)")


class DeviceTest(unittest.TestCase):
    def test_auto_is_the_gpu_where_there_is_one_else_the_cpu(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(embeddings.resolve_device(cuda=lambda: True), "cuda")
            self.assertEqual(embeddings.resolve_device(cuda=lambda: False), "cpu")
            self.assertEqual(embeddings.resolve_device("auto", cuda=lambda: True), "cuda")

    def test_the_setting_or_app_device_chooses_and_cuda_without_a_gpu_is_the_cpu(self):
        self.assertEqual(embeddings.resolve_device("cpu", cuda=lambda: True), "cpu")
        self.assertEqual(embeddings.resolve_device("cuda", cuda=lambda: False), "cpu")
        with mock.patch.dict("os.environ", {embeddings.DEVICE_VARIABLE: " CPU "}):
            self.assertEqual(embeddings.resolve_device(cuda=lambda: True), "cpu")
            self.assertEqual(embeddings.resolve_device("cuda", cuda=lambda: True), "cuda")  # the argument before the variable
        with mock.patch.dict("os.environ", {embeddings.DEVICE_VARIABLE: "gpu"}), self.assertRaises(ValueError):
            embeddings.resolve_device(cuda=lambda: True)

    def test_the_encoder_resolves_its_device_at_the_load_and_hands_it_to_the_loader(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader, device="cpu")
        self.assertEqual(encoder.load().device, "cpu")
        with mock.patch.object(embeddings, "cuda_available", return_value=True), \
                mock.patch.dict("os.environ", {embeddings.DEVICE_VARIABLE: "auto"}):
            encoder = embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader)
            self.assertIsNone(encoder.device)  # nothing resolved before the load
            encoder.fit_transform(TEXTS)
        self.assertEqual(encoder.device, "cuda")
        self.assertEqual(encoder.load().device, "cuda")


class HybridTest(unittest.TestCase):
    def setUp(self):
        self.tfidf = similar.tfidf_vectoriser(min_df=1)
        self.encoder = embeddings.SentenceEncoder(embeddings.MPNET, loader=model_support.fake_loader)
        self.hybrid = embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), self.encoder)
        self.joined = self.hybrid.fit_transform(TEXTS)

    def test_the_blocks_stand_side_by_side_scaled_by_their_weights(self):
        a = similar.unit_rows(self.tfidf.fit_transform(TEXTS))
        b = similar.unit_rows(embeddings.SentenceEncoder(embeddings.MPNET, loader=model_support.fake_loader).fit_transform(TEXTS))
        self.assertEqual(self.joined.shape, (len(TEXTS), a.shape[1] + 16))
        m1, m2 = self.hybrid.minima
        self.assertEqual(m1, similar.own_minimum(similar.cosine_similarities(a)))
        self.assertEqual(m2, similar.own_minimum(similar.cosine_similarities(b)))
        self.assertEqual(self.hybrid.weights, (1 / (1 - m1), 1 / (1 - m2)))
        w1, w2 = self.hybrid.weights
        joined = np.asarray(self.joined.todense())
        self.assertTrue(np.allclose(joined[:, : a.shape[1]], np.asarray(a.todense()) * np.sqrt(w1)))
        self.assertTrue(np.allclose(joined[:, a.shape[1]:], np.asarray(b.todense()) * np.sqrt(w2)))

    def test_the_indexs_cosine_is_the_weighted_sum_of_the_halves_cosines(self):
        a = similar.unit_rows(self.tfidf.fit_transform(TEXTS))
        b = similar.unit_rows(embeddings.SentenceEncoder(embeddings.MPNET, loader=model_support.fake_loader).fit_transform(TEXTS))
        w1, w2 = self.hybrid.weights
        expected = (w1 * similar.cosine_similarities(a) + w2 * similar.cosine_similarities(b)) / (w1 + w2)
        self.assertTrue(np.allclose(similar.cosine_similarities(similar.unit_rows(self.joined)), expected))
        # The ranking of the min-normalised sum: a constant apart, the same order
        m1, m2 = self.hybrid.minima
        normalised = (similar.cosine_similarities(a) - m1) / (1 - m1) + (similar.cosine_similarities(b) - m2) / (1 - m2)
        for row in range(len(TEXTS)):
            self.assertEqual(list(np.argsort(-expected[row], kind="stable")), list(np.argsort(-normalised[row], kind="stable")))

    def test_transform_uses_the_fitted_halves_and_the_stored_weights(self):
        rows = self.hybrid.transform([TEXTS[0], "git merge a branch"])
        self.assertEqual(rows.shape, (2, self.joined.shape[1]))
        self.assertTrue(np.allclose(np.asarray(rows.todense())[0], np.asarray(self.joined.todense())[0]))

    def test_the_hybrid_plugs_into_the_index_at_its_own_minimum(self):
        index = similar.SimilarityIndex.fit(CARDS, embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), self.encoder), min_score=None)
        self.assertEqual(index.min_score, similar.own_minimum(similar.cosine_similarities(index.vectors)))
        for card in CARDS:
            for other, score in index.related(card.id):
                self.assertGreaterEqual(score, index.min_score)
        self.assertEqual(embeddings.describe(index.vectoriser), "hybrid: TF-IDF + mpnet (16 dimensions)")

    def test_a_request_on_the_hybrid_or_the_encoder_is_a_unit_vector_and_a_cards_text_finds_itself(self):
        for vectoriser in (embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), self.encoder), self.encoder):
            index = similar.SimilarityIndex.fit(CARDS, vectoriser)
            vector = index.query("git merge a branch")
            self.assertEqual(vector.shape, (index.vectors.shape[1],))
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
            for row, card in enumerate(CARDS):
                cosines = index.vectors @ index.query(similar.card_text(card))
                self.assertAlmostEqual(float(cosines[row]), 1.0, places=5)


class AppVectoriserTest(unittest.TestCase):
    def test_tfidf_by_name_and_an_unknown_name_refused(self):
        self.assertIsInstance(embeddings.app_vectoriser(name=embeddings.TFIDF), type(similar.tfidf_vectoriser()))
        self.assertEqual(embeddings.APP_VECTORISER, "hybrid-minilm")  # the hand check's verdict (similarity.md, "The numbers")
        self.assertIn(embeddings.APP_VECTORISER, embeddings.VECTORISERS)
        with mock.patch.object(embeddings, "encoder_available", return_value=False):
            self.assertEqual(embeddings.describe(embeddings.app_vectoriser(output=lambda line: None)), "TF-IDF")  # without the library
        with self.assertRaises(ValueError):
            embeddings.app_vectoriser(name="word2vec")

    def test_without_the_library_tfidf_with_one_line(self):
        lines = []
        with mock.patch.object(embeddings, "encoder_available", return_value=False):
            vectoriser = embeddings.app_vectoriser(name="hybrid-minilm", output=lines.append)
        self.assertEqual(embeddings.describe(vectoriser), "TF-IDF")
        self.assertEqual(len(lines), 1)
        self.assertIn("sentence-transformers", lines[0])
        self.assertIn(embeddings.MINILM, lines[0])

    def test_a_model_that_does_not_load_gives_tfidf_and_the_fetching_command(self):
        def failing(model_name, device):
            raise OSError("offline")

        lines = []
        vectoriser = embeddings.app_vectoriser(name="minilm", output=lines.append, loader=failing)
        self.assertEqual(embeddings.describe(vectoriser), "TF-IDF")
        self.assertIn("OSError", lines[0])
        self.assertIn(f"SentenceTransformer('{embeddings.MINILM}')", lines[0])

    def test_the_encoder_and_the_hybrid_by_name_with_the_cache_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            encoder = embeddings.app_vectoriser(Path(folder), name="mpnet", loader=model_support.fake_loader)
            self.assertIsInstance(encoder, embeddings.SentenceEncoder)
            self.assertEqual(encoder.model_name, embeddings.MPNET)
            self.assertEqual(encoder.cache.folder, Path(folder) / embeddings.CACHE_FOLDER)
            hybrid = embeddings.app_vectoriser(Path(folder), name="hybrid-minilm", loader=model_support.fake_loader)
            self.assertIsInstance(hybrid, embeddings.HybridVectoriser)
            self.assertEqual(hybrid.second.model_name, embeddings.MINILM)
            self.assertIsNone(embeddings.app_vectoriser(name="minilm", loader=model_support.fake_loader).cache)
            self.assertEqual(embeddings.VECTORISERS, ("tfidf", "minilm", "mpnet", "hybrid-minilm", "hybrid-mpnet"))

    def test_the_model_fits_on_the_hybrid_as_on_tfidf(self):
        hybrid = embeddings.app_vectoriser(name="hybrid-minilm", loader=model_support.fake_loader)
        model = model_support.course_model(vectoriser=hybrid)
        self.assertEqual(set(model.card_points), set(model_support.ALL_CARDS))
        self.assertEqual(len(model.deck_points), 3)
        self.assertEqual(model.index.min_score, similar.MIN_SCORE)  # the fixture's fixed minimum
        own = model_support.CourseModel(CARDS, model_support.DECK_IDS, model_support.SEEDS, hybrid, min_score=None)
        self.assertEqual(own.index.min_score, similar.own_minimum(similar.cosine_similarities(own.index.vectors)))


def cached_model(name: str) -> bool:
    """Whether the Hugging Face cache holds the model (no download in a test)."""
    home = Path(__import__("os").environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (home / "hub" / f"models--sentence-transformers--{name}").is_dir()


@unittest.skipUnless(embeddings.encoder_available() and cached_model(embeddings.MINILM), "sentence-transformers or the cached MiniLM missing")
class RealEncoderTest(unittest.TestCase):
    def test_minilm_encodes_the_test_course_offline(self):
        with mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}), contextlib.redirect_stderr(io.StringIO()):
            encoder = embeddings.SentenceEncoder(embeddings.MINILM)
            vectors = encoder.fit_transform(TEXTS[:3])
        self.assertEqual(vectors.shape, (3, 384))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4))
        self.assertEqual(encoder.dimensions, 384)
        self.assertGreater(encoder.seconds, 0.0)


@unittest.skipUnless(embeddings.encoder_available() and cached_model(embeddings.MINILM) and embeddings.cuda_available(),
                     "sentence-transformers, the cached MiniLM or a CUDA GPU missing")
class RealGpuTest(unittest.TestCase):
    def test_minilm_on_the_gpu_gives_the_cpus_vectors_and_neighbours(self):
        with mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}), contextlib.redirect_stderr(io.StringIO()):
            gpu = embeddings.SentenceEncoder(embeddings.MINILM, device="cuda")
            cpu = embeddings.SentenceEncoder(embeddings.MINILM, device="cpu")
            on_gpu, on_cpu = gpu.fit_transform(TEXTS), cpu.fit_transform(TEXTS)
        self.assertEqual((gpu.device, cpu.device), ("cuda", "cpu"))
        self.assertGreater(float(np.min(np.sum(on_gpu * on_cpu, axis=1))), 0.9999)  # fp32: the cache's key has no device

        def nearest(vectors):
            return np.sort(np.argsort(-(vectors @ vectors.T), axis=1)[:, 1:similar.TOP_K + 1], axis=1)

        np.testing.assert_array_equal(nearest(on_gpu), nearest(on_cpu))


if __name__ == "__main__":
    unittest.main()
