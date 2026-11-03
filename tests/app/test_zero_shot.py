"""D2's zero-shot tags (app.models.zero_shot): the premise, the two templates, the tagger with a fake pipeline and
the cache, the tags of a course's quizzes with the skip lines, and the real model, which skips without the library
or the cached model."""

import model_support  # noqa: F401  (the helper's package on the path)

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.cards import CardRecord
from app.models import zero_shot
from app.models.difficulty_model import LEVELS
from app.models.embeddings import CACHE_FOLDER, EmbeddingCache
from app.models.zero_shot import (
    BASE_MODEL, CRITERION, LARGE_MODEL, LEVEL_NAMES, NLI_MODEL, NLI_MODELS, TEMPLATES, Tags, ZeroShotTagger, cached_tags, premise,
    size_of, tagger_available, tags_of,
)

QUIZ = CardRecord(
    "201001", "Alpha", "  What does `len` return?  ", "The number of items.", stack=2, explain="It counts.\nSource: `lists.ipynb`.",
    options=("The number of items.", "The last item.", "A copy."), answers=("Right: it counts the items.", "No, try again.", ""), correct=(0,),
)
MANY = CardRecord(
    "201002", "Alpha", "Which are lists?", "a and c", stack=2, options=("a", "b", "c"), answers=("a is one", "b is not", "c is one"), correct=(0, 2),
)


def cached_model(model_name: str) -> bool:
    """Whether the hub's cache holds the model (nothing is fetched)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return False
    return (Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}").is_dir()


class PremiseTest(unittest.TestCase):
    def test_the_premise_is_the_question_the_options_the_correct_feedback_and_the_explanation_without_its_source_line(self):
        self.assertEqual(premise(QUIZ), "What does `len` return?\nThe number of items.\nThe last item.\nA copy.\nRight: it counts the items.\nIt counts.")
        self.assertNotIn("try again", premise(QUIZ))  # the wrong options' feedback stays out
        self.assertNotIn("Source", premise(QUIZ))

    def test_every_correct_options_feedback_of_a_many_choice_quiz_in_order(self):
        self.assertEqual(premise(MANY), "Which are lists?\na\nb\nc\na is one\nc is one")

    def test_a_card_without_options_gives_its_question_its_answer_and_its_explanation(self):
        card = CardRecord("101001", "Alpha", "What is a list?", "A sequence.", explain="Wikipedia. Source: `x.ipynb`.")
        self.assertEqual(premise(card), "What is a list?\nA sequence.\nWikipedia.")
        self.assertEqual(premise(CardRecord("101002", "Alpha", "Q?", "")), "Q?")
        self.assertEqual(premise(CardRecord("101003", "Alpha", "Q?", "A.", explain=None)), "Q?\nA.")


class TemplatesTest(unittest.TestCase):
    def test_the_criterion_is_the_primary_and_the_level_names_the_control(self):
        self.assertEqual(TEMPLATES, (CRITERION, LEVEL_NAMES))
        self.assertEqual(CRITERION.hypothesis, "To answer this question, a learner must {}.")
        self.assertEqual(len(CRITERION.labels), 3)
        self.assertTrue(CRITERION.labels[0].startswith("recall") and CRITERION.labels[1].startswith("apply") and CRITERION.labels[2].startswith("reason"))
        self.assertEqual(LEVEL_NAMES.labels, LEVELS)
        self.assertEqual(LEVEL_NAMES.hypothesis, "This question is {}.")
        self.assertEqual({template.name for template in TEMPLATES}, {"criterion", "level names"})

    def test_the_two_models_are_one_family_and_the_app_reads_one_of_them(self):
        self.assertEqual(NLI_MODELS, (BASE_MODEL, LARGE_MODEL))
        self.assertIn(NLI_MODEL, NLI_MODELS)
        self.assertTrue(all(name.startswith("MoritzLaurer/ModernBERT-") and name.endswith("-zeroshot-v2.0") for name in NLI_MODELS))
        self.assertEqual((size_of(BASE_MODEL), size_of(LARGE_MODEL), size_of("some/other-model")), ("base", "large", "other-model"))

    def test_the_expected_level_is_on_d1s_scale(self):
        self.assertAlmostEqual(zero_shot.expected_level((0.2, 0.3, 0.5)), 1.3)
        self.assertEqual(zero_shot.expected_level((1.0, 0.0, 0.0)), 0.0)
        self.assertEqual(zero_shot.expected_level((0.0, 0.0, 1.0)), 2.0)

    def test_the_library_is_probed_not_imported(self):
        self.assertTrue(tagger_available(lambda name: object()))
        self.assertFalse(tagger_available(lambda name: None))
        self.assertFalse(tagger_available(mock.Mock(side_effect=ImportError)))
        self.assertEqual(zero_shot.LIBRARY, "transformers")


class TaggerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.cards = [card for card in model_support.difficulty_course() if card.stack in (2, 3)]

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_scores_are_one_softmax_per_premise_in_the_templates_label_order(self):
        tagger = ZeroShotTagger(template=CRITERION, loader=model_support.fake_classifier_loader)
        scores = tagger.scores(self.cards)
        self.assertEqual(scores.shape, (len(self.cards), 3))
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)
        classifier = tagger.load()
        answer = classifier(premise(self.cards[0]), candidate_labels=list(CRITERION.labels), hypothesis_template=CRITERION.hypothesis)
        by_label = dict(zip(answer["labels"], answer["scores"]))
        np.testing.assert_allclose(scores[0], [by_label[label] for label in CRITERION.labels], atol=1e-6)  # mapped back, not in the sorted order
        self.assertGreater(tagger.seconds, 0.0)
        self.assertFalse(tagger.cached)

    def test_the_tagger_runs_on_s2s_device_handed_to_the_loader(self):
        tagger = ZeroShotTagger(loader=model_support.fake_classifier_loader, device="cpu")
        self.assertEqual(tagger.load().device, "cpu")
        with mock.patch("app.models.embeddings.cuda_available", return_value=True), \
                mock.patch.dict(os.environ, {"APP_DEVICE": "auto"}):
            tagger = ZeroShotTagger(loader=model_support.fake_classifier_loader)
            tagger.scores(self.cards)
        self.assertEqual((tagger.device, tagger.load().device), ("cuda", "cuda"))

    def test_the_cache_serves_the_second_call_and_keeps_a_file_per_model_and_template(self):
        cache = EmbeddingCache(self.folder / CACHE_FOLDER)
        tagger = ZeroShotTagger("fake-nli", CRITERION, cache=cache, loader=model_support.fake_classifier_loader)
        first = tagger.scores(self.cards)
        self.assertFalse(tagger.cached)
        calls = tagger.load().calls
        again = ZeroShotTagger("fake-nli", CRITERION, cache=cache, loader=model_support.fake_classifier_loader)
        second = again.scores(self.cards)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(again.cached)
        self.assertEqual((again.seconds, again._classifier), (0.0, None))  # nothing loaded, nothing classified
        self.assertEqual(tagger.load().calls, calls)
        control = ZeroShotTagger("fake-nli", LEVEL_NAMES, cache=cache, loader=model_support.fake_classifier_loader)
        control.scores(self.cards)
        self.assertFalse(control.cached)  # another template: another file
        files = sorted(path.name for path in (self.folder / CACHE_FOLDER).glob("*.npy"))
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].startswith("fake-nli-criterion-") and files[1].startswith("fake-nli-level-names-"))
        changed = ZeroShotTagger("fake-nli", CRITERION, cache=cache, loader=model_support.fake_classifier_loader)
        changed.scores(self.cards[:-1])  # other premises: a miss, and the template's older file goes
        self.assertFalse(changed.cached)
        self.assertEqual(len(list((self.folder / CACHE_FOLDER).glob("fake-nli-criterion-*.npy"))), 1)

    def test_no_cards_no_classification(self):
        tagger = ZeroShotTagger(loader=model_support.fake_classifier_loader)
        self.assertEqual(tagger.scores([]).shape, (0, 3))


class TagsOfTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.cards = model_support.difficulty_course()

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_quizzes_are_tagged_and_the_flashcards_not_with_the_cache_serving_the_next_call(self):
        tags = tags_of(self.cards, self.folder, loader=model_support.fake_classifier_loader)
        self.assertIsInstance(tags, Tags)
        self.assertEqual((tags.template, tags.model_name, tags.cached), (CRITERION, NLI_MODEL, False))
        self.assertEqual(set(tags.scores), {card.id for card in self.cards if card.stack in (2, 3)})
        self.assertEqual(len(tags.scores), 42)
        expected = tags.expected()
        for card_id, probabilities in tags.scores.items():
            self.assertAlmostEqual(expected[card_id], probabilities[1] + 2 * probabilities[2], places=6)
        again = tags_of(self.cards, self.folder, loader=model_support.fake_classifier_loader)
        self.assertTrue(again.cached)
        np.testing.assert_array_equal(again.scores["201101"], tags.scores["201101"])
        control = tags_of(self.cards, self.folder, LEVEL_NAMES, loader=model_support.fake_classifier_loader)
        self.assertEqual(control.template, LEVEL_NAMES)
        self.assertEqual(len(list((self.folder / CACHE_FOLDER).glob("*.npy"))), 2)
        self.assertEqual(NLI_MODEL, LARGE_MODEL)  # the app's D2
        base = tags_of(self.cards, self.folder, model_name=BASE_MODEL, loader=model_support.fake_classifier_loader)
        self.assertEqual((base.model_name, base.template, base.cached), (BASE_MODEL, CRITERION, False))  # another model: its own file
        self.assertEqual(len(list((self.folder / CACHE_FOLDER).glob("*.npy"))), 3)
        self.assertEqual(cached_tags(self.cards, self.folder, CRITERION, BASE_MODEL).model_name, BASE_MODEL)
        self.assertIsNone(cached_tags(self.cards, self.folder, LEVEL_NAMES, BASE_MODEL))  # the base model's control: not tagged

    def test_the_cached_tags_need_neither_the_library_nor_the_model(self):
        self.assertIsNone(cached_tags(self.cards, None))
        self.assertIsNone(cached_tags(self.cards, self.folder))  # nothing tagged yet
        tags = tags_of(self.cards, self.folder, loader=model_support.fake_classifier_loader)
        with mock.patch.object(zero_shot, "tagger_available", return_value=False), \
                mock.patch.object(ZeroShotTagger, "load", side_effect=AssertionError("no model is loaded")):
            cached = cached_tags(self.cards, self.folder)
        self.assertEqual((cached.template, cached.cached, cached.seconds), (CRITERION, True, 0.0))
        self.assertEqual(set(cached.scores), set(tags.scores))
        np.testing.assert_array_equal(cached.scores["201101"], tags.scores["201101"])
        self.assertIsNone(cached_tags(self.cards, self.folder, LEVEL_NAMES))  # another template: its own file, not there
        self.assertIsNone(cached_tags(self.cards[:-5], self.folder))  # another course: another key

    def test_without_the_library_none_with_one_line(self):
        lines = []
        with mock.patch.object(zero_shot, "tagger_available", return_value=False):
            self.assertIsNone(tags_of(self.cards, self.folder, output=lines.append))
        self.assertEqual(len(lines), 1)
        self.assertIn("transformers", lines[0])

    def test_a_model_that_does_not_load_gives_none_and_the_fetching_command(self):
        lines = []

        def failing(model_name, device):
            raise OSError("not cached, offline")

        self.assertIsNone(tags_of(self.cards, self.folder, output=lines.append, loader=failing))
        self.assertEqual(len(lines), 1)
        self.assertIn("OSError", lines[0])
        self.assertIn(zero_shot.FETCH_COMMAND.format(name=NLI_MODEL), lines[0])
        self.assertEqual(list((self.folder / CACHE_FOLDER).glob("*.npy")) if (self.folder / CACHE_FOLDER).is_dir() else [], [])


@unittest.skipUnless(tagger_available() and cached_model(NLI_MODEL), "transformers or the cached NLI model missing")
class RealModelTest(unittest.TestCase):
    def test_the_model_tags_the_test_courses_quizzes_offline(self):
        cards = [card for card in model_support.difficulty_course() if card.stack in (2, 3)][:4]
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), contextlib.redirect_stderr(io.StringIO()):
            tagger = ZeroShotTagger()
            scores = tagger.scores(cards)
        self.assertEqual(scores.shape, (4, 3))
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-4)
        self.assertTrue((scores >= 0).all())


if __name__ == "__main__":
    unittest.main()
