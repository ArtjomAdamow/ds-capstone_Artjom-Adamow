"""The navigation prompt P1 (models.prompt_model) on the test course of model_support, on an index that keeps
every term (min_df 1: on 13 cards the baseline's min_df 2 drops "slice" and "merge", each in one card): the floors
as percentiles over the self set, the three routes' scores per topic, the decision (the few-shot route gates, the
mean ranks, the nearest card selected inside the entered topic and only there), "no match" with the three nearest, the self set with
the card held out and its Source: line cut, the demo's deck 00 alone with its own floors, the label of a set of cards
(the nearest three topics by its centroid against the headings, their names within the target length), and the same
code on S2's hybrid (a stand-in encoder, no torch)."""

import model_support  # noqa: F401  (the package on the path, the test course as card records)

import dataclasses
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from models.cards import CardRecord
from app.course_table import load_table
from models import embeddings, similar
from models.prompt_model import (
    CARD, CENTROID, CUT_MARK, DEFAULT_ROUTES, ENTERED, LABEL_SEPARATOR, LABEL_TARGET, NEAREST, NO_MATCH, ROUTES, SELECTED,
    TEXT, Label, PromptModel, checked, first_heading, gate_of, label_text, request_text,
)
from models.topic_model import TopicModel, unit

CARDS = model_support.CARDS
LISTS, DICTIONARIES, EMPTY = ("Alpha", "01"), ("Alpha", "02"), ("Alpha", "03")
COMMITS, BRANCHES = ("Beta", "01"), ("Beta", "02")
INTERVALS, TESTS = ("Gamma", "01"), ("Gamma", "02")


def index_of(cards=CARDS, vectoriser=None) -> similar.SimilarityIndex:
    return similar.SimilarityIndex.fit(cards, vectoriser or similar.tfidf_vectoriser(min_df=1))


def model_of(index, seeds=None, cards=CARDS) -> PromptModel:
    return PromptModel(index, TopicModel(index, model_support.SEEDS if seeds is None else seeds), cards)


def with_headings(key, headings) -> list:
    """The test course's seeds with one topic's headings replaced."""
    return [dataclasses.replace(seed, headings=headings) if seed.key == key else seed for seed in model_support.SEEDS]


class FloorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = index_of()
        cls.model = model_of(cls.index)

    def test_the_floors_are_percentiles_of_the_self_set(self):
        model, index = self.model, self.index
        questions = index.queries([card.question for card in CARDS])
        cosines = np.asarray((questions @ index.vectors.T).todense())
        np.fill_diagonal(cosines, -np.inf)  # the card held out
        first = cosines.max(axis=1)
        self.assertAlmostEqual(model.floors.card, float(np.percentile(first, 10)))
        members = model.topic_model.members
        topic_of = {card_id: key for key, ids in members.items() for card_id in ids}
        own_centroid, own_text = [], []
        for number, card in enumerate(CARDS):
            key = topic_of[card.id]
            others = [index.row(card_id) for card_id in members[key] if card_id != card.id]
            loo = unit(np.asarray(index.vectors[others].sum(axis=0)).ravel()) if others else np.zeros(index.vectors.shape[1])
            question = np.asarray(questions[number].todense()).ravel()
            own_centroid.append(float(question @ loo))
            seed = next(seed for seed in model_support.SEEDS if seed.key == key)
            own_text.append(float(question @ index.query(" | ".join(seed.headings))))
        self.assertAlmostEqual(model.floors.centroid, float(np.percentile(own_centroid, 10)))
        self.assertAlmostEqual(model.floors.text, float(np.percentile(own_text, 10)))
        self.assertEqual(model.floors.of(CARD), model.floors.card)
        self.assertEqual(len(dataclasses.fields(type(model.floors))), 3)  # no card bar

    def test_the_routes_score_every_topic_by_its_best_member_its_full_centroid_and_its_headings(self):
        model, index = self.model, self.index
        request = "git merge a branch"
        vector = index.query(request)
        cosines = index.vectors @ vector
        by_key = {target.key: target for target in model.score(request)}
        self.assertEqual(set(by_key), {seed.key for seed in model_support.SEEDS})
        for key, members in model.topic_model.members.items():
            scores = by_key[key].scores
            self.assertEqual(set(scores), {CARD, CENTROID, TEXT})
            if members:
                rows = [index.row(card_id) for card_id in members]
                self.assertAlmostEqual(scores[CARD], float(cosines[rows].max()))
                self.assertEqual(by_key[key].card, members[int(np.argmax(cosines[rows]))])
                self.assertAlmostEqual(scores[CENTROID], float(unit(np.asarray(index.vectors[rows].sum(axis=0)).ravel()) @ vector))
            else:  # the emptied topic: no example, no card
                self.assertEqual((scores[CARD], scores[CENTROID], by_key[key].card), (0.0, 0.0, None))
            seed = next(seed for seed in model_support.SEEDS if seed.key == key)
            self.assertAlmostEqual(scores[TEXT], float(index.query(" | ".join(seed.headings)) @ vector))
        self.assertGreater(by_key[BRANCHES].scores[CARD], by_key[COMMITS].scores[CARD])

    def test_the_variants_and_their_gates(self):
        self.assertEqual(DEFAULT_ROUTES, (CARD,))
        self.assertEqual(list(ROUTES), ["card", "centroid", "text", "card+text", "all"])
        self.assertEqual([gate_of(routes) for routes in ROUTES.values()], [CARD, CENTROID, TEXT, CARD, CARD])
        self.assertEqual(checked(["card", "text"]), (CARD, TEXT))
        for wrong in ((), ("card", "headings")):
            with self.assertRaises(ValueError):
                checked(wrong)
        with self.assertRaises(ValueError):
            self.model.find("git", ("nearest",))


class DecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = index_of()
        cls.model = model_of(cls.index)

    def test_how_do_i_slice_a_list_enters_lists_and_selects_the_slicing_quiz(self):
        answer = self.model.find("how do I slice a list")
        self.assertEqual((answer.decision, answer.topic, answer.card), (SELECTED, LISTS, "201002"))
        self.assertTrue(answer.navigate)
        self.assertEqual(answer.message, "card selected")
        self.assertEqual(answer.targets[0].key, LISTS)
        self.assertEqual(len(answer.targets), NEAREST)

    def test_git_merge_enters_betas_rest_topic(self):
        answer = self.model.find("git merge")
        self.assertEqual(answer.topic, BRANCHES)
        self.assertTrue(next(seed for seed in model_support.SEEDS if seed.key == BRANCHES).rest)
        self.assertEqual(answer.card, "102003")

    def test_null_hypothesis_enters_gammas_tests(self):
        answer = self.model.find("null hypothesis")
        self.assertEqual(answer.topic, TESTS)
        self.assertIn(answer.card, ("103002", "103003"))

    def test_how_to_bake_bread_is_no_match_with_the_three_nearest_listed(self):
        for routes in ROUTES.values():
            answer = self.model.find("how to bake bread", routes)
            self.assertEqual((answer.decision, answer.topic, answer.card), (NO_MATCH, None, None), routes)
            self.assertFalse(answer.navigate)
            self.assertEqual(answer.message, "no match, the three nearest below the floor")
            self.assertEqual([target.key for target in answer.targets], [LISTS, DICTIONARIES, EMPTY])  # ties in file order
        self.assertEqual(self.model.find("").decision, NO_MATCH)

    def test_a_request_in_a_topics_own_words_enters_it_with_its_nearest_card(self):
        answer = self.model.find("the history of a repository")  # Commits' heading, no one card's question
        self.assertEqual((answer.decision, answer.topic), (SELECTED, COMMITS))
        self.assertIn(answer.card, self.model.topic_model.members[COMMITS])
        self.assertEqual(answer.card, answer.targets[0].card)  # the topic's nearest card goes with the topic: no card bar
        self.assertTrue(answer.navigate)
        self.assertGreater(answer.targets[0].scores[CARD], self.model.floors.card)

    def test_a_topic_first_by_the_mean_but_below_the_few_shot_floor_is_no_match(self):
        model = model_of(self.index, with_headings(INTERVALS, ("git branch merge",)))  # Beta's words on Gamma's topic
        answer = model.find("git branch merge", ROUTES["card+text"])
        first = answer.targets[0]
        self.assertEqual(first.key, INTERVALS)  # the text route pulls it first
        self.assertEqual(first.scores[CARD], 0.0)
        self.assertEqual(answer.decision, NO_MATCH)
        self.assertEqual(answer.targets[1].key, BRANCHES)
        self.assertEqual(model.find("git branch merge", ROUTES["card"]).topic, BRANCHES)  # the card route alone

    def test_a_card_of_another_topic_as_the_top_card_is_not_selected(self):
        model = model_of(self.index, with_headings(DICTIONARIES, ("list",)))
        answer = model.find("list", ROUTES["card+text"])
        self.assertEqual((answer.decision, answer.topic, answer.card), (ENTERED, DICTIONARIES, None))
        self.assertEqual(answer.message, "topic entered")
        cosines = self.index.vectors @ self.index.query("list")
        top = self.index.ids[int(np.argmax(cosines))]
        self.assertEqual(top, "201002")  # in Lists
        self.assertEqual(model.find("list", ROUTES["card"]).card, "201002")

    def test_the_text_route_alone_never_clears_its_zero_floor_with_nothing_in_common(self):
        self.assertEqual(self.model.floors.text, 0.0)
        self.assertEqual(self.model.find("how to bake bread", ROUTES["text"]).decision, NO_MATCH)
        self.assertEqual(self.model.find("null hypothesis", ROUTES["text"]).topic, TESTS)

    def test_variants_without_the_card_route_enter_and_never_select(self):
        for name in ("centroid", "text"):
            answer = self.model.find("null hypothesis", ROUTES[name])
            self.assertEqual((answer.decision, answer.topic, answer.card), (ENTERED, TESTS, None), name)
            self.assertEqual(answer.message, "topic entered")

    def test_the_mean_ranks_and_the_answer_is_the_same_on_every_call(self):
        answer = self.model.find("git merge", ROUTES["all"])
        means = [target.mean for target in answer.ranking]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertEqual(len(answer.ranking), len(model_support.SEEDS))
        self.assertEqual(answer, self.model.find("git merge", ROUTES["all"]))
        self.assertEqual(answer.routes, ROUTES["all"])


class SelfSetTest(unittest.TestCase):
    def test_every_cards_question_holds_its_card_out(self):
        model = model_of(index_of())
        cases = model.self_set()
        self.assertEqual([case.card_id for case in cases], model_support.ALL_CARDS)
        members = model.topic_model.members
        for case in cases:
            self.assertIn(case.card_id, members[case.key])  # the truth: T1's topic
            self.assertNotEqual(case.card, case.card_id)
            self.assertEqual(len(case.nearest), NEAREST)
            self.assertIn(case.decision, (SELECTED, ENTERED, NO_MATCH))
            self.assertEqual(case.card is not None, case.decision == SELECTED)
        self.assertEqual(model.self_questions, [card.question for card in CARDS])
        self.assertEqual(len(model.self_set(ROUTES["all"])), len(CARDS))

    def test_the_source_line_is_cut_from_the_question(self):
        self.assertEqual(request_text("What does git log show?\nSource: 02_git/1_Commits.ipynb"), "What does git log show?")
        self.assertEqual(request_text("What is a p value?"), "What is a p value?")
        cards = [
            dataclasses.replace(card, question=card.question + " Source: 02_git/1_Commits.ipynb") if card.id == "102004" else card
            for card in CARDS
        ]
        model = model_of(index_of(cards), cards=cards)
        self.assertEqual(model.self_questions[model.self_ids.index("102004")], "What does git log show?")

    def test_a_topic_whose_only_member_is_held_out_has_no_example_left(self):
        index = index_of()
        fitted = TopicModel(index, model_support.SEEDS)
        members = {**fitted.members, INTERVALS: ["103001"], TESTS: ["103002", "103003", "103004"]}
        model = PromptModel(index, types.SimpleNamespace(seeds=fitted.seeds, members=members), CARDS)
        case = next(case for case in model.self_set() if case.card_id == "103001")
        self.assertEqual(case.key, INTERVALS)
        self.assertNotEqual(case.nearest[0], INTERVALS)  # nothing of its topic left to match
        self.assertTrue(all(np.isfinite([model.floors.card, model.floors.centroid, model.floors.text])))


class DeckZeroTest(unittest.TestCase):
    """P1 on a deck of deck 00's size alone (the demo's board): T1's membership as written, the floors from its own
    self set on the baseline TF-IDF, a request in its words entering its topic, the course's words no match."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder:
            table = load_table(model_support.write_demo_table(Path(folder)))
        cls.cards = table.cards
        cls.index = similar.SimilarityIndex.fit(table.cards)
        cls.model = PromptModel(cls.index, TopicModel(cls.index, table.seeds, moves=False), table.cards)

    def test_the_floors_come_from_deck_00s_own_self_set(self):
        model = self.model
        self.assertEqual(model.self_ids, [card.id for card in model_support.DEMO_CARDS])  # its cards alone
        cosines = np.asarray((self.index.queries(model.self_questions) @ self.index.vectors.T).todense())
        np.fill_diagonal(cosines, -np.inf)
        self.assertAlmostEqual(model.floors.card, float(np.percentile(cosines.max(axis=1), 10)))
        self.assertTrue(all(np.isfinite([model.floors.card, model.floors.centroid, model.floors.text])))
        self.assertNotEqual(model.floors, model_of(index_of()).floors)

    def test_a_request_in_its_words_enters_its_topic_and_the_courses_words_find_nothing(self):
        answer = self.model.find("how do you send a request")
        self.assertTrue(answer.navigate)
        self.assertEqual(answer.topic, ("How to use", "02"))
        self.assertEqual(self.model.find("what does git commit do").decision, NO_MATCH)
        self.assertTrue(all(scored.key[0] == "How to use" for scored in self.model.find("a confidence interval").ranking))


class LabelTest(unittest.TestCase):
    """The label of a set of cards: their centroid against the topics' headings, the nearest three topics with a
    cosine above 0, their names each its first heading, cut to the target length."""

    @classmethod
    def setUpClass(cls):
        cls.index = index_of()
        cls.model = model_of(cls.index)

    def test_three_distinct_topics_by_the_centroid_against_the_headings(self):
        label = self.model.label(model_support.ALL_CARDS)
        self.assertEqual(len(label.keys), NEAREST)
        self.assertEqual(len(set(label.keys)), NEAREST)
        rows = [self.index.row(card_id) for card_id in model_support.ALL_CARDS]
        centroid = unit(np.asarray(self.index.vectors[rows].sum(axis=0)).ravel())
        texts = {seed.key: unit(self.index.query(" | ".join(seed.headings))) for seed in model_support.SEEDS if seed.headings}
        scores = {key: float(centroid @ vector) for key, vector in texts.items()}
        self.assertEqual(list(label.keys), sorted(scores, key=lambda key: -scores[key])[:NEAREST])  # the nearest three
        self.assertEqual(label.text, LABEL_SEPARATOR.join(first_heading(self.name(key)) for key in label.keys))

    def name(self, key):
        return next(seed.name for seed in model_support.SEEDS if seed.key == key)

    def test_a_topics_own_members_are_labelled_with_it_among_the_three(self):
        members = self.model.topic_model.members
        for key, card_ids in members.items():
            if key in (EMPTY, BRANCHES):
                continue  # Empty has no card; Branches' headings share no word with its cards (no stemming)
            with self.subTest(key=key):
                self.assertIn(key, self.model.label(card_ids).keys)
        self.assertEqual(self.model.label(members[TESTS]).keys[0], TESTS)  # the p value and the null hypothesis
        self.assertNotIn(BRANCHES, self.model.label(members[BRANCHES]).keys)  # "branch" and "merge" against "Branches | Merging"

    def test_no_label_without_a_known_card_or_a_word_in_common(self):
        self.assertEqual(self.model.label([]), Label())
        self.assertEqual(self.model.label(["999999"]).text, "")  # a card the index does not hold
        self.assertEqual(self.model.label(["999999", "103002", "103002"]), self.model.label(["103002"]))  # left out, once
        cards = CARDS + [CardRecord("109001", "Delta", "Zebra?", "Quokka.")]
        model = model_of(index_of(cards), cards=cards)
        self.assertEqual(model.label(["109001"]), Label())  # nothing in common with any topic's headings

    def test_the_text_is_within_the_target_with_the_short_names_whole(self):
        self.assertEqual(LABEL_TARGET, 70)
        for card_ids in ([card.id for card in CARDS[:3]], model_support.GAMMA_CARDS, model_support.ALL_CARDS):
            self.assertLessEqual(len(self.model.label(card_ids).text), LABEL_TARGET)
        names = ["Accuracy", "Why do we make data visualisation", "Gestalt Principles of Visual Perception"]
        text = label_text(names)
        self.assertEqual(len(text), LABEL_TARGET)
        first, second, third = text.split(LABEL_SEPARATOR)
        self.assertEqual(first, "Accuracy")  # the short one whole
        self.assertTrue(second.endswith(CUT_MARK) and third.endswith(CUT_MARK))  # the long ones cut to one length
        self.assertEqual(len(second), len(third))
        self.assertEqual(label_text(["Lists", "Tests"]), "Lists · Tests")  # within the target: nothing cut
        self.assertEqual(label_text([]), "")
        self.assertLessEqual(len(label_text(names, 30)), 30)
        self.assertEqual(label_text(names, 4), " · ".join([CUT_MARK] * 3))  # too small: the marks alone
        long_label = self.model.label(model_support.ALL_CARDS, target=12)
        self.assertLessEqual(len(long_label.text), 12)

    def test_a_names_first_heading(self):
        self.assertEqual(first_heading("Large Language Models (LLMs); Prompt structure …"), "Large Language Models (LLMs)")
        self.assertEqual(first_heading("Git; Using Git Locally"), "Git")
        self.assertEqual(first_heading("GitHub"), "GitHub")
        self.assertEqual(first_heading("  Lists  "), "Lists")
        self.assertEqual(first_heading(""), "")
        self.assertEqual(self.model.label_names, [first_heading(seed.name) for seed in model_support.SEEDS])


class EncoderTest(unittest.TestCase):
    def test_the_same_code_runs_on_the_hybrid_with_one_encode_per_request(self):
        encoder = embeddings.SentenceEncoder(embeddings.MINILM, loader=model_support.fake_loader)
        hybrid = embeddings.HybridVectoriser(similar.tfidf_vectoriser(min_df=1), encoder)
        model = model_of(index_of(vectoriser=hybrid))
        self.assertTrue(-1.0 <= model.floors.card <= 1.0)
        calls = encoder.load().calls
        answer = model.find("git merge")
        self.assertEqual(encoder.load().calls, calls + 1)
        self.assertEqual(len(answer.ranking), len(model_support.SEEDS))
        self.assertEqual(len(model.self_set(ROUTES["all"])), len(CARDS))
        label = model.label(model_support.GAMMA_CARDS)
        self.assertEqual(encoder.load().calls, calls + 1)  # the label encodes nothing
        self.assertTrue(label.keys and len(label.text) <= LABEL_TARGET)


if __name__ == "__main__":
    unittest.main()
