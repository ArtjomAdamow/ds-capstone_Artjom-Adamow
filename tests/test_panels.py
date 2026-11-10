"""The model panels' parts (app.panels; Dash), as the dashboard
uses them: the letters and their names, the cut labels with the decks view's preset, the figure slot's two
helpers, and S2's and D2's variant labels of the models the app runs. RecordingApp, the stub Dash app whose callback
decorator returns the function, serves the dashboard's tests."""

import model_support  # noqa: F401  (the package on the path)

import unittest

import plotly.graph_objects as go

from app.panels import (
    CARDS_VIEW, CUT_LABELS, DECK_PART, DECKS_VIEW, PANEL_LETTERS, PANEL_NAMES, TOPICS_VIEW, d2_label, empty_slot, graph, s2_label,
)


class RecordingApp:
    def callback(self, *dependencies):
        return lambda function: function


class PanelsTest(unittest.TestCase):
    def test_the_letters_in_their_order_with_their_names(self):
        self.assertEqual(PANEL_LETTERS, ("S", "C", "T", "D", "P"))
        self.assertEqual(PANEL_NAMES["C"], "Course model")

    def test_the_cut_labels_name_the_preset_of_each_view(self):
        self.assertEqual(DECK_PART, 5)
        self.assertEqual(CUT_LABELS[DECKS_VIEW].format(size=DECK_PART), "groups of ≤ 5 decks")
        self.assertEqual(CUT_LABELS[CARDS_VIEW].format(size=4), "groups of ≤ 4 cards")
        self.assertEqual(CUT_LABELS[TOPICS_VIEW], "the full tree")

    def test_a_graph_fills_its_slot_without_plotlys_bar_and_an_empty_slot_is_a_line(self):
        shown = graph(go.Figure(), "panel-graph")
        self.assertEqual((shown.className, shown.config["displayModeBar"], shown.style["height"]), ("panel-graph", False, "100%"))
        self.assertTrue(shown.responsive)
        empty = empty_slot("No cards in this view.")
        self.assertEqual((empty.className, empty.children.className, empty.children.children), ("slot-note", "empty-state", "No cards in this view."))

    def test_the_variant_labels_are_evaluate_pys_names_of_the_apps_models(self):
        self.assertEqual(s2_label(), "hybrid: chosen + MiniLM")
        self.assertEqual(s2_label("minilm"), "MiniLM")
        self.assertEqual(s2_label("hybrid-mpnet"), "hybrid: chosen + mpnet")
        self.assertIsNone(s2_label("tfidf"))
        self.assertEqual(d2_label(), "criterion, large")  # the app's D2: the criterion template on the large model


if __name__ == "__main__":
    unittest.main()
