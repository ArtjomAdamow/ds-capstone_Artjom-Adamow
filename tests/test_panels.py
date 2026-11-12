"""The model panels' parts (app.panels; Dash), as the dashboard
uses them: the letters and their names, the decks' preset with its cut label, the figure slot's two
helpers, and S2's variant label of the vectoriser the app runs. RecordingApp, the stub Dash app whose callback
decorator returns the function, serves the dashboard's tests."""

import model_support  # noqa: F401  (the package on the path)

import unittest

import plotly.graph_objects as go

from app.panels import DECK_CUT, DECK_PART, PANEL_LETTERS, PANEL_NAMES, empty_slot, graph, s2_label


class RecordingApp:
    def callback(self, *dependencies):
        return lambda function: function


class PanelsTest(unittest.TestCase):
    def test_the_letters_in_their_order_with_their_names(self):
        self.assertEqual(PANEL_LETTERS, ("S", "C", "T", "D", "P"))
        self.assertEqual(PANEL_NAMES["C"], "Course model")

    def test_the_cut_label_names_the_decks_preset(self):
        self.assertEqual(DECK_PART, 5)
        self.assertEqual(DECK_CUT.format(size=DECK_PART), "groups of ≤ 5 decks")

    def test_a_graph_fills_its_slot_without_plotlys_bar_and_an_empty_slot_is_a_line(self):
        shown = graph(go.Figure(), "panel-graph")
        self.assertEqual((shown.className, shown.config["displayModeBar"], shown.style["height"]), ("panel-graph", False, "100%"))
        self.assertTrue(shown.responsive)
        empty = empty_slot("No cards in this view.")
        self.assertEqual((empty.className, empty.children.className, empty.children.children), ("slot-note", "empty-state", "No cards in this view."))

    def test_the_variant_label_is_evaluate_pys_name_of_the_apps_vectoriser(self):
        self.assertEqual(s2_label(), "hybrid: chosen + MiniLM")
        self.assertEqual(s2_label("minilm"), "MiniLM")
        self.assertEqual(s2_label("hybrid-mpnet"), "hybrid: chosen + mpnet")
        self.assertIsNone(s2_label("tfidf"))


if __name__ == "__main__":
    unittest.main()
