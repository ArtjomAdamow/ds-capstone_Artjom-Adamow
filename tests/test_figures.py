"""The models' figures of the dashboard's panels (app.figures; plotly only): D1's confusion matrix on a measured
report, the sphere's surface, the connections as one trace of straight lines, T1's margins on the teal scale with
the threshold and the focus labelled; no font below 16 px and no fixed height in any figure."""

import model_support

import math
import unittest

from app import figures


def font_sizes(value, under_font: bool = False) -> list:
    """Every font size in a figure's JSON: the "size" of a dict reached through a key that names a font."""
    sizes = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if key == "size" and under_font and isinstance(inner, (int, float)):
                sizes.append(inner)
            else:
                sizes += font_sizes(inner, key.endswith("font"))
    elif isinstance(value, (list, tuple)):
        for inner in value:
            sizes += font_sizes(inner, under_font)
    return sizes


class MetricsFiguresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = model_support.difficulty_report({"201002": "hard"})

    def test_the_confusion_matrix_holds_each_true_levels_shares(self):
        matrix = figures.confusion_figure(self.report)
        heatmap = matrix.data[0]
        self.assertEqual(list(heatmap.x), ["easy", "medium", "hard"])
        self.assertEqual([[float(value) for value in row] for row in heatmap.z], self.report.confusion.tolist())
        self.assertEqual({annotation.font.size for annotation in matrix.layout.annotations}, {18})  # the cells at 18 px
        self.assertEqual(len(matrix.layout.annotations), 9)

    def test_no_font_is_below_16_px_and_no_figure_has_a_fixed_height(self):
        for name, figure in (
            ("confusion", figures.confusion_figure(self.report)),
            ("margins", figures.margin_histogram([0.01, 0.2, 0.4], 0.05, 0.3)),
        ):
            sizes = font_sizes(figure.to_plotly_json())
            self.assertTrue(sizes, name)
            self.assertGreaterEqual(min(sizes), figures.FONT_SIZE, (name, sorted(sizes)))
            self.assertIsNone(figure.layout.height, name)  # the slot gives the size
            self.assertTrue(figure.layout.autosize, name)
        self.assertEqual((figures.FONT_SIZE, figures.LABEL_SIZE), (16, 18))


def radius(x, y, z) -> float:
    return math.sqrt(x * x + y * y + z * z)


class SphereTest(unittest.TestCase):
    def test_the_surface_is_a_mesh_of_the_unit_sphere_in_one_colour(self):
        surface = figures.sphere_surface("#7c8b99")
        rings, steps = figures.SURFACE_STEPS // 2, figures.SURFACE_STEPS
        self.assertEqual((len(surface.x), len(surface.x[0])), (rings + 1, steps + 1))
        for row, column in ((0, 0), (3, 7), (rings, steps)):
            self.assertAlmostEqual(radius(surface.x[row][column], surface.y[row][column], surface.z[row][column]), 1.0)
        self.assertEqual(surface.colorscale[0][1], "#7c8b99")
        self.assertEqual(surface.hoverinfo, "skip")

    def test_the_connections_are_one_trace_of_straight_lines_between_the_drawn_dots(self):
        points = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0), "c": (0.0, 0.0, 1.0)}
        straight = figures.links_trace(points, [("a", "b"), ("b", "c"), ("a", "gone")])
        self.assertEqual(list(straight.x), [1.0, 0.0, None, 0.0, 0.0, None])  # a pair with a dot not drawn is left out
        self.assertEqual(list(straight.z), [0.0, 0.0, None, 0.0, 1.0, None])
        self.assertEqual((straight.line.color, straight.line.dash, straight.line.width), (figures.GOLD, "dash", figures.LINK_WIDTH))
        self.assertEqual((straight.name, straight.showlegend), ("Connections", False))
        named = figures.links_trace(points, [("a", "b")], name="connections: groups of ≤ 5 decks", legend=True)
        self.assertEqual((named.name, named.showlegend), ("connections: groups of ≤ 5 decks", True))
        self.assertIsNone(figures.links_trace(points, [("a", "gone")]))


class MarginsTest(unittest.TestCase):
    def test_the_teal_scale_runs_from_pale_at_zero_to_full_at_the_scale(self):
        self.assertEqual(figures.margin_color(0.0, 0.2), "#ccfbf1")
        self.assertEqual(figures.margin_color(0.2, 0.2), "#14b8a6")
        self.assertEqual(figures.margin_color(0.5, 0.2), "#14b8a6")  # above the scale: full
        self.assertEqual(figures.margin_color(0.1, 0.0), "#ccfbf1")  # no scale: pale

    def test_the_histogram_labels_the_threshold_and_the_dashboards_focus_on_the_plot(self):
        figure = figures.margin_histogram([0.01, 0.2, 0.4], 0.05)
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertEqual(figure.layout.annotations[0].text, "between two topics below 0.05")
        self.assertEqual(figure.data[0].marker.color, "#14b8a6")
        focus = figures.margin_histogram([0.01, 0.2, 0.4], 0.05, 0.3, 0.25)
        self.assertEqual(len(focus.layout.shapes), 2)
        self.assertEqual(focus.layout.annotations[1].text, "the 25 % most confident above 0.30")
        self.assertEqual({note.bgcolor for note in focus.layout.annotations}, {figures.LABEL_BACKING})  # on a dark backing


if __name__ == "__main__":
    unittest.main()
