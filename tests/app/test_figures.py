"""The shared figures (app.figures; plotly only, nothing of the viewer): D1's score rows, subtitle,
confusion matrix and shares on a measured report, the sphere's surface and its arcs, and the dendrogram of a
clustering over scipy's coordinates with the real merge distances and the preset's cut-off line."""

import model_support

import math
import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from app import figures
from app.models import clustering


class MetricsFiguresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = model_support.difficulty_report({"201002": "hard"})

    def test_the_score_rows_are_d1_with_its_spread_and_the_baselines_means(self):
        self.assertEqual(figures.score_header(self.report), ["D1"] + list(self.report.baselines))
        rows = figures.score_rows(self.report)
        self.assertEqual([label for label, _d1, _baselines in rows], [label for _key, label in figures.TABLE_ROWS])
        label, d1, baselines = rows[0]
        mean, sd = self.report.metrics["macro_f1"]
        self.assertEqual(d1, f"{mean:.2f} ± {sd:.2f}")
        self.assertEqual(len(baselines), len(self.report.baselines))
        self.assertEqual(figures.number(float("nan")), "–")

    def test_the_subtitle_names_the_scope(self):
        self.assertTrue(figures.metrics_subtitle(self.report).startswith("All decks: 36 rated quizzes"))
        deck = self.report.for_deck("Alpha")
        self.assertTrue(figures.metrics_subtitle(deck).startswith("This deck: its 12 rated quizzes (8 easy, 3 medium, 1 hard)"))
        topic = deck.for_topic("Some topic", ("301001", "201101"))
        self.assertTrue(figures.metrics_subtitle(topic).startswith("This topic: its 1 rated quiz (1 easy, 0 medium, 0 hard)"))

    def test_the_confusion_matrix_and_the_shares(self):
        heatmap = figures.confusion_figure(self.report).data[0]
        self.assertEqual(list(heatmap.x), ["easy", "medium", "hard"])
        self.assertEqual([[float(value) for value in row] for row in heatmap.z], self.report.confusion.tolist())
        rated, predicted = figures.shares_figure(self.report).data
        self.assertEqual(list(rated.customdata), list(self.report.rated_counts))
        self.assertEqual((rated.marker.color, predicted.marker.color), (figures.RATED_COLOR, figures.PREDICTED_COLOR))
        self.assertEqual(list(predicted.customdata), [0, 0, 1])


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

    def test_an_arc_follows_the_great_circle_a_hair_above_the_surface(self):
        xs, ys, zs = figures.surface_arc((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.assertEqual(len(xs), math.ceil((math.pi / 2) / figures.ARC_STEP) + 1)
        self.assertAlmostEqual(xs[0], figures.ARC_LIFT)
        self.assertAlmostEqual(ys[-1], figures.ARC_LIFT)
        for x, y, z in zip(xs, ys, zs):
            self.assertAlmostEqual(radius(x, y, z), figures.ARC_LIFT)
            self.assertAlmostEqual(z, 0.0)

    def test_antipodes_take_a_half_circle_and_the_centre_a_straight_line(self):
        xs, ys, zs = figures.surface_arc((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), lift=1.0)
        self.assertAlmostEqual(xs[-1], -1.0)
        self.assertGreater(max(zs), 0.99)  # over the pole, not through the centre
        for x, y, z in zip(xs, ys, zs):
            self.assertAlmostEqual(radius(x, y, z), 1.0)
        self.assertEqual(figures.surface_arc((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0]))


class DendrogramTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.RandomState(0)
        self.vectors = generator.rand(8, 5)
        self.ids = [f"card-{index}" for index in range(8)]
        self.merges = linkage(squareform(clustering.cosine_distances(self.vectors), checks=False), method=clustering.METHOD)
        self.pairs = clustering.merge_links(self.ids, self.vectors)

    def test_the_tree_is_scipys_with_the_merge_distances_as_heights(self):
        figure = figures.dendrogram_figure(self.merges, self.ids, {card_id: card_id.upper() for card_id in self.ids})
        merges, leaves = figure.data[0], figure.data[1]
        heights = {round(float(y), 9) for y in merges.y if y is not None and y > 0}
        self.assertEqual(heights, {round(float(height), 9) for height in self.merges[:, 2]})
        self.assertEqual(sorted(leaves.customdata), sorted(self.ids))
        self.assertEqual(list(figure.layout.xaxis.ticktext), [card_id.upper() for card_id in leaves.customdata])
        self.assertEqual(merges.line.color, figures.GOLD)
        self.assertIsNone(figure.layout.meta)  # no cut-off line without a preset

    def test_the_cut_off_line_sits_between_the_last_kept_merge_and_the_next(self):
        kept = clustering.preset_rank(self.pairs, 3)
        self.assertGreater(kept, 0)
        self.assertLess(kept, len(self.pairs))
        figure = figures.dendrogram_figure(self.merges, self.ids, {}, kept)
        cut = figure.data[2]
        self.assertEqual((cut.name, cut.line.dash), ("Cut-off", "dash"))
        height = cut.y[0]
        self.assertLess(self.merges[kept - 1, 2], height)
        self.assertLess(height, self.merges[kept, 2])
        self.assertEqual(figure.layout.meta, {"cut": height, "kept": kept})

    def test_the_line_goes_under_the_first_merge_or_over_the_top(self):
        self.assertAlmostEqual(figures.cut_height(self.merges, 0), self.merges[0, 2] / 2)
        self.assertGreater(figures.cut_height(self.merges, len(self.merges)), self.merges[-1, 2])


if __name__ == "__main__":
    unittest.main()
