"""The models' figures (app.figures; plotly only): D1's score rows, subtitle,
confusion matrix and shares on a measured report, the sphere's surface and its arcs, the connections as one trace,
T1's margins on the teal scale with the threshold and the focus labelled, the dendrogram of a
clustering over scipy's coordinates with the real merge distances and the preset's cut-off line labelled on the
plot, its leaves labelled up to twelve, a card's neighbours as bars; no font below 16 px and no fixed height in
any figure."""

import model_support

import math
import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from app import figures
from app.models import clustering


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


def merges_of(vectors):
    return linkage(squareform(clustering.cosine_distances(vectors), checks=False), method=clustering.METHOD)


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

    def test_d2s_scores_and_subtitle_name_it_and_its_one_pass(self):
        from app.models.difficulty_model import judge_tags

        cards = model_support.difficulty_course()
        tags = {card.id: (0.5, 0.3, min(0.2, len(card.question) / 400)) for card in cards}
        report = judge_tags(cards, tags)
        self.assertEqual(figures.score_header(report), ["D2", "always easy"])
        self.assertIn("measured in one pass, nothing fitted", figures.metrics_subtitle(report))
        deck = report.for_deck("Alpha")
        self.assertIn("(8 easy, 3 medium, 1 hard), in one pass, nothing fitted", figures.metrics_subtitle(deck))
        self.assertIn("out of fold (", figures.metrics_subtitle(self.report))  # D1's stays its cross-validation

    def test_the_confusion_matrix_and_the_shares(self):
        matrix = figures.confusion_figure(self.report)
        heatmap = matrix.data[0]
        self.assertEqual(list(heatmap.x), ["easy", "medium", "hard"])
        self.assertEqual([[float(value) for value in row] for row in heatmap.z], self.report.confusion.tolist())
        self.assertEqual({annotation.font.size for annotation in matrix.layout.annotations}, {18})  # the cells at 18 px
        self.assertEqual(len(matrix.layout.annotations), 9)
        rated, predicted = figures.shares_figure(self.report).data
        self.assertEqual(list(rated.customdata), list(self.report.rated_counts))
        self.assertEqual((rated.marker.color, predicted.marker.color), (figures.RATED_COLOR, figures.PREDICTED_COLOR))
        self.assertEqual(list(predicted.customdata), [0, 0, 1])

    def test_no_font_is_below_16_px_and_no_figure_has_a_fixed_height(self):
        generator = np.random.RandomState(0)
        vectors = generator.rand(6, 4)
        ids = [f"card-{index}" for index in range(6)]
        for name, figure in (
            ("confusion", figures.confusion_figure(self.report)), ("shares", figures.shares_figure(self.report)),
            ("dendrogram", figures.dendrogram_figure(merges_of(vectors), ids, {}, 2, cut_label="groups of ≤ 4 cards")),
            ("neighbours", figures.neighbours_figure([("a", 0.9, figures.HERE), ("b", 0.5, figures.THIS_DECK), ("c", 0.3, figures.ELSEWHERE)])),
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

    def test_the_connections_are_one_trace_of_the_drawn_dots_pairs_straight_or_along_the_surface(self):
        points = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0), "c": (0.0, 0.0, 1.0)}
        straight = figures.links_trace(points, [("a", "b"), ("b", "c"), ("a", "gone")], along_surface=False)
        self.assertEqual(list(straight.x), [1.0, 0.0, None, 0.0, 0.0, None])  # a pair with a dot not drawn is left out
        self.assertEqual((straight.line.color, straight.line.dash, straight.line.width), (figures.GOLD, "dash", figures.LINK_WIDTH))
        self.assertFalse(straight.showlegend)
        arc = figures.links_trace(points, [("a", "b")], along_surface=True)
        self.assertEqual(len(arc.x), len(figures.surface_arc(points["a"], points["b"])[0]) + 1)
        self.assertIsNone(figures.links_trace(points, [("a", "gone")], along_surface=True))
        grouped = figures.links_trace(points, [("a", "b")], True, "pairs", legend=True, group={"legendgroup": "S"})
        self.assertEqual((grouped.name, grouped.showlegend, grouped.legendgroup), ("pairs", True, "S"))


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


class DendrogramTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.RandomState(0)
        self.vectors = generator.rand(8, 5)
        self.ids = [f"card-{index}" for index in range(8)]
        self.merges = merges_of(self.vectors)
        self.pairs = clustering.merge_links(self.ids, self.vectors)

    def test_the_tree_is_scipys_with_the_merge_distances_as_heights(self):
        figure = figures.dendrogram_figure(self.merges, self.ids, {card_id: card_id.upper() for card_id in self.ids})
        merges, leaves = figure.data[0], figure.data[1]
        heights = {round(float(y), 9) for y in merges.y if y is not None and y > 0}
        self.assertEqual(heights, {round(float(height), 9) for height in self.merges[:, 2]})
        self.assertEqual(sorted(leaves.customdata), sorted(self.ids))
        self.assertEqual(list(figure.layout.xaxis.ticktext), [card_id.upper() for card_id in leaves.customdata])
        self.assertEqual(list(leaves.text), list(figure.layout.xaxis.ticktext))  # the hover: the label, without another
        self.assertEqual(merges.line.color, figures.GOLD)
        self.assertIsNone(figure.layout.meta)  # no cut-off line without a preset
        self.assertEqual(figure.layout.annotations, ())  # no label without a cut-off line
        self.assertEqual((figure.layout.yaxis.title.text, figure.layout.yaxis.title.font.size, figure.layout.yaxis.nticks), ("distance", 18, 4))
        self.assertEqual(figure.layout.yaxis.title.font.color, figures.INK)  # in full colour
        self.assertEqual(figure.layout.xaxis.tickfont.size, 16)

    def test_the_hover_keeps_a_label_the_ticks_do_not_show_and_many_leaves_get_no_ticks(self):
        hovers = {card_id: f"{card_id}-hash" for card_id in self.ids}
        figure = figures.dendrogram_figure(self.merges, self.ids, {card_id: card_id[-1] for card_id in self.ids}, hover_of=hovers)
        leaves = figure.data[1]
        self.assertEqual(list(figure.layout.xaxis.ticktext), [card_id[-1] for card_id in leaves.customdata])  # the ticks by the short label
        self.assertEqual(list(leaves.text), [hovers[card_id] for card_id in leaves.customdata])  # the hover with the hash
        generator = np.random.RandomState(1)
        vectors = generator.rand(figures.LEAF_LABELS + 1, 5)
        ids = [f"card-{index}" for index in range(len(vectors))]
        many = figures.dendrogram_figure(merges_of(vectors), ids, {})
        self.assertEqual(list(many.layout.xaxis.tickvals), [])  # beyond twelve leaves the hover alone names them
        self.assertEqual(sorted(many.data[1].text), sorted(ids))
        self.assertEqual(figures.LEAF_LABELS, 12)
        twelve = figures.dendrogram_figure(merges_of(vectors[:12]), ids[:12], {})
        self.assertEqual(len(twelve.layout.xaxis.tickvals), 12)

    def test_the_cut_off_line_sits_between_the_last_kept_merge_and_the_next_and_is_labelled(self):
        kept = clustering.preset_rank(self.pairs, 3)
        self.assertGreater(kept, 0)
        self.assertLess(kept, len(self.pairs))
        figure = figures.dendrogram_figure(self.merges, self.ids, {}, kept, cut_label="groups of ≤ 3 cards")
        cut = figure.data[2]
        self.assertEqual((cut.name, cut.line.dash), ("Cut-off", "dash"))
        height = cut.y[0]
        self.assertLess(self.merges[kept - 1, 2], height)
        self.assertLess(height, self.merges[kept, 2])
        self.assertEqual(figure.layout.meta, {"cut": height, "kept": kept})
        (label,) = figure.layout.annotations  # the cut-off line labelled on the plot
        self.assertEqual((label.text, label.y, label.font.color, label.xanchor), ("groups of ≤ 3 cards", height, figures.CUT_COLOR, "right"))
        self.assertEqual(figures.dendrogram_figure(self.merges, self.ids, {}, kept).layout.annotations, ())  # none without a label

    def test_the_line_goes_under_the_first_merge_or_over_the_top(self):
        self.assertAlmostEqual(figures.cut_height(self.merges, 0), self.merges[0, 2] / 2)
        self.assertGreater(figures.cut_height(self.merges, len(self.merges)), self.merges[-1, 2])


class NeighboursTest(unittest.TestCase):
    def test_a_bar_per_neighbour_its_length_the_score_coloured_by_where_it_lies(self):
        rows = [("102003", 0.9, figures.HERE), ("102001", 0.6, figures.THIS_DECK), ("103002", 0.4, figures.ELSEWHERE), ("102004", 0.3, figures.HERE)]
        figure = figures.neighbours_figure(rows)
        self.assertEqual([trace.name for trace in figure.data], [figures.HERE, figures.THIS_DECK, figures.ELSEWHERE])
        here = figure.data[0]
        self.assertEqual((list(here.y), list(here.x), here.marker.color), (["102003", "102004"], [0.9, 0.3], figures.PLACE_COLORS[figures.HERE]))
        self.assertEqual(list(here.text), ["0.90", "0.30"])
        self.assertEqual(list(figure.layout.yaxis.categoryarray), ["102004", "103002", "102001", "102003"])  # best on top
        self.assertEqual(figure.layout.xaxis.title.text, "cosine")
        other = figures.neighbours_figure([("a", 0.5, "deck 04")])
        self.assertEqual([trace.name for trace in other.data], [figures.ELSEWHERE])  # an unknown place counts as elsewhere


if __name__ == "__main__":
    unittest.main()
