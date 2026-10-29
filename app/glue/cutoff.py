"""The cut-off preset (glue; no Dash, no state): where the project's cut-off line starts after the Helper button, so
that the connected parts below it stay small enough to read.

The viewer replays a view's shown connections in their order (project.services.dendrogram, a module's first): a
connection that joins two separate parts is a merge with its rank (1, 2, 3, ...), and the cut-off line keeps the
merges up to a share of the merge count. The preset is measured over the model's connections alone (2026-10-29,
roadmap step 22; until then over every stored connection of the view): the model's pairs among the view's dots, in
merge order, are a forest, so each is a merge, and the preset is the share of them kept until the next would make a
part larger than the size (models.clustering.preset_rank). On them the preset cuts the clustering tree at a height;
the line then cuts the shown replay. A tree within the size is kept whole (1).
"""

from typing import Iterable

from ..models.clustering import preset_rank

FULL_TREE = 1.0
DECK_PART = 5  # the decks view: parts of at most five decks
CARD_PART = 4  # a topic's cards: parts of at most four cards


def preset_cutoff(dot_ids: Iterable[str], pairs: Iterable[tuple[str, str]], size: int) -> float:
    """The highest cut-off share (0 to 1) whose parts among `dot_ids` hold at most `size` dots, over the model's
    pairs (in merge order) with both ends among the dots; 1 without such a pair."""
    dots = set(dot_ids)
    merges = [(source, target) for source, target in pairs if source != target and source in dots and target in dots]
    if not merges:
        return FULL_TREE
    return preset_rank(merges, size) / len(merges)
