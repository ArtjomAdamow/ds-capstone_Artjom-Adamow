"""The Helper (glue): the models' results for the whole course, handed to the viewer once at the start, and the
viewer's switch that shows or hides the model's connections.

Roadmap step 33 of the helper: the models are on from the start. At every start, after the fit, the Helper gives every
deck the model's topics (the derived course file, derived.write_derived_course, written for every deck) and makes the
model state of the whole course (model_state), closed decks included:

  * the positions: the decks at the model's points on the unit sphere, every topic's cards on the surface of its
    sphere (the topic dots keep the viewer's positions);
  * the connections in merge order: the decks', every deck's topics' (the clustering of the topic vectors of the
    model's membership, n - 1 per deck, none for an emptied topic; no preset, the full tree, shown in the topics view's
    dendrogram and not on its map) and every topic's cards' (n - 1 per topic with two cards or more);
  * the presets of the cut-off line: the decks view's (no connected part above five decks) and every topic's (four
    cards), each over all the view's objects, since the viewer's checks only show and hide;
  * the difficulty groups: every rated card in its difficulty's group, every quiz D1 predicted in "Quiz difficulty:
    predicted" and in its predicted level's group (difficulty_groups), the oldest groups.

The viewer takes it through compose's `model_state` hook (services.actions.ModelState, imported here, not with the
module, since a viewer older than sputnik 0.10.0 has none) and holds it for every deck; the model's connections are
its own (LINK_ORIGIN, the viewer's CardLink.origin), read only, drawn gold and dashed, never saved with the user's
state. The Helper's switch (the viewer's ModuleAction, its label alone) stands in the Connections panel next to
Display: it shows or hides the model's connections in every view, hidden at every start, and showing them sets every
view's preset. The same results go into the model state file (derived.write_model_state, next to the derived file),
written at every start after the fit and read by nothing: the record of what the viewer got. Until step 33 the
Helper button in the top bar ran the models per view from its dialog (dialog.py), and the model state was read back
at a restart.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

from project.services.actions import ModuleAction

from ..cards import Card
from ..models.course_model import CourseModel
from ..models.difficulty_model import DifficultyReport
from .cutoff import CARD_PART, DECK_PART, preset_cutoff
from .derived import COURSE_MODEL, DIFFICULTY_MODEL, write_derived_course, write_model_state
from .difficulty_groups import fill_difficulty_groups, fill_predicted_groups

HELPER_LABEL = "Helper"
LINK_ORIGIN = "course_model"  # the maker of the model's connections (CardLink.origin), as the groups' "difficulty_model"


class Helper:
    """The models' results for the viewer: the derived course file with the model's topics for every deck, the model
    state of the whole course (model_state) with its file, the course's questions for the card section, and the
    switch's label."""

    def __init__(
        self,
        model: CourseModel,
        difficulty: Optional[DifficultyReport],
        cards_path: Path,
        derived_path: Path,
        model_state_path: Path,
        cards: Sequence[Card] = (),
    ) -> None:
        """`cards` are the course's records (the difficulty groups' members; the card section's labels: the questions
        by card id)."""
        self.model = model
        self.difficulty = difficulty
        self.cards = list(cards)
        self.questions: dict[str, str] = {card.id: card.question for card in self.cards}
        self.cards_path = Path(cards_path)
        self.derived_path = Path(derived_path)
        self.model_state_path = Path(model_state_path)

    @property
    def action(self) -> ModuleAction:
        """The viewer's switch of the model's connections: the label alone (a viewer older than sputnik 0.10.0 gets
        no action from the composition)."""
        return ModuleAction(HELPER_LABEL)

    def write_course(self) -> Path:
        """Write the derived course file: the model's topics for every deck."""
        write_derived_course(
            self.cards_path, self.derived_path, self.model.topic_model.topic_of, decks=sorted(self.model.deck_points),
        )
        return self.derived_path

    def model_state(self):
        """The models' results for the whole course, as the viewer takes them (services.actions.ModelState): the
        positions, the connections, the presets and the difficulty groups."""
        from project.services.actions import ModelState

        model = self.model
        topics = list(model.topics.values())
        return ModelState(
            origin=LINK_ORIGIN,
            deck_positions=dict(model.deck_points),
            card_positions=dict(model.card_points),
            deck_links=list(model.deck_links),
            topic_links={deck_id: model.topic_links(deck_id) for deck_id in sorted(model.deck_points)},
            card_links=[pair for placed in topics for pair in placed.links],
            deck_preset=preset_cutoff(sorted(model.deck_points), model.deck_links, DECK_PART),
            topic_presets={
                (placed.deck_id, placed.topic_id): preset_cutoff(placed.ids, placed.links, CARD_PART)
                for placed in topics if placed.links
            },
            groups=self.groups(),
        )

    def groups(self) -> list:
        """The difficulty groups of the whole course, the oldest first: the rated cards by their difficulty, the quizzes
        D1 predicted by their predicted level and in "Quiz difficulty: predicted"."""
        scratch = SimpleNamespace(groups=[])
        fill_difficulty_groups(scratch, self.cards)
        if self.difficulty is not None:
            fill_predicted_groups(scratch, self.difficulty.predictions)
        return scratch.groups

    def write_state(self, state) -> None:
        """Store the model state file: one entry per model, the course model's positions, connections and presets, and
        D1's predicted levels (`state` is the ModelState handed to the viewer)."""
        write_model_state(self.model_state_path, {
            COURSE_MODEL: {
                "positions": {
                    "decks": {deck_id: list(point) for deck_id, point in state.deck_positions.items()},
                    "cards": {card_id: list(point) for card_id, point in state.card_positions.items()},
                },
                "links": {
                    "decks": [list(pair) for pair in state.deck_links],
                    "topics": {deck_id: [list(pair) for pair in pairs] for deck_id, pairs in state.topic_links.items()},
                    "cards": [list(pair) for pair in state.card_links],
                },
                "presets": {
                    "decks": state.deck_preset,
                    "topics": [[deck_id, topic_id, share] for (deck_id, topic_id), share in state.topic_presets.items()],
                },
            },
            DIFFICULTY_MODEL: {"levels": dict(self.difficulty.predictions) if self.difficulty is not None else {}},
        })
