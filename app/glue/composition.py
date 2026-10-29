"""The composition (glue): the models behind the Helper button on the viewer's app (used by the launcher's
composition root, modules/helper/delta/cards.py, and the tests).

A fresh start is the viewer's own: every deck and topic checked, the course file's scripted topics, random
positions, no connections and no generated groups. The models are fitted in the serving process at every start
(serving_process), and the Helper button (button.Helper, the viewer's module action left of Random, with its dialog of
the models) lets them work on the level the user is on. The model state (derived.MODEL_STATE_NAME, next to the
derived course file) holds which decks have the model's topics, the difficulty groups and the model's connections:
read at a start with a stored card state, so a restart reads the same topics, the difficulty groups and the model's
connections come back, and so do the presets and Display's check of the model's connections in the views that hold
them (both runtime only, like the viewer's cut-off line); forgotten at a fresh card state.

The viewer's hooks (2026-10-28, roadmap step 21.2.1; until then the helper's viewer part in modules/helper/delta/
did their work): compose takes the Helper's refill of its generated groups (refill_groups) and its write of the
model state (after_save); the groups apart, the read only generated groups and the navigation rules are the
viewer's own, and the Helper button reloads the viewer's topic index (TopicIndex.reload).

The difficulty model D1 (models.difficulty_model) is fitted with the course model, and its metrics are the back of
the card panel (the viewer's overview_back, difficulty_metrics): with nothing selected, Enter or a click on the
panel turns it over, to the metrics of the course in the decks view, of the entered deck in the topics view, and
of the entered topic's counts with the deck's metrics in the cards view (metrics_back).

With Dash's reloader the launcher's parent process composes too, but never serves and never saves, so it composes
the viewer's app on the raw course file without the models.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from project.composition import Composition, compose
from project.services.actions import ModuleAction
from project.services.views import DECKS, TOPICS
from project.store import state as state_store

from ..models.course_model import CourseModel
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from .button import Helper
from .course import load_course
from .derived import DERIVED_NAME, MODEL_STATE_NAME, read_model_state
from .difficulty_metrics import metrics_panel


@dataclass(frozen=True)
class ModelComposition:
    composition: Composition
    model: Optional[CourseModel]  # None when the model was not fitted (the reloader's parent process)
    fresh_cards: bool  # the state file held no card state before composing
    helper: Optional[Helper] = None  # the Helper button's models, with the model
    difficulty: Optional[DifficultyReport] = None  # D1's metrics and predictions, with the model
    derived_path: Optional[Path] = None  # the derived course file compose read (None without the model)

    @property
    def action(self) -> Optional[ModuleAction]:
        return self.helper.action if self.helper is not None else None

    @property
    def user_state(self):
        return self.composition.user_state

    @property
    def dash_app(self):
        return self.composition.dash_app

    @property
    def core(self):
        return self.composition.core

    @property
    def decks_core(self):
        return self.composition.space_core

    @property
    def topics(self):
        return self.composition.topics


def serving_process(reloader: bool = True, environ: Mapping[str, str] = os.environ) -> bool:
    """Whether this process serves the app. With Dash's reloader (app.run(debug=True)) the parent process only
    watches for changes and the child (WERKZEUG_RUN_MAIN) serves; without the reloader, this process serves."""
    return not reloader or environ.get("WERKZEUG_RUN_MAIN") == "true"


def is_fresh(state_path: Path) -> bool:
    """Whether the state file holds no card state yet. Asked through the viewer's document, not the path: the
    launcher redirects the document by the requested file name to its temp/ folder, where the file may not exist
    even though the requested path (data/user_state.json of the main project) does."""
    return state_store.StateDocument(state_path).section(state_store.CARDS_SECTION).read() is None


def build_model(cards_path: Path, vectoriser=None) -> CourseModel:
    """Fit the course model on every card of the course file, seeded by its scripted topics (course.load_course).
    The file is version 3: the viewer's parser refuses anything else."""
    course = load_course(cards_path)
    return CourseModel(course.cards, course.deck_ids, course.seeds, vectoriser)


def build_difficulty(cards_path: Path, model: CourseModel) -> DifficultyReport:
    """Measure and fit D1 on the course file's cards, with the course model's similarity index."""
    return fit_difficulty(load_course(cards_path).cards, model.index)


def compose_model(
    base_dir: Path,
    boot_id: str,
    cards_path: Path,
    state_path: Path,
    fit: bool = True,
    model: Optional[CourseModel] = None,
    difficulty: Optional[DifficultyReport] = None,
    derived_path: Optional[Path] = None,
) -> ModelComposition:
    """Compose the viewer's app on the derived course file with the Helper button. A fresh card state starts as
    the viewer's (the scripted topics, random positions, nothing generated); a stored one reads the model's
    topics of the decks that had them, the difficulty groups and the model's connections, with the cut-off presets
    and Display's check where those connections are. With fit=False and no
    model, the app is the viewer's on the raw course file (no Helper button, no back of the card panel), and
    nothing is written. D1 is fitted with the model unless given. `state_path` is the viewer's one state file;
    `derived_path` is where the derived course file and the model state go (the launcher's cards.py passes its
    temp/); by default next to the requested state file."""
    derived_path = Path(derived_path) if derived_path is not None else Path(state_path).with_name(DERIVED_NAME)
    fresh_cards = is_fresh(state_path)
    if fit and model is None:
        model = build_model(cards_path)
    if model is None:
        composition = compose(base_dir, boot_id, cards_path, state_path)
        return ModelComposition(composition=composition, model=None, fresh_cards=fresh_cards)
    if difficulty is None:
        difficulty = build_difficulty(cards_path, model)
    model_state = derived_path.with_name(MODEL_STATE_NAME)
    helper = Helper(model, difficulty, cards_path, derived_path, model_state, {} if fresh_cards else read_model_state(model_state))
    helper.write_course()
    composition = compose(
        base_dir, boot_id, derived_path, state_path,
        card_positions=helper.card_positions, action=helper.action,
        overview_back=lambda state: metrics_back(state, difficulty),
        refill_groups=helper.refill_groups, after_save=helper.write_state,
    )
    router = composition.user_state
    helper.load_groups(router)  # the stored difficulty groups, the empty ones filled again
    helper.load_links(router)  # the model's stored connections, cut to the loaded dots
    helper.preset_cutoffs(router)  # their views' presets and Display's check again
    return ModelComposition(
        composition=composition, model=model, fresh_cards=fresh_cards, helper=helper, difficulty=difficulty,
        derived_path=derived_path,
    )


def metrics_back(state, difficulty: DifficultyReport):
    """The back of the card panel: D1's metrics of the course in the decks view, of the entered deck in the topics
    view, and of the entered topic (its counts and predictions with the deck's metrics) in the cards view."""
    if state.view == DECKS:
        return metrics_panel(difficulty)
    if state.view == TOPICS:
        return metrics_panel(difficulty, state.entered_name)
    return metrics_panel(difficulty, state.entered_name, state.entered_topic_info)
