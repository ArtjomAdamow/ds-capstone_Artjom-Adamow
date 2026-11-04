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

The viewer's hooks: compose takes the Helper's refill of its generated groups (refill_groups) and its write of the
model state (after_save); the groups apart, the read only generated groups and the navigation rules are the
viewer's own, and the Helper button reloads the viewer's topic index (TopicIndex.reload).

The difficulty model D1 (models.difficulty_model) is fitted with the course model, and its metrics are the back of
the card panel (the viewer's overview_back, difficulty_metrics): with nothing selected, Enter or a click on the
panel turns it over, to the scores of the course in the decks view, of the entered deck in the topics view, and
of the entered topic's counts with the deck's metrics in the cards view (metrics_back); the confusion matrix and
the shares of the levels go to the window under the card's window (the viewer's overview_window hook;
window_back). The viewer's newer hooks are used where the viewer has them (viewer_hooks: the window and
the top bar links since sputnik 0.4.0, the card section and the focus slider since 0.5.0, the capstone's pin; 0.3.0 has
none): on a viewer without the window the back keeps the figures, links are passed where the top bar can take them,
and the card section (glue.related: the selected card's neighbours with the next card first) and the margin slider
(glue.focus: T1's margins on the share scale) go where the viewer draws them.

The Models button (models_links): a link to the standalone page's URL that the launcher put into the environment
(MODELS_URL, the same name in both launchers), passed as a TopBarLink where the viewer takes links; none without the
URL.

With Dash's reloader the launcher's parent process composes too, but never serves and never saves, so it composes
the viewer's app on the raw course file without the models.
"""

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from project.composition import Composition, compose
from project.services.actions import ModuleAction
from project.services.views import DECKS, TOPICS
from project.store import state as state_store

from ..models import similar
from ..models.course_model import CourseModel
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from .button import Helper
from .course import load_course
from .derived import DERIVED_NAME, MODEL_STATE_NAME, read_model_state
from .difficulty_metrics import metrics_panel, metrics_window
from .focus import helper_focus
from .related import helper_section

LINKS_HOOK = "links"  # compose's top bar links (services.actions.TopBarLink)
WINDOW_HOOK = "overview_window"  # compose's window under the card's window
SECTION_HOOK = "card_section"  # compose's card section under the card panel (services.actions.ModuleSection)
FOCUS_HOOK = "focus"  # compose's focus slider of the cards view (services.actions.ModuleFocus)
NEWER_HOOKS = (LINKS_HOOK, WINDOW_HOOK, SECTION_HOOK, FOCUS_HOOK)
MODELS_URL = "HELPER_MODELS_URL"  # the launchers' variable with the standalone page's URL (app.launch.MODELS_URL)
MODELS_LABEL = "Models"
MODELS_TITLE = "The models' page, served next to this viewer: the model panels with the offline results, in a new tab"


def viewer_hooks(compose_function: Optional[Callable] = None) -> set[str]:
    """The newer hooks this viewer's compose takes (this module's `compose` unless one is given), among NEWER_HOOKS
    (none on sputnik 0.3.0; the links and the window since 0.4.0, the section and the focus since 0.5.0)."""
    return set(NEWER_HOOKS) & set(inspect.signature(compose_function or compose).parameters)


def models_links(environ: Mapping[str, str] = os.environ, compose_function: Optional[Callable] = None) -> list:
    """The Models button: a TopBarLink to the standalone page's URL the launcher put into the environment; none
    without it (the capstone's launcher sets it in full mode too, since step 28), and none where the viewer's
    compose takes no links (older than sputnik 0.4.0, which has no TopBarLink either)."""
    url = environ.get(MODELS_URL, "")
    if not url or LINKS_HOOK not in viewer_hooks(compose_function):
        return []
    from project.services.actions import TopBarLink

    return [TopBarLink(MODELS_LABEL, url, MODELS_TITLE)]


@dataclass(frozen=True)
class ModelComposition:
    composition: Composition
    model: Optional[CourseModel]  # None when the model was not fitted (the reloader's parent process)
    fresh_cards: bool  # the state file held no card state before composing
    helper: Optional[Helper] = None  # the Helper button's models, with the model
    difficulty: Optional[DifficultyReport] = None  # D1's metrics and predictions, with the model
    derived_path: Optional[Path] = None  # the derived course file compose read (None without the model)
    overview_back: Optional[Callable] = None  # the back of the card panel, as passed to compose (with the model)
    overview_window: Optional[Callable] = None  # the window under it, where the viewer has the hook
    links: tuple = ()  # the top bar links passed to compose (none where the viewer has no hook)
    card_section: Optional[object] = None  # the card section passed to compose (the related cards), where the viewer has the hook
    focus: Optional[object] = None  # the margin slider passed to compose, where the viewer has the hook

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


def build_model(cards_path: Path, vectoriser=None, min_score: Optional[float] = similar.MIN_SCORE) -> CourseModel:
    """Fit the course model on every card of the course file, seeded by its scripted topics (course.load_course).
    The file is version 3: the viewer's parser refuses anything else. `vectoriser` is the index's (TF-IDF by
    default; the app's, embeddings.app_vectoriser, from the launchers), `min_score` the neighbours' minimum (None
    for the index's own)."""
    course = load_course(cards_path)
    return CourseModel(course.cards, course.deck_ids, course.seeds, vectoriser, min_score=min_score)


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
    links: Sequence = (),
    vectoriser=None,
    min_score: Optional[float] = similar.MIN_SCORE,
) -> ModelComposition:
    """Compose the viewer's app on the derived course file with the Helper button. A fresh card state starts as
    the viewer's (the scripted topics, random positions, nothing generated); a stored one reads the model's
    topics of the decks that had them, the difficulty groups and the model's connections, with the cut-off presets
    and Display's check where those connections are. With fit=False and no
    model, the app is the viewer's on the raw course file (no Helper button, no back of the card panel), and
    nothing is written. D1 is fitted with the model unless given. `state_path` is the viewer's one state file;
    `derived_path` is where the derived course file and the model state go (the launcher's cards.py passes its
    temp/); by default next to the requested state file. `links` are the viewer's TopBarLinks for its top bar
    (the launcher's link to the models' page), passed where the viewer's compose takes them. `vectoriser` and
    `min_score` are the index's when the model is fitted here (the app's vectoriser from the launchers, at its own
    minimum; TF-IDF at the fixed minimum by default)."""
    derived_path = Path(derived_path) if derived_path is not None else Path(state_path).with_name(DERIVED_NAME)
    fresh_cards = is_fresh(state_path)
    hooks = viewer_hooks()
    extra = {LINKS_HOOK: list(links)} if links and LINKS_HOOK in hooks else {}
    course = load_course(cards_path) if fit or model is not None else None  # read once: the model, D1 and the section's labels
    if fit and model is None:
        model = CourseModel(course.cards, course.deck_ids, course.seeds, vectoriser, min_score=min_score)
    if model is None:
        composition = compose(base_dir, boot_id, cards_path, state_path, **extra)
        return ModelComposition(composition=composition, model=None, fresh_cards=fresh_cards, links=tuple(extra.get(LINKS_HOOK, ())))
    if difficulty is None:
        difficulty = fit_difficulty(course.cards, model.index)
    model_state = derived_path.with_name(MODEL_STATE_NAME)
    helper = Helper(
        model, difficulty, cards_path, derived_path, model_state, {} if fresh_cards else read_model_state(model_state), course.cards,
    )
    helper.write_course()
    windowed = WINDOW_HOOK in hooks
    back = lambda state: metrics_back(state, difficulty, figures=not windowed)  # noqa: E731
    window = (lambda state: window_back(state, difficulty)) if windowed else None
    if windowed:
        extra[WINDOW_HOOK] = window
    section = helper_section(model, helper.questions, difficulty.scores) if SECTION_HOOK in hooks else None
    focus = helper_focus(helper) if FOCUS_HOOK in hooks else None
    if section is not None:
        extra[SECTION_HOOK] = section
    if focus is not None:
        extra[FOCUS_HOOK] = focus
    composition = compose(
        base_dir, boot_id, derived_path, state_path,
        card_positions=helper.card_positions, action=helper.action, overview_back=back,
        refill_groups=helper.refill_groups, after_save=helper.write_state, **extra,
    )
    router = composition.user_state
    helper.load_groups(router)  # the stored difficulty groups, the empty ones filled again
    helper.load_links(router)  # the model's stored connections, cut to the loaded dots
    helper.preset_cutoffs(router)  # their views' presets and Display's check again
    return ModelComposition(
        composition=composition, model=model, fresh_cards=fresh_cards, helper=helper, difficulty=difficulty,
        derived_path=derived_path, overview_back=back, overview_window=window, links=tuple(extra.get(LINKS_HOOK, ())),
        card_section=section, focus=focus,
    )


def _scope(state) -> tuple:
    """(deck, topic) of the view for the metrics: none in the decks view, the entered deck in the topics view, the
    entered deck and topic in the cards view."""
    if state.view == DECKS:
        return None, None
    if state.view == TOPICS:
        return state.entered_name, None
    return state.entered_name, state.entered_topic_info


def metrics_back(state, difficulty: DifficultyReport, figures: bool = True):
    """The back of the card panel: D1's scores of the course in the decks view, of the entered deck in the topics
    view, and of the entered topic (its counts and predictions with the deck's metrics) in the cards view; with
    `figures` the matrix and the shares too (a viewer without the window under the card's window)."""
    return metrics_panel(difficulty, *_scope(state), figures=figures)


def window_back(state, difficulty: DifficultyReport):
    """The window under the card's window while the back is up: D1's confusion matrix and the shares of the levels
    of the view's part."""
    return metrics_window(difficulty, *_scope(state))
