"""The composition (glue): the models behind the Helper button on the viewer's app (used by start.py, by the
composition root of the flashcards workspace's module launcher, and by the tests).

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

The difficulty model D1 (models.difficulty_model) is fitted with the course model for the Helper's difficulty
groups and the card section's ranking; its metrics are the standalone page's (the Models button), not the viewer's:
the card panel has no back here. The viewer's newer hooks are used where the viewer has them (viewer_hooks: the top
bar links since sputnik 0.4.0, the card section and the focus slider since 0.5.0, the Ask dialog and the episodes'
hook since 0.7.0; 0.3.0 has none): links are passed where the top bar can take them, and the card section
(glue.related: the selected card's neighbours with the next card first), the margin slider (glue.focus: T1's margins
on the share scale) and the Ask dialog (glue.prompt: P1's answer as the viewer's targets) go where the viewer draws
them; the episodes' log (glue.episodes: every ended episode of the dialog or the card section, a row of episodes.csv
next to the model state) where the viewer reports them. P1's model (models.prompt_model) is fitted only for the
dialog, after the course model, and kept in memory with it.

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
from project.store import state as state_store

from ..models import similar
from ..models.course_model import CourseModel
from ..models.difficulty_model import DifficultyReport, fit_difficulty
from ..models.prompt_model import PromptModel
from .button import Helper
from .course import load_course
from .derived import DERIVED_NAME, MODEL_STATE_NAME, read_model_state
from .episodes import EPISODES_NAME, helper_episodes
from .focus import helper_focus
from .prompt import helper_prompt
from .related import helper_section

LINKS_HOOK = "links"  # compose's top bar links (services.actions.TopBarLink)
SECTION_HOOK = "card_section"  # compose's card section under the card panel (services.actions.ModuleSection)
FOCUS_HOOK = "focus"  # compose's focus slider of the cards view (services.actions.ModuleFocus)
PROMPT_HOOK = "prompt"  # compose's Ask dialog of the top bar (services.actions.ModulePrompt)
EPISODE_HOOK = "after_episode"  # compose's report of an ended episode (services.episode.Episode)
NEWER_HOOKS = (LINKS_HOOK, SECTION_HOOK, FOCUS_HOOK, PROMPT_HOOK, EPISODE_HOOK)
MODELS_URL = "HELPER_MODELS_URL"  # the launchers' variable with the standalone page's URL (app.launch.MODELS_URL)
MODELS_LABEL = "Models"
MODELS_TITLE = "The models' page, served next to this viewer: the model panels with the offline results, in a new tab"


def viewer_hooks(compose_function: Optional[Callable] = None) -> set[str]:
    """The newer hooks this viewer's compose takes (this module's `compose` unless one is given), among NEWER_HOOKS
    (none on sputnik 0.3.0; the links since 0.4.0, the section and the focus since 0.5.0, the prompt and the episodes'
    hook since 0.7.0)."""
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
    links: tuple = ()  # the top bar links passed to compose (none where the viewer has no hook)
    card_section: Optional[object] = None  # the card section passed to compose (the related cards), where the viewer has the hook
    focus: Optional[object] = None  # the margin slider passed to compose, where the viewer has the hook
    prompt: Optional[object] = None  # the Ask dialog passed to compose, where the viewer has the hook
    after_episode: Optional[Callable] = None  # the episodes' log passed to compose, where the viewer has the hook

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
    """Whether the state file holds no card state yet. Asked through the viewer's document, not the path: a module
    launcher redirects the document by the requested file name to its own folder, where the file may not exist
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
    model, the app is the viewer's on the raw course file (no Helper button), and
    nothing is written. D1 is fitted with the model unless given. `state_path` is the viewer's one state file;
    `derived_path` is where the derived course file and the model state go (a module launcher's composition root
    passes its own folder); by default next to the requested state file. `links` are the viewer's TopBarLinks for its top bar
    (the launcher's link to the models' page), passed where the viewer's compose takes them. `vectoriser` and
    `min_score` are the index's when the model is fitted here (the app's vectoriser from the launchers, at its own
    minimum; TF-IDF at the fixed minimum by default). The episodes' log goes next to the model state."""
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
    section = helper_section(model, helper.questions, difficulty.scores) if SECTION_HOOK in hooks else None
    focus = helper_focus(helper) if FOCUS_HOOK in hooks else None
    prompt = helper_prompt(PromptModel(model.index, model.topic_model, course.cards)) if PROMPT_HOOK in hooks else None
    after_episode = helper_episodes(derived_path.with_name(EPISODES_NAME)) if EPISODE_HOOK in hooks else None
    if section is not None:
        extra[SECTION_HOOK] = section
    if focus is not None:
        extra[FOCUS_HOOK] = focus
    if prompt is not None:
        extra[PROMPT_HOOK] = prompt
    if after_episode is not None:
        extra[EPISODE_HOOK] = after_episode
    composition = compose(
        base_dir, boot_id, derived_path, state_path,
        card_positions=helper.card_positions, action=helper.action,
        refill_groups=helper.refill_groups, after_save=helper.write_state, **extra,
    )
    router = composition.user_state
    helper.load_groups(router)  # the stored difficulty groups, the empty ones filled again
    helper.load_links(router)  # the model's stored connections, cut to the loaded dots
    helper.preset_cutoffs(router)  # their views' presets and Display's check again
    return ModelComposition(
        composition=composition, model=model, fresh_cards=fresh_cards, helper=helper, difficulty=difficulty,
        derived_path=derived_path, links=tuple(extra.get(LINKS_HOOK, ())),
        card_section=section, focus=focus, prompt=prompt, after_episode=after_episode,
    )

