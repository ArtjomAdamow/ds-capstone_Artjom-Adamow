"""The Helper button (glue): the viewer's module action, left of Random. It opens a dialog under itself (2026-10-29,
roadmap step 22; until then one click ran every model at once): the viewer draws the models the view offers from
Helper.options (the catalogue, dialog.py), and Confirm runs the checked ones on the level the user is on and its
sublevels (Helper.run), in the run order Topics, Positions, Connections, Difficulty:
  * decks view: Topics gives every deck the model's topics; Positions places the decks and every checked topic's
    cards; Connections writes the decks' connections and every checked topic's, with their cut-off presets (no
    connected part above five decks, four cards); Difficulty fills the difficulty groups of every loaded card;
  * topics view: the same for the entered deck and its checked topics (the topic dots keep the viewer's positions,
    and topic connections are roadmap step 19);
  * cards view: Positions, Connections and Difficulty for the entered topic's cards. A scripted topic is laid out on
    the spot.
Each model runs alone (give_model_topics, place_cards, connect_cards, fill_groups): Topics alone makes the new dots
and leaves the cards' positions and connections as they are (the card ids do not change). run(state) without the
checked list runs every model the view offers, the one click of a viewer without the dialog's hook.

Giving a deck the model's topics (T1, models.topic_model): the derived course file (derived.write_derived_course)
is written again with the model's topics for those decks, the viewer's topic index is rebuilt from it in place
(TopicIndex.reload), and the deck's dots are reloaded, so the dialogs, the predefined groups and the state follow
the new topics. Every card of the deck is shown again (the topics are new).

The model's connections are its own (LINK_ORIGIN, the viewer's CardLink.origin), apart from the user's, like the
generated groups: a run replaces the model's earlier connections among the objects it ran on with the fresh ones in
merge order (set_links), whether or not the user connected a pair too, and never touches the user's. The viewer saves
the user's only, keeps the model's read only and draws them gold and dashed; every view the run connected shows them
(Display's "Helper" checked there, next to the view's cut-off preset, measured over the model's pairs alone). What
exists already is kept: the user's connections and groups stay; the generated groups become the oldest groups again.

The model state (derived.write_model_state) holds which decks have the model's topics (the topic model's entry), the
difficulty groups with their members (the difficulty model's entry) and the model's connections by state (the course
model's entry); the Helper writes it with the derived file and after every save of the router (the viewer's
after_save hook), so the file mirrors memory. At a start with a stored card state the difficulty groups and the
connections come back from it (load_groups, load_links), and the views that hold the model's connections get their
presets and Display's check again (preset_cutoffs).
"""

from dataclasses import asdict
from pathlib import Path
from typing import Collection, Iterable, Mapping, Optional

from project.content.parser import CardsFile
from project.content.topics import TopicInfo
from project.models import CardLink
from project.services.actions import ActionOption, ModuleAction
from project.services.views import CARDS, DECKS, TOPICS

from ..cards import Card
from ..models.course_model import CourseModel, TopicLayout
from ..models.difficulty_model import DifficultyReport
from . import positions
from .cutoff import CARD_PART, DECK_PART, preset_cutoff
from .derived import (
    COURSE_MODEL, DIFFICULTY_MODEL, LINK_STATES, TOPIC_MODEL, stored_links, write_derived_course, write_model_state,
)
from .dialog import MODEL_CONNECTIONS, MODEL_DIFFICULTY, MODEL_POSITIONS, MODEL_TOPICS, OFFERED, options_for, run_order
from .difficulty_groups import (
    fill_difficulty_groups, fill_predicted_groups, group_records, refill_empty, restore_groups, stored_groups,
)

HELPER_LABEL = "Helper"
HELPER_TITLE = (
    "Choose the models to run on this view: in the decks view the model's topics, positions, connections and "
    "difficulty groups for every deck with its topics and cards; in a deck the same for its topics and cards; in a "
    "topic its cards' positions, connections and difficulty groups"
)
LINK_ORIGIN = "course_model"  # the maker of the model's connections (CardLink.origin), as the groups' "difficulty_model"


class Helper:
    """The models behind the Helper button: the dialog's options per view (options), the checked models per view
    (run), the decks with the model's topics, the derived course file, the model state and the card state's position
    source."""

    def __init__(
        self,
        model: CourseModel,
        difficulty: Optional[DifficultyReport],
        cards_path: Path,
        derived_path: Path,
        model_state_path: Path,
        stored: Optional[Mapping[str, Mapping]] = None,
    ) -> None:
        """`stored` is the model state as read (derived.read_model_state), one mapping per model; none at a fresh
        start."""
        self.model = model
        self.difficulty = difficulty
        self.cards_path = Path(cards_path)
        self.derived_path = Path(derived_path)
        self.model_state_path = Path(model_state_path)
        stored = stored or {}
        decks = stored.get(TOPIC_MODEL, {}).get("decks")
        known = set(model.deck_points)
        self.model_decks: set[str] = {str(deck_id) for deck_id in decks if str(deck_id) in known} if isinstance(decks, list) else set()
        # The difficulty groups and the model's connections as stored, until load_groups and load_links put them
        # into the states
        self.stored_groups = stored_groups(stored.get(DIFFICULTY_MODEL, {}).get("groups"))
        self.stored_links = stored_links(stored.get(COURSE_MODEL, {}))
        self.action = ModuleAction(HELPER_LABEL, self.run, HELPER_TITLE, options=self.options)

    # --- files and sources ---------------------------------------------------

    def write_course(self, state=None) -> Path:
        """Write the derived course file (the model's topics for the decks that have them, the scripted ones
        elsewhere) and the model state (write_state)."""
        topic_of = {
            card_id: topic for card_id, topic in self.model.topic_model.topic_of.items()
            if self.model.deck_of.get(card_id) in self.model_decks
        }
        write_derived_course(self.cards_path, self.derived_path, topic_of, decks=sorted(self.model_decks))
        self.write_state(state)
        return self.derived_path

    def write_state(self, state=None) -> None:
        """Store the model state: the decks with the model's topics, the difficulty groups of the card state and the
        model's connections of both states (`state` is the ViewRouter: the after_save hook) or, before there is
        one, as stored."""
        if state is None:
            groups = [asdict(group) for group in self.stored_groups]
            links = {name: list(self.stored_links[name]) for name in LINK_STATES}
        else:
            groups = group_records(state.cards_state)
            links = {"decks": model_pairs(state.decks_state), "cards": model_pairs(state.cards_state)}
        write_model_state(self.model_state_path, {
            TOPIC_MODEL: {"decks": sorted(self.model_decks), "groups": []},
            DIFFICULTY_MODEL: {"groups": groups},
            COURSE_MODEL: {"links": links},
        })

    def load_groups(self, state) -> None:
        """At the start: the stored difficulty groups into the card state, their members cut to the loaded cards,
        as the oldest groups; then the ones left empty are filled again."""
        restore_groups(state.cards_state, self.stored_groups, state.cards_state.card_by_id)
        self.refill_groups(state)

    def load_links(self, state) -> None:
        """At the start: the stored connections of the model into the decks state and the card state, as the
        model's, cut to the loaded dots (as the groups are), in their stored order."""
        for core, name in ((state.decks_state, "decks"), (state.cards_state, "cards")):
            loaded = core.card_by_id
            core.links.extend(
                CardLink(source, target, LINK_ORIGIN) for source, target in self.stored_links[name]
                if source != target and source in loaded and target in loaded
            )

    def refill_groups(self, state) -> int:
        """Fill the generated groups left empty again from the loaded cards, rated and predicted."""
        cards = [card for card in state.cards_state.cards if not state.is_topic(card.id)]
        predictions = self.difficulty.predictions if self.difficulty is not None else {}
        return refill_empty(state.cards_state, cards, predictions)

    def card_positions(self, ids, seed: int = 42):
        """The card state's position source: the model's position for a card of a deck with the model's topics
        (a deck checked again gets them back), the viewer's random one on the surface for any other."""
        return positions.card_positions(self.model, ids, seed, decks=self.model_decks)

    def preset_cutoffs(self, router) -> None:
        """At a start with the model state: in every view that holds the model's connections (the decks view, a
        checked topic), its cut-off preset over the model's pairs and Display's check of them again, as the run
        that wrote them set them."""
        deck_pairs = model_pairs(router.decks_state)
        if deck_pairs:
            router.cutoffs[DECKS] = preset_cutoff(self._checked_deck_ids(router), deck_pairs, DECK_PART)
            router.show_origin(DECKS, LINK_ORIGIN)
        card_pairs = model_pairs(router.cards_state)
        if not card_pairs:
            return
        for deck_id in sorted(info.deck_id for info in router.decks.infos):
            for topic in self._checked_topics(router, deck_id):
                members = set(topic.card_ids)
                pairs = [pair for pair in card_pairs if pair[0] in members and pair[1] in members]
                if pairs:
                    router.cutoffs[topic_key(topic)] = preset_cutoff(shown_ids(router, topic), pairs, CARD_PART)
                    router.show_origin(topic_key(topic), LINK_ORIGIN)

    # --- the button and its dialog ---------------------------------------------

    def options(self, state) -> list[ActionOption]:
        """The dialog's checklist on the router's view: the models it offers, every box checked (dialog.py)."""
        return options_for(state.view)

    def run(self, state, checked: Optional[Iterable[str]] = None) -> int:
        """The checked models on the current view and its sublevels (`state` is the ViewRouter), in the run order;
        None (a click without the dialog) runs every model the view offers. Returns the connections and memberships
        it wrote."""
        models = run_order(state.view, OFFERED.get(state.view, ()) if checked is None else checked)
        if state.view == DECKS:
            return self.run_decks(state, models)
        if state.view == TOPICS:
            return self.run_deck(state, models)
        return self.run_topic(state, models)

    def run_decks(self, state, models: Collection[str]) -> int:
        """The decks view: every deck, and every checked topic of the checked decks (the sublevels)."""
        written = 0
        if MODEL_TOPICS in models:
            self.give_model_topics(state, self.model.deck_points)
        topics = [topic for deck_id in sorted(self.model.deck_points) for topic in self._checked_topics(state, deck_id)]
        if MODEL_POSITIONS in models:
            self.place_decks(state)
            for topic in topics:
                self.place_cards(state, topic)
        if MODEL_CONNECTIONS in models:
            written += self.connect_decks(state)
            for topic in topics:
                written += self.connect_cards(state, topic)
        if MODEL_DIFFICULTY in models:
            written += self.fill_groups(state, [card for card in state.cards_state.cards if not state.is_topic(card.id)])
        return written

    def run_deck(self, state, models: Collection[str]) -> int:
        """The topics view: the entered deck, and its checked topics' cards (the sublevel)."""
        if not state.entered_deck:
            return 0
        written = 0
        if MODEL_TOPICS in models:
            self.give_model_topics(state, [state.entered_deck])
        topics = self._checked_topics(state, state.entered_deck)
        if MODEL_POSITIONS in models:
            for topic in topics:
                self.place_cards(state, topic)
        if MODEL_CONNECTIONS in models:
            for topic in topics:
                written += self.connect_cards(state, topic)
        if MODEL_DIFFICULTY in models:
            name = state.entered_name
            cards = [card for card in state.cards_state.visible_cards(name) if not state.is_topic(card.id)] if name else []
            written += self.fill_groups(state, cards)
        return written

    def run_topic(self, state, models: Collection[str]) -> int:
        """The cards view: the entered topic's cards."""
        topic = state.entered_topic_info
        if topic is None:
            return 0
        written = 0
        if MODEL_POSITIONS in models:
            self.place_cards(state, topic)
        if MODEL_CONNECTIONS in models:
            written += self.connect_cards(state, topic)
        if MODEL_DIFFICULTY in models:
            members = set(topic.card_ids)
            written += self.fill_groups(state, [card for card in state.cards_state.cards if card.id in members])
        return written

    # --- the models, each alone ------------------------------------------------

    def give_model_topics(self, state, deck_ids: Collection[str]) -> int:
        """T1 alone: give these decks the model's topics (once). The viewer's topic index is rebuilt from the
        derived file in place (the card source, the position source and the router hold the index object itself),
        the decks' view checks start over, stale topic checks are dropped and the dots reloaded; the cards keep
        their positions and connections. Returns the number of decks that got the model's topics."""
        new = sorted(deck_id for deck_id in deck_ids if deck_id not in self.model_decks)
        if not new:
            return 0
        self.model_decks.update(new)
        self.write_course(state)
        state.topics.reload(CardsFile(self.derived_path), state.decks)
        for deck_id in new:
            state.settings.drop(deck_id)  # new topics: every topic and every card of the deck checked
        state.settings.drop_stale_topics(state.topics.signatures())
        state.reload_dots()
        return len(new)

    def place_decks(self, state) -> int:
        """The decks at the model's positions. Returns the decks placed."""
        decks = state.decks_state
        placed = 0
        for deck_id, (x, y, z) in self.model.deck_points.items():
            if deck_id in decks.position_by_id:
                decks.set_position(deck_id, x, y, z)
                placed += 1
        return placed

    def connect_decks(self, state) -> int:
        """The decks' connections of the model in merge order (replacing its earlier ones), the decks' cut-off
        preset and Display's check of them. Returns the connections written."""
        written = set_links(state.decks_state, self.model.deck_links, self.model.deck_points)
        state.cutoffs[DECKS] = preset_cutoff(self._checked_deck_ids(state), self.model.deck_links, DECK_PART)
        if written:
            state.show_origin(DECKS, LINK_ORIGIN)
        return written

    def layout(self, topic: TopicInfo) -> TopicLayout:
        """The topic's layout: the model's own, or one made on the spot for a scripted topic (CourseModel.layout_of)."""
        return self.model.layout_of(topic.deck_id, topic.topic_id, topic.card_ids, topic.name)

    def place_cards(self, state, topic: TopicInfo) -> int:
        """A topic's cards at the model's positions on its sphere. Returns the cards placed."""
        cards = state.cards_state
        placed = 0
        for card_id, (x, y, z) in self.layout(topic).positions.items():
            if card_id in cards.position_by_id:
                cards.set_position(card_id, x, y, z)
                placed += 1
        return placed

    def connect_cards(self, state, topic: TopicInfo) -> int:
        """A topic's connections of the model in merge order (replacing its earlier ones among the topic's cards),
        the topic's cut-off preset and Display's check of them. Returns the connections written."""
        pairs = self.layout(topic).links
        written = set_links(state.cards_state, pairs, topic.card_ids)
        state.cutoffs[topic_key(topic)] = preset_cutoff(shown_ids(state, topic), pairs, CARD_PART)
        if written:
            state.show_origin(topic_key(topic), LINK_ORIGIN)
        return written

    def fill_groups(self, state, cards: list[Card]) -> int:
        """The missing difficulty memberships of these cards, rated and predicted (the generated groups become the
        oldest again)."""
        if not cards:
            return 0
        added = fill_difficulty_groups(state.cards_state, cards)
        if self.difficulty is not None:
            ids = {card.id for card in cards}
            added += fill_predicted_groups(
                state.cards_state, {card_id: level for card_id, level in self.difficulty.predictions.items() if card_id in ids}
            )
        return added

    @staticmethod
    def _checked_deck_ids(state) -> list[str]:
        """The checked decks, the dots of the decks view."""
        checked = set(state.cards_state.enabled_decks)
        return [info.deck_id for info in state.decks.infos if info.name in checked]

    @staticmethod
    def _checked_topics(state, deck_id: str) -> list[TopicInfo]:
        """The deck's checked topics (their dots are loaded), none for an unchecked deck."""
        return [topic for topic in state.topics.topics(deck_id) if topic.dot_id in state.cards_state.card_by_id]


def topic_key(topic: TopicInfo) -> str:
    """The key of a topic's cut-off line and Display in the router (as ViewRouter.view_key names the cards view's)."""
    return f"{CARDS}:{topic.dot_id}"


def shown_ids(state, topic: TopicInfo) -> list[str]:
    """The topic's checked cards (the Cards dialog's), the dots its cards view shows."""
    stored = state.settings.shown(topic.deck_id)
    if stored is None:
        return list(topic.card_ids)
    checked = set(stored)
    return [card_id for card_id in topic.card_ids if card_id in checked]


def model_pairs(core) -> list[tuple[str, str]]:
    """The model's connections of a state as pairs, in the live list's order."""
    return [(link.source_id, link.target_id) for link in core.links if link.origin == LINK_ORIGIN]


def set_links(core, pairs, scope_ids: Iterable[str]) -> int:
    """Replace the model's connections among a scope's dots (the decks; a topic's cards) with `pairs` in merge order:
    the model's earlier connections with an end in the scope go (after T1 moved cards, an earlier topic's would
    distort the new topic's tree), the fresh ones are appended as the model's, and the user's stay, the same pair
    or not. A pair with an end that is not loaded is left out. The list is changed in place, as always; `core` is a
    state with a live `links` list and `card_by_id` (a UserStateService). Returns the connections written."""
    scope = set(scope_ids)
    loaded = core.card_by_id
    fresh = [
        CardLink(source, target, LINK_ORIGIN) for source, target in pairs
        if source != target and source in loaded and target in loaded
    ]
    kept = [
        link for link in core.links
        if link.origin != LINK_ORIGIN or not (link.source_id in scope or link.target_id in scope)
    ]
    core.links[:] = kept + fresh
    return len(fresh)
