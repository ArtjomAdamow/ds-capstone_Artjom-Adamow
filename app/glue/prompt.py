"""The Ask dialog (glue): P1's answer to a request typed in the viewer, as the targets the viewer navigates to,
through its `prompt` hook (project.services.actions.ModulePrompt; step 26 of the helper).

The model answers course-wide on T1's membership (models.prompt_model.PromptModel.find, the app's variant): the
topics in rank order, the decision and the selected card. The glue turns the answer into the viewer's targets, the
nearest three topics of the checked decks in rank order, each with the deck id, the topic's dot id from the viewer's
topic index, the card id or none, and the label: the deck's and the topic's names, with the card's question shortened
to 90 characters where a card is given. No score: a cosine says nothing to a learner (the standalone page shows them).

* A target's topic is the viewer's topic of the same id: T1 keeps the scripted topic ids, so the topic the model
  names is the one entered, whether the deck has the scripted topics or the model's. A target with a card goes to the
  topic that holds the card in the viewer, which differs from T1's where the deck has its scripted topics and T1
  moved the card, so the card is selected where it is shown.
* A target's card: the decision's own on the first target where the answer navigates; on the others the topic's
  nearest card, as the standalone page lists them (no card bar; prompt.md, "Hard decisions", 3). A card the router
  does not hold is none.
* With the dialog's checkbox All decks off (the router's runtime flag `all_decks`, roadmap step 33 of the helper; off
  at every start) the targets in unchecked decks are left out: when the best match lies in an unchecked deck nothing
  navigates and the message names the deck; "no match" says so. With it on, every deck's topics are targets and Go
  enters an unchecked deck, whose card the viewer's selection then explores and checks. An empty request has no
  targets and no message.

`navigation` is the answer in plain values; `helper_prompt` is the hook that hands them to the viewer.
"""

from typing import NamedTuple, Optional

from ..models.prompt_model import NEAREST
from ..neighbours import shorten

ASK_LABEL = "Ask"
ASK_TITLE = "Ask the course"
PLACEHOLDER = "A question, a concept or a code name"
NO_MATCH_MESSAGE = "No match in the course."
UNCHECKED_MESSAGE = "The best match is in {deck}, unchecked in the Decks dialog."
UNCHECKED_NEAREST = "No match in the course; the nearest topic is in {deck}, unchecked in the Decks dialog."


class Target(NamedTuple):
    """A place the viewer navigates to, in the fields' order of the viewer's PromptTarget."""

    deck_id: str
    topic: str  # the topic's dot id
    card: Optional[str]  # a card id among the loaded cards, or None
    label: str


class Navigation(NamedTuple):
    """The answer in plain values, in the fields' order of the viewer's PromptAnswer."""

    message: str
    targets: tuple[Target, ...]  # rank order, the decision's own first where it lies in a checked deck
    navigate: bool


def target_of(state, key, card_id: Optional[str]) -> Optional[Target]:
    """The viewer's target for a topic of T1 (`key`: deck name, topic id) and the card given with it; None for a
    deck the viewer does not show (unknown or unchecked) or a topic its index does not hold."""
    info = state.decks.by_name.get(key[0])
    if info is None or (not getattr(state, "all_decks", False) and info.name not in state.enabled_decks):
        return None
    card = state.cards_state.card_by_id.get(card_id) if card_id is not None else None
    topic = state.topics.topic_of(card_id) if card is not None else None
    if topic is None:
        topic = next((topic for topic in state.topics.topics(info.deck_id) if topic.topic_id == key[1]), None)
    if topic is None:
        return None
    label = f"{topic.deck_name} › {topic.name}" + (f" · {shorten(card.question)}" if card is not None else "")
    return Target(info.deck_id, topic.dot_id, card_id if card is not None else None, label)


def navigation(prompt_model, state, text: Optional[str]) -> Navigation:
    """The answer to a request on the router `state`: the nearest three topics of the checked decks as targets,
    the message and whether the viewer navigates to the first (the module docstring)."""
    text = (text or "").strip()
    if not text:
        return Navigation("", (), False)
    answer = prompt_model.find(text)
    targets, first = [], None
    for place, scored in enumerate(answer.ranking):
        card_id = answer.card if place == 0 and answer.navigate else scored.card
        target = target_of(state, scored.key, card_id)
        if place == 0:
            first = target
        if target is not None:
            targets.append(target)
        if len(targets) == NEAREST:
            break
    if answer.ranking and first is None:
        best = state.decks.by_name.get(answer.ranking[0].key[0])
        if best is not None and best.name not in state.enabled_decks:
            return Navigation((UNCHECKED_MESSAGE if answer.navigate else UNCHECKED_NEAREST).format(deck=best.name), tuple(targets), False)
        return Navigation(NO_MATCH_MESSAGE, tuple(targets), False)
    return Navigation(answer.message if answer.navigate else NO_MATCH_MESSAGE, tuple(targets), answer.navigate)


def helper_prompt(prompt_model):
    """The viewer's Ask dialog: the button's label, `find(state, text)` (navigation in the viewer's records), the
    dialog's title and the input's placeholder. ModulePrompt, PromptTarget and PromptAnswer are imported here, not
    with the module: a viewer without the hook has none, and the composition calls this only where the viewer's
    compose takes it."""
    from project.services.actions import ModulePrompt, PromptAnswer, PromptTarget

    def find(state, text):
        found = navigation(prompt_model, state, text)
        return PromptAnswer(found.message, tuple(PromptTarget(*target) for target in found.targets), found.navigate)

    return ModulePrompt(ASK_LABEL, find, ASK_TITLE, PLACEHOLDER)
