"""Difficulty groups: the course's easy, medium and hard as generated card groups of the card state ("Quiz: easy",
"Quiz: medium", "Quiz: hard"), and the amber group of the predicted quizzes ("Quiz difficulty: predicted").

Every card with a difficulty is a member of its difficulty's group; unrated cards are in none and keep the grey of
a card without a group, so in the cards view the flashcards alone are grey. A quiz that the
difficulty model D1 rated (models.difficulty_model) gets two memberships: the amber group "Quiz difficulty:
predicted" (PREDICTED_COLOR, the colour of the predicted series in the shares figure and of a predicted card on the
standalone page's map, written there anew) and the group of its predicted level, the same groups (and colours) as
the rated cards. A card can also be in any of the user's groups.

The groups are this module's records (GROUP_IDS, `difficulty-<key>`, apart from the user's "group-<n>"), of the cards
view's kind and of this maker's origin (ORIGIN): the viewer lists them in the cards view only, keeps them read only
(no Add, Remove or Delete by hand; Focus still works) and never saves them with the user's state. Since roadmap step 33
of the helper they are filled at every start for the whole course (button.Helper.groups) and handed to the viewer with
the model state (its `groups`); the model state file keeps D1's predicted levels they are filled from.

They are always the oldest groups: every fill gives them the lowest ranks ("Quiz difficulty: predicted" first, then
easy, medium and hard), and the viewer puts the groups it is handed first after its predefined groups of the topic
dots (project.content.topic_groups), which keep theirs. A predicted card therefore shows the predicted colour (the
oldest group's), a rated card its difficulty colour, unless Focus unchecks them.
"""

from dataclasses import fields
from typing import Iterable, Mapping

from project.models import USER_ORIGIN, CardGroup
from project.services.views import CARDS

ORIGIN = "difficulty_model"  # the maker of these groups (CardGroup.origin)
# The predicted quizzes' marker on the map: a cross (CardGroup.symbol, a viewer newer than sputnik 0.3.0; a viewer
# without the field draws circles), as the standalone page draws them
PREDICTED_SYMBOL = "cross"
HAS_SYMBOL = any(item.name == "symbol" for item in fields(CardGroup))

# difficulty -> (group name, colour)
DIFFICULTY_GROUPS = {
    "easy": ("Quiz: easy", "#4f7fcf"),  # blue (a Blue-Gray moved into blue, apart from the no-group grey)
    "medium": ("Quiz: medium", "#b8956a"),  # Beige-Brown
    "hard": ("Quiz: hard", "#ff5349"),  # Red-Orange
}
GROUP_NAMES = [name for name, _ in DIFFICULTY_GROUPS.values()]
# The predicted cards: the amber of figures.PREDICTED_COLOR, so a predicted card shows as predicted, not as its
# level, until Focus unchecks this group; the flashcards alone keep the no-group grey
PREDICTED = "predicted"
PREDICTED_COLOR = "#c98500"
PREDICTED_GROUP = ("Quiz difficulty: predicted", PREDICTED_COLOR)
# key (PREDICTED or a difficulty) -> group id, the oldest first
GROUP_IDS = {key: f"difficulty-{key}" for key in (PREDICTED, *DIFFICULTY_GROUPS)}
GENERATED_IDS = list(GROUP_IDS.values())


def _age(group) -> tuple:
    return (group.rank is None, group.rank if group.rank is not None else 0)


def _record(key: str, members: Iterable[str]) -> CardGroup:
    """The group of a key as this maker makes it (its id, name, colour, kind, origin and, for the predicted group on
    a viewer with group symbols, the cross), with these members."""
    name, color = PREDICTED_GROUP if key == PREDICTED else DIFFICULTY_GROUPS[key]
    symbol = {"symbol": PREDICTED_SYMBOL} if key == PREDICTED and HAS_SYMBOL else {}
    return CardGroup(GROUP_IDS[key], name, color, list(members), kind=CARDS, origin=ORIGIN, **symbol)


def _group(core, key: str):
    group_id = GROUP_IDS[key]
    return next((item for item in core.groups if item.id == group_id and item.origin == ORIGIN), None)


def _join(core, card_id: str, key: str) -> int:
    """Put the card into the group of this key, making it; 1 when a membership was added."""
    group = _group(core, key)
    if group is None:
        core.groups.append(_record(key, [card_id]))
        return 1
    if card_id in group.card_ids:
        return 0
    group.card_ids.append(card_id)
    return 1


def fill_difficulty_groups(core, cards) -> int:
    """Put every card with a difficulty into its difficulty group, making a missing group, then make the difficulty
    groups the oldest. `core` is anything with a live `groups` list (a StateCore or a UserStateService). Idempotent;
    returns the number of memberships added."""
    added = 0
    for card in cards:
        if card.difficulty in DIFFICULTY_GROUPS:
            added += _join(core, card.id, card.difficulty)
    make_oldest(core)
    return added


def fill_predicted_groups(core, predictions: Mapping[str, str]) -> int:
    """Every predicted card (card id -> level) into "Quiz difficulty: predicted" and into its level's difficulty group,
    making missing groups; a card that was predicted at another level before leaves that level's group. Then the
    generated groups are made the oldest. Idempotent; returns the number of memberships added."""
    added = 0
    for card_id, level in predictions.items():
        if level not in DIFFICULTY_GROUPS:
            continue
        added += _join(core, card_id, PREDICTED)
        added += _join(core, card_id, level)
        for other in DIFFICULTY_GROUPS:
            group = _group(core, other)
            if other != level and group is not None and card_id in group.card_ids:
                group.card_ids = [member for member in group.card_ids if member != card_id]
    make_oldest(core)
    return added


def make_oldest(core) -> None:
    """The lowest ranks for the generated groups ("Quiz difficulty: predicted", then easy, medium and hard), the user's
    groups after them in their own order; the generated groups also come first in the list, after the other makers'
    groups (the viewer's predefined topic groups keep their ranks and the very first places); the user's keep their
    places."""
    own = sorted(generated_groups(core), key=lambda group: GENERATED_IDS.index(group.id))
    users = [group for group in core.groups if group.origin == USER_ORIGIN]
    others = [group for group in core.groups if group.origin not in (ORIGIN, USER_ORIGIN)]
    for rank, group in enumerate(own + sorted(users, key=_age), start=1):
        group.rank = rank
    core.groups[:] = others + own + users


def generated_groups(core) -> list[CardGroup]:
    """This maker's groups of the card state, in list order."""
    return [group for group in core.groups if group.origin == ORIGIN and group.id in GENERATED_IDS]
