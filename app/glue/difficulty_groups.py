"""Difficulty groups: the course's easy, medium and hard as generated card groups of the card state ("Quiz: easy",
"Quiz: medium", "Quiz: hard"), and the grey group of the predicted quizzes ("Quiz difficulty: predicted").

Every card with a difficulty is a member of its difficulty's group; unrated cards are in none and keep the grey of
a card without a group. A quiz that the difficulty model D1 rated (models.difficulty_model) gets two memberships:
the grey group "Quiz difficulty: predicted" and the group of its predicted level, the same groups (and colours) as the
rated cards. A card can also be in any of the user's groups.

The groups are this module's records (GROUP_IDS, `difficulty-<key>`, apart from the user's "group-<n>"), of the cards
view's kind and of this maker's origin (ORIGIN): the viewer lists them in the cards view only, keeps them read only
(no Add, Remove or Delete by hand; Focus still works) and never saves them with the user's state. They live in the
helper's model state instead (derived.write_model_state, written after every save): stored_groups and
restore_groups bring them back at a restart, with this maker's kind, origin, names and colours, so the file can't
smuggle another group in.

They are always the oldest groups: every fill gives them the lowest ranks ("Quiz difficulty: predicted" first, then
easy, medium and hard) and the first places after the viewer's predefined groups of the topic dots
(project.content.topic_groups), which keep theirs. A predicted card therefore shows grey, a rated card its
difficulty colour, unless Focus unchecks them; and a group missing from the state comes back as the oldest when the
Helper button generates it again. A generated group left without members (its decks unchecked) is filled again from
the loaded cards (refill_empty), so the same set of generated groups stays.
"""

from dataclasses import asdict
from typing import Iterable, Mapping

from project.models import USER_ORIGIN, CardGroup
from project.services.views import CARDS

ORIGIN = "difficulty_model"  # the maker of these groups (CardGroup.origin)

# difficulty -> (group name, colour)
DIFFICULTY_GROUPS = {
    "easy": ("Quiz: easy", "#4f7fcf"),  # blue (a Blue-Gray moved into blue, apart from the no-group grey)
    "medium": ("Quiz: medium", "#b8956a"),  # Beige-Brown
    "hard": ("Quiz: hard", "#ff5349"),  # Red-Orange
}
GROUP_NAMES = [name for name, _ in DIFFICULTY_GROUPS.values()]
# The predicted cards: the no-group grey, so a predicted card looks unrated until Focus unchecks this group
PREDICTED = "predicted"
PREDICTED_GROUP = ("Quiz difficulty: predicted", "#7c8b99")
# key (PREDICTED or a difficulty) -> group id, the oldest first
GROUP_IDS = {key: f"difficulty-{key}" for key in (PREDICTED, *DIFFICULTY_GROUPS)}
GENERATED_IDS = list(GROUP_IDS.values())


def _age(group) -> tuple:
    return (group.rank is None, group.rank if group.rank is not None else 0)


def _record(key: str, members: Iterable[str]) -> CardGroup:
    """The group of a key as this maker makes it (its id, name, colour, kind and origin), with these members."""
    name, color = PREDICTED_GROUP if key == PREDICTED else DIFFICULTY_GROUPS[key]
    return CardGroup(GROUP_IDS[key], name, color, list(members), kind=CARDS, origin=ORIGIN)


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


def generated_members(group_id: str, cards, predictions: Mapping[str, str]) -> list[str]:
    """The members a generated group has among these cards, in their order: for a level's group the cards rated at
    that level and the ones predicted at it, for "Quiz difficulty: predicted" every predicted card."""
    if group_id == GROUP_IDS[PREDICTED]:
        return [card.id for card in cards if card.id in predictions]
    level = next((level for level in DIFFICULTY_GROUPS if GROUP_IDS[level] == group_id), None)
    if level is None:
        return []
    return [card.id for card in cards if card.difficulty == level or predictions.get(card.id) == level]


def refill_empty(core, cards, predictions: Mapping[str, str]) -> int:
    """Fill every generated group that exists but holds nothing again with its members among these cards (the
    loaded ones), so the same set of generated groups stays; one with no members among them stays empty. Ranks and
    places are kept. Returns the number of memberships added."""
    added = 0
    for group in generated_groups(core):
        if not group.card_ids:
            members = generated_members(group.id, cards, predictions)
            group.card_ids.extend(members)
            added += len(members)
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


# --- the groups in the helper's model state ------------------------------------------------------------------------


def generated_groups(core) -> list[CardGroup]:
    """This maker's groups of the card state, in list order."""
    return [group for group in core.groups if group.origin == ORIGIN and group.id in GENERATED_IDS]


def group_records(core) -> list[dict]:
    """The generated groups as the model state stores them (with their members)."""
    return [asdict(group) for group in generated_groups(core)]


def stored_groups(records) -> list[CardGroup]:
    """The groups of stored records: the known ids only, each once, with this maker's kind, origin, name and colour;
    the members as stored."""
    groups = {}
    for item in records if isinstance(records, list) else []:
        key = next((key for key, group_id in GROUP_IDS.items() if isinstance(item, dict) and item.get("id") == group_id), None)
        members = item.get("card_ids") if key is not None else None
        if key is not None and key not in groups and isinstance(members, list):
            groups[key] = _record(key, [str(member) for member in members])
    return list(groups.values())


def restore_groups(core, groups: Iterable[CardGroup], loaded_ids) -> None:
    """Put the stored generated groups into the card state (instead of any there), their members cut to the loaded
    cards, then make them the oldest."""
    loaded = set(loaded_ids)
    core.groups[:] = [group for group in core.groups if group.origin != ORIGIN]
    for group in groups:
        core.groups.append(CardGroup(
            group.id, group.name, group.color, [card_id for card_id in group.card_ids if card_id in loaded],
            kind=group.kind, origin=group.origin,
        ))
    make_oldest(core)
