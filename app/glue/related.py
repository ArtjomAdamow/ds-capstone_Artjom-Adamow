"""The card section (glue): the selected card's related cards and its next card, the entries the viewer draws under
the card panel through its `card_section` hook (project.services.actions.ModuleSection; step 28 of the helper).

* Related cards: the selected card's neighbours of the index (CourseModel.related: the five nearest above the
  index's own minimum score, across the whole course), in the index's order, each with where it lies: this topic,
  this deck, or another deck by name. A neighbour in a checked deck is a target the viewer navigates to on a click
  (its topic entered, the card selected); a neighbour in an unchecked deck is listed without a click, its note says
  so. The cards view shows about 44 % of the neighbours; the list is where the other 56 % show
  (docs/models/similarity.md, "The hand check", point 8).
* The next card: the first entry, marked, the neighbour the learner should take next by the learner's confidence
  (rank_by_confidence): unread cards first, then the read ones by their confidence, the lowest ("review") first
  (CardConfidence.confidence, 1 to 100; the course's easy, medium and hard play no part). Among the unread cards the
  quizzes D1 predicted take their places in D1's order, the easiest first (its score, the cold-start order of the
  extra quizzes, docs/models/difficulty.md, "Conclusion"); the other unread cards keep the neighbour
  order, the cold-start order of the flashcards until a prior exists. A neighbour without a target is never the next
  card. The next card is listed again among the neighbours, in its place.

The entries' labels are the cards' questions, shortened; a card the router does not hold (an unchecked deck) is named
by its deck. Nothing here is stored: the entries are made for every draw of the panel.
"""

from typing import Mapping, Optional, Sequence

from project.services.views import CARDS

SECTION_TITLE = "Related cards"
LABEL_LENGTH = 90
THIS_TOPIC = "this topic"
THIS_DECK = "this deck"
UNCHECKED = "deck unchecked"


def rank_by_confidence(
    ids: Sequence[str], read: Mapping[str, bool], confidence: Mapping[str, int], scores: Optional[Mapping[str, float]] = None,
) -> list[str]:
    """The cards in the order the learner should take them: the unread ones first (`read` by card id, False
    without an entry), then the read ones by their confidence ascending (`confidence` by card id, the lowest, "review",
    first; a card without one counts as 1), ties in the given order. Among the unread cards the ones with a score
    (`scores`: D1's, the lower the easier) take their places in score order; the others keep the given order."""
    scores = scores or {}
    unread = [card_id for card_id in ids if not read.get(card_id, False)]
    scored = sorted((card_id for card_id in unread if card_id in scores), key=lambda card_id: (scores[card_id], unread.index(card_id)))
    scored_places = iter(scored)
    ordered = [next(scored_places) if card_id in scores else card_id for card_id in unread]
    done = sorted((card_id for card_id in ids if read.get(card_id, False)), key=lambda card_id: (confidence.get(card_id, 1), ids.index(card_id)))
    return ordered + done


def shorten(text: str, length: int = LABEL_LENGTH) -> str:
    text = " ".join(text.replace("`", "").replace("**", "").split())
    return text if len(text) <= length else text[: length - 1] + "…"


def helper_section(model, questions: Mapping[str, str], scores: Optional[Mapping[str, float]] = None):
    """The viewer's card section: `entries(state, dot)` lists the selected card's neighbours with the next card
    first (the module docstring); nothing for a deck, a topic dot or outside the cards view. `questions` are the
    course's by card id (the entries' labels), `scores` D1's by card id (the next card's order among the extra
    quizzes). ModuleSection and SectionEntry are imported here, not with the module: a viewer older than sputnik
    0.5.0 has none, and the composition calls this only where the viewer's compose takes the hook."""
    from project.services.actions import ModuleSection, SectionEntry

    def place(state, other: str) -> str:
        topic = state.topics.topic_of(other)
        if state.entered_topic and topic is not None and topic.dot_id == state.entered_topic:
            return THIS_TOPIC
        deck_id = model.deck_of.get(other)
        if deck_id is not None and deck_id == state.entered_deck:
            return THIS_DECK
        info = state.decks.by_id.get(deck_id) if deck_id is not None else None
        return info.name if info is not None else f"deck {deck_id or '–'}"

    def entries(state, dot) -> list:
        if state.view != CARDS or dot is None or state.is_topic(dot.id):
            return []
        loaded = state.cards_state.card_by_id
        records = state.cards_state.confidence
        neighbours = [other for other, _score in model.related(dot.id)]
        if not neighbours:
            return []
        listed = []
        for other in neighbours:
            card = loaded.get(other)
            target = other if card is not None else None
            label = shorten(card.question if card is not None else questions.get(other, other))
            note = place(state, other) if target else f"{place(state, other)} · {UNCHECKED}"
            listed.append(SectionEntry(label, target, note))
        read = {other: bool(records[other].read) for other in neighbours if other in records}
        confidence = {other: int(records[other].confidence) for other in neighbours if other in records}
        with_target = [entry.target for entry in listed if entry.target]
        ranked = rank_by_confidence(with_target, read, confidence, scores)
        if not ranked:
            return listed
        first = next(entry for entry in listed if entry.target == ranked[0])
        return [SectionEntry(first.label, first.target, first.note, marked=True)] + listed

    return ModuleSection(SECTION_TITLE, entries)
