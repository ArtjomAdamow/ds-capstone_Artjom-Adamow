"""The dashboard's state file (the standard library only): `dashboard_state.json` in `data/state/`, apart from the
viewer's state file, written at every change.

What it holds (version 1): the read and the explored flag per card (only the cards with a flag set), the shown
card and the trail, the cards shown before it, oldest first. A missing file is the closed start: nothing explored,
no card shown, an empty trail. A file of another version is refused, not converted: the launcher's backup question
(app.launch.prepare_state) names the file among the state files, so a fresh start is its y. The flags of cards the
course does not hold (an older course) are dropped at the load, and a shown card or a trail entry among them too.

The rules (step 33's, on the dashboard's parts):

* what explores a card is a selection in any way: Go's answer that shows a card, a click on a Related entry or on
  one of Go's entries, Back; nothing else and nothing ahead (`show`);
* the turn marks the card read, a quiz once judged (`mark_read`); explored is not turned;
* `show` puts the card shown before onto the trail; `back` takes the last card of the trail back as the shown card
  and drops the one left, as the viewer's Return restores the view before a navigation.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Optional

VERSION = 1
STATE_NAME = "dashboard_state.json"
REFUSED = (
    "{path} is a dashboard state file of version {found}, this dashboard reads version {wanted}: answer y at the "
    "backup question to start fresh."
)


@dataclass
class CardFlags:
    """What the learner did with one card: turned it (read), and reached it by a selection (explored)."""

    read: bool = False
    explored: bool = False

    def document(self) -> dict:
        return {"read": self.read, "explored": self.explored}

    @property
    def empty(self) -> bool:
        return not (self.read or self.explored)


@dataclass
class DashboardState:
    """The flags per card, the shown card and the trail, read from and written to `path` (None: kept in memory
    alone, the tests' and a start without a state folder)."""

    path: Optional[Path] = None
    cards: dict[str, CardFlags] = field(default_factory=dict)
    shown: Optional[str] = None
    trail: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path], known: Collection[str]) -> "DashboardState":
        """The state of the file at `path`, the flags of unknown cards (not among `known`, the course's card ids)
        dropped; the closed start without a file. A file of another version is refused (ValueError)."""
        state = cls(Path(path) if path is not None else None)
        if state.path is None or not state.path.is_file():
            return state
        document = json.loads(state.path.read_text(encoding="utf-8"))
        found = document.get("version") if isinstance(document, dict) else None
        if found != VERSION:
            raise ValueError(REFUSED.format(path=state.path, found=found, wanted=VERSION))
        known = set(known)
        for card_id, flags in (document.get("cards") or {}).items():
            if card_id in known and isinstance(flags, dict):
                state.cards[card_id] = CardFlags(bool(flags.get("read")), bool(flags.get("explored")))
        shown = document.get("shown")
        state.shown = shown if shown in known else None
        state.trail = [card_id for card_id in document.get("trail") or [] if card_id in known]
        return state

    def document(self) -> dict:
        """The file's content: the version, the cards with a flag set, the shown card and the trail."""
        return {
            "version": VERSION,
            "cards": {card_id: flags.document() for card_id, flags in self.cards.items() if not flags.empty},
            "shown": self.shown,
            "trail": list(self.trail),
        }

    def save(self) -> None:
        """The whole document written atomically (a temporary file next to it, then replaced); nothing without a
        path."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.document(), indent=2, ensure_ascii=False)
        handle, temporary = tempfile.mkstemp(dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(text)
            os.replace(temporary, self.path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    # --- the flags ---------------------------------------------------------------------------------------------

    def flags(self, card_id: str) -> CardFlags:
        return self.cards.setdefault(card_id, CardFlags())

    def is_read(self, card_id: str) -> bool:
        return card_id in self.cards and self.cards[card_id].read

    def is_explored(self, card_id: str) -> bool:
        return card_id in self.cards and self.cards[card_id].explored

    @property
    def explored_ids(self) -> set[str]:
        return {card_id for card_id, flags in self.cards.items() if flags.explored}

    @property
    def fresh(self) -> bool:
        """The closed start: nothing explored, nothing read, no card shown."""
        return self.shown is None and not self.trail and all(flags.empty for flags in self.cards.values())

    def mark_read(self, card_id: str) -> None:
        """The turn: the card is read from here on (a quiz once judged)."""
        self.flags(card_id).read = True

    def explore(self, card_id: str) -> None:
        self.flags(card_id).explored = True

    # --- the shown card and the trail -------------------------------------------------------------------------

    def show(self, card_id: str) -> None:
        """A selection: the card is shown and explored, the card shown before goes onto the trail (nothing when it
        is the same card)."""
        if card_id != self.shown:
            if self.shown is not None:
                self.trail.append(self.shown)
            self.shown = card_id
        self.explore(card_id)

    def back(self) -> Optional[str]:
        """Back: the last card of the trail is shown again and the card left is dropped; None on an empty trail,
        where nothing changes."""
        if not self.trail:
            return None
        self.shown = self.trail.pop()
        self.explore(self.shown)
        return self.shown

    @property
    def can_go_back(self) -> bool:
        return bool(self.trail)
