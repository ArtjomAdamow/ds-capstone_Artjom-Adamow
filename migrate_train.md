# Migrate train

- app/, start.py, tests/app/ and this file come from the flashcards workspace with every train: commit them as they are and never edit them here; a fix comes with a later train.
- data/topics.json, data/course/, data/images/ and data/jupyter/ are refreshed by the trains (git-ignored, never edited here); data/state/ holds the app's state, written by start.py.
- README, the app section: start it with uv run start.py (with the viewer extra it asks "Launch Sputnik? (y/n)", without it the standalone page starts). app/ and start.py are Artjom Adamow's code under the MIT licence; sputnik, the viewer extra, is a separate private package with its own licence, and none of its code is in this repository.
- LICENSE names only the template's holder (Copyright (c) 2020 neuefische GmbH): add the line "Copyright (c) 2026 Artjom Adamow" for app/ and start.py, or name the holder in the README's app section.
- exclude app/, start.py and tests/app/ from ruff lint and format (the exported code keeps the workspace's line length).
- protocol_export.md in the root is an old copy of the export side's draft and no train updates it: remove it.
- delete this file after all instructions are done.
