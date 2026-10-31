# app: guided learning

The capstone's main app: the `models/`, the course table they read, the standalone page that shows them, the glue that puts them behind the optional viewer's Sputnik **Helper** button (`glue/`, the only part that imports the viewer), the launcher's mode question, the launcher `start.py` and their tests. What comes next is in [roadmap.md](roadmap.md).

Development arrives in trains: `app/`, `start.py`, `tests/app/` and the course data in `data/` are replaced with each train, so they are never edited here; a fix comes with a later train.

Run the commands below from the capstone's root.

## Structure

```text
├── start.py                   # the launcher (below)
├── app/
│   ├── cards.py               # the card record the models read (the seven fields of Flashcard as a protocol), the stacks
│   ├── course_table.py        # the course table: cards.csv, decks.csv, topics.csv, read with the csv module
│   ├── figures.py             # the plotly figures the page and the glue's card panel share (plotly only)
│   ├── launch.py              # the mode question: full (with the viewer) or standalone
│   ├── models/                # nothing of the viewer: numpy, scipy, scikit-learn
│   │   ├── similar.py              # card text, vectoriser, unit vectors, neighbours, deck vectors
│   │   ├── topic_model.py          # T1: nearest centroid over the scripted topics (TopicSeed, TopicModel)
│   │   ├── layout.py               # the layout step: SVD start, force layout on the sphere, measures
│   │   ├── clustering.py           # connections from scipy's clustering, in merge order; the cut-off preset's rank
│   │   ├── course_model.py         # the fitted model: deck points and connections, the topics' layouts and connections
│   │   ├── difficulty_model.py     # D1: the difficulty of the unrated quizzes, measured out of fold
│   │   └── evaluation.py           # the offline measures and the judges of the topic model
│   ├── standalone/            # the models' own page: Dash and plotly, nothing of the viewer or the glue
│   │   ├── __init__.py             # main(course_dir): the page on the course table
│   │   ├── app.py                  # Page (the fitted models, what each view shows), the app and its one callback
│   │   ├── layout.py               # the map figure and the panels
│   │   └── assets/style.css        # the viewer's theme tokens, written anew
│   ├── glue/                  # the models on sputnik's map, over its public API: the Helper button and its dialog
│   └── docs/                  # this readme and the roadmap
├── tests/app/                 # the model tests, the page's and the launchers' tests on model_support.py
└── data/                      # git-ignored: topics.json, course/, images/, jupyter/ and the app's state in state/
```

## The models

- **The topic model T1** takes the scripted topics as its input and decides every card's topic: the nearest of the deck's topic centroids by cosine, each centroid the mean of the topic's cards' vectors (leave-one-out, so a card never votes for itself). On the course 351 of the 1,064 cards (33 %) get another topic than the script; 2 topics end up empty.
- **The difficulty model D1** rates the flattened quizzes that have no difficulty (56 in the course; flashcards are not rated). The levels are relative to the deck: each deck's quizzes keep the course's shares (8 easy, 3 medium and 1 hard of 12).
- **The connections and their cut** (`models/clustering.py`): one connection per merge of the hierarchical clustering, in merge order, and `preset_rank`, the last merge whose connected parts stay within a size (five decks, four cards), which the page and the glue both use.

## The course table

The models' input without the viewer's parser, three CSV files in `data/course/`:

- `cards.csv`: every card the parser keeps, in its order: `comb_id`, the deck's id and name, the scripted topic's id, name and rest flag, the stack, the authored difficulty, the question, the answer and the explanation;
- `decks.csv`: every deck in deck id order with its description;
- `topics.csv`: every deck's scripted topics in file order, the empty ones too (T1's seeds hold them, so an emptied topic keeps its dot).

`course_table.load_table(folder)` reads them with the csv module into what the models take: the card records, the deck ids by name, the scripted topics as `TopicSeed`s, and the deck list. The ids agree with the viewer's, since the table is written with its parser from the same course file `data/topics.json`.

## The standalone page

The models on a page of their own, without the viewer, in its theme. It fits the course model and D1 on the course table at the start (a few seconds) and stores nothing.

- **Decks view:** the decks at the model's positions on the sphere, the clustering's connections gold and dashed as far as the preset keeps them (parts of at most five decks: 9 of the 23 on the course).
- **Topics view** (Open on a selected deck): the deck's model topics on a ring around the faint centroid in file order (the model has no topic positions yet); an emptied topic is an open circle.
- **Cards view** (Open on a selected topic): the topic's cards on the surface of its sphere (grey, opacity 0.1), connected as far as the preset keeps them (parts of at most four cards), coloured by difficulty: easy blue, medium beige-brown, hard red-orange, predicted and unrated grey.
- **Moving around:** a click selects a dot, Open (or a click on a selected deck's or topic's panel) enters it, Back goes up a level with the dot it came from selected.
- **The map** is a square of 12.5 % of the window's area (`assets/map_side.js` sets its side on load and on resize, at least 240 px), with the course model C1's readout under it: C1's own measures of the view. In every view it shows its axes, x, y and z with ticks at −1, −0.5, 0, 0.5 and 1: the unit sphere's coordinates, in which the readout measures a dot's nearest. The difficulty colour legend shows in the cards view only, under the title of D1's metrics; a selected card's panel names its difficulty.
- **The panels, ids only:** on the right the selected dot's ids and counts (a deck: `deck_id`, topics, cards; a topic: `deck_id`, `topic_id`, the rest flag, its cards by kind; a card: `comb_id`, `deck_id`, `topic_id`, the stack, the authored or predicted difficulty, whether T1 placed it between two topics) or, with nothing selected, the dendrogram of the view's connections with the preset's cut-off line, its heights the real merge distances, its leaves by id; under it D1's metrics of the view (all decks, a deck, a topic, named by ids).
- **No learning material:** no card's question, answer or explanation and no deck's description, anywhere on the page; the map's hover and the line under the top bar are ids: a deck's `deck_id` (`deck 01` under the top bar), a card's `comb_id`; no deck's name. No topic name either: the map's hover and the line under the top bar give a topic's label, its `topic_id` and a hash of its name (`02-3fa9c1`), made at runtime and stored nowhere.

Start it with the launcher: answer `n` at "Launch Sputnik? (y/n)"; without sputnik installed it starts without asking.

## The launcher: start.py

It takes no arguments; the folders and the commands are written in it:

```bash
uv run start.py
```

- **It searches for sputnik** (the viewer extra, imported as `project`). Found, it asks "Launch Sputnik? (y/n)": `y`, or no input, starts the full app, sputnik's map with the Helper button; `n` the standalone page. Not found, it starts the standalone page without asking and prints the command that installs the viewer (`uv sync --all-groups --extra viewer`). The question is `launch.py`'s (`choose_mode`).
- **The standalone page** reads the course table in `data/course/` and stores nothing.
- **It stops with the command that fixes it** when a library is missing (`uv sync --all-groups`), the course data is missing, or the installed sputnik is older than `launch.VIEWER_RELEASE`, the release whose hooks the glue uses.
- Until the models are needed it imports only the standard library and `launch.py`, so it starts on any interpreter.

## The tests

The model tests (`test_similar.py`, `test_layout.py`, `test_clustering.py`, `test_topic_model.py`, `test_course_model.py`, `test_difficulty_model.py`, `test_models_package.py`) are plain `unittest.TestCase` classes. They import `tests/app/model_support.py` only: the package on the path, the test course as a table of card records (three decks, 13 cards in seven scripted topics, one of them empty and one the rest topic; T1 keeps its scripted topics, so `moving_cards()` adds a card that makes T1 move another) and a course for the difficulty model (`difficulty_course()`, three decks of 12 rated quizzes, and `difficulty_report()`). They run in an env without the viewer; `test_models_package.py` checks that no model source imports it and runs the models in a process where it cannot be imported.

The page's tests (`test_figures.py`, `test_standalone.py`) import `model_support.py` too, and nothing of the viewer: `test_standalone.py` writes the test course as a course table, drives the three views through the page's callback, runs the page in a process where the viewer cannot be imported (and checks that nothing of it or of the glue is loaded), and, when `data/course/` is there, serves the real course table.

The launchers' tests import `model_support.py` too: `test_launch.py` the mode question (`launch.py`) with the viewer passed in or faked, `test_start.py` the launcher with its two runs mocked (the question only with sputnik found, the reloader's child, the stops with their commands, and a process without the models' libraries in which it loads and stops).

`model_support.py` finds the package from where it lies (the nearest folder above it that holds `app/`) and the real course table in `data/course/` next to it. None of the tests needs the course data or the viewer. Run them under either runner:

```bash
uv run python -m pytest tests/app
uv run python -m unittest discover -s tests/app
```
