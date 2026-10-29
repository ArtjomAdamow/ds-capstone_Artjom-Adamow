# app: the models and the standalone page

The helper's own package, the capstone's main app. This readme covers the part that does not depend on the viewer: the models (`models/`, importing nothing of the flashcards package), the course table they read without the viewer's parser, the standalone page that shows them without the viewer, the launcher's mode question, the capstone's launcher `start.py`, their tests and their offline evaluation. The glue that puts the models behind the Helper button (`glue/`, over the flashcards package's public API) is described with the helper (`modules/helper/readme.md`). The concept, architecture and measured numbers are in `docs/draft/helper/`, what comes next in [roadmap.md](roadmap.md).

Run the commands below from the root of the main project repository.

## Structure

```text
modules/helper/
├── evaluate.py                # offline evaluation script (run in the module env)
├── export_course.py           # writes the course table with the viewer's parser (this workspace only)
├── start.py                   # the capstone's launcher, exported to its root next to app/ (below)
├── app/
│   ├── cards.py               # the card record the models read (the seven fields of Flashcard as a protocol), the stacks
│   ├── course_table.py        # the course table: cards.csv, decks.csv, topics.csv, read with the csv module
│   ├── figures.py             # the plotly figures the page and the glue's card panel share (plotly only)
│   ├── launch.py              # the launcher's mode question: full (with the viewer) or standalone
│   ├── models/                # nothing of the project: numpy, scipy, scikit-learn
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
│   └── docs/                  # this readme, the roadmap, the analysis of the two emptied topics
├── results/                   # evaluation results (git-ignored: derived course data)
└── tests/                     # the model tests, the page's and the launchers' tests on model_support.py
```

## The models

- **The topic model T1** takes the scripted topics as its input and decides every card's topic: the nearest of the deck's topic centroids by cosine, each centroid the mean of the topic's cards' vectors (leave-one-out, so a card never votes for itself). On the course 351 of the 1,064 cards (33 %) get another topic than the script; 2 topics end up empty ([T1_empties_analysis.md](T1_empties_analysis.md)).
- **The difficulty model D1** rates the flattened quizzes that have no difficulty (56 in the course; flashcards are not rated). The levels are relative to the deck: each deck's quizzes keep the course's shares (8 easy, 3 medium and 1 hard of 12).
- **The connections and their cut** (`models/clustering.py`): one connection per merge of the hierarchical clustering, in merge order, and `preset_rank`, the last merge whose connected parts stay within a size (five decks, four cards), which the page and the glue both use.

## The course table

The models' input without the viewer's parser, three CSV files in one folder (`data/course/` here):

- `cards.csv`: every card the parser keeps, in its order: `comb_id`, the deck's id and name, the scripted topic's id, name and rest flag, the stack, the authored difficulty, the question, the answer and the explanation;
- `decks.csv`: every deck in deck id order with its description;
- `topics.csv`: every deck's scripted topics in file order, the empty ones too (T1's seeds hold them, so an emptied topic keeps its dot).

`course_table.load_table(folder)` reads them with the csv module into what the models take: the card records, the deck ids by name, the scripted topics as `TopicSeed`s, and the deck list. The ids agree with the viewer's, since the table is written with its parser:

```bash
.\modules\helper\.venv\Scripts\python.exe modules/helper/export_course.py [course file] [--out <folder>]
```

The course file is `data/topics.json` by default, the folder `data/course/`. Export again after the course file changes. The table travels into the capstone's git-ignored `data/` together with the course file, the images and the notebooks, whenever they change (`scripts/migrate_helper.py` of the flashcards workspace, `docs/draft/helper/protocol_export.md`).

## The standalone page

The models on a page of their own, without the viewer, in its theme. It fits the course model and D1 on the course table at the start (a few seconds) and stores nothing.

- **Decks view:** the decks at the model's positions on the sphere, the clustering's connections gold and dashed as far as the preset keeps them (parts of at most five decks: 9 of the 23 on the course).
- **Topics view** (Open on a selected deck): the deck's model topics on a ring around the centroid in file order (the model has no topic positions yet); an emptied topic is an open circle.
- **Cards view** (Open on a selected topic): the topic's cards on the surface of its sphere, connected as far as the preset keeps them (parts of at most four cards), coloured by difficulty: easy blue, medium beige-brown, hard red-orange, predicted and unrated grey.
- **Moving around:** a click selects a dot, Open (or a click on a selected deck's or topic's panel) enters it, Back goes up a level with the dot it came from selected.
- **The panels:** on the left the selected dot (a deck's description, a topic's counts, a card's question and answer, a predicted quiz's level) or, with nothing selected, the dendrogram of the view's connections with the preset's cut-off line, its heights the real merge distances; on the right D1's metrics of the view (the course, the deck, the topic).

Start it with the helper's launcher: answer `n` at "Launch Sputnik? (y/n)", or skip the question:

```bash
.\modules\helper\.venv\Scripts\python.exe modules/helper/helper.py --mode standalone
```

Without the course table the launcher stops with the export command. The mode question is `launch.py`'s: without the viewer (the package imported as `project`) the mode is standalone and nothing is asked; with it the launcher asks, and without input it starts the full app. `--mode full` without the viewer stops with the hint to install the viewer extra.

## The capstone's launcher: start.py

The capstone has no launcher of its own: `start.py` is developed here next to the package and exported with it into the capstone's root (`docs/draft/helper/protocol_export.md`). It takes no arguments; the folders and the commands are written in it. From the capstone's root:

```bash
uv run start.py
```

- **It searches for sputnik** (the viewer extra, imported as `project`). Found, it asks "Launch Sputnik? (y/n)": `y`, or no input, starts the full app, sputnik's map with the Helper button; `n` the standalone page. Not found, it starts the standalone page without asking and prints the command that installs the viewer (`uv sync --all-groups --extra viewer`).
- **The standalone page** reads the course table in `data/course/` and stores nothing.
- **The full app** reads the course file `data/topics.json` and the images in `data/images/`, and keeps its state in `data/state/`: the viewer's state file `user_state.json`, the derived course file `helper_topics.json` and the model state `model_state.json` (no backup question: delete the folder for a fresh start). It runs with Dash's debug reloader; the reloader's child asks nothing and runs the chosen mode.
- **It stops with the command that fixes it** when a library is missing (`uv sync --all-groups`), the course data is missing (the migration copies it into `data/`), or the installed sputnik is older than `launch.VIEWER_RELEASE`, the release whose hooks the glue uses.
- Until the models are needed it imports only the standard library and `launch.py`, so it starts on any interpreter.

The workspace's launcher is `modules/helper/helper.py` (`modules/helper/readme.md`): the module env, the delta layer and `temp/`.

## The tests without the viewer

The model tests (`test_similar.py`, `test_layout.py`, `test_clustering.py`, `test_topic_model.py`, `test_course_model.py`, `test_difficulty_model.py`, `test_models_package.py`) are plain `unittest.TestCase` classes. They import `tests/model_support.py` only: the package on the path, the test course as a table of card records (three decks, 13 cards in seven scripted topics, one of them empty and one the rest topic; T1 keeps its scripted topics, so `moving_cards()` adds a card that makes T1 move another) and a course for the difficulty model (`difficulty_course()`, three decks of 12 rated quizzes, and `difficulty_report()`). They run in an env without the flashcards package; `test_models_package.py` checks that no model source imports it and runs the models in a process where it cannot be imported.

The page's tests (`test_figures.py`, `test_standalone.py`) import `model_support.py` too, and nothing of the viewer: `test_standalone.py` writes the test course as a course table, drives the three views through the page's callback, runs the page in a process where the viewer cannot be imported (and checks that nothing of it or of the glue is loaded), and, when `data/course/` is exported, serves the real course table.

The launchers' tests import `model_support.py` too: `test_launch.py` the mode question (`launch.py`) with the viewer passed in or faked, `test_start.py` the capstone's launcher with its two runs mocked (the question only with sputnik found, the reloader's child, the stops with their commands, and a process without the models' libraries in which it loads and stops).

These eleven files and `model_support.py` are the tests exported with the package (into the capstone's `tests/app/`, `docs/draft/helper/protocol_export.md`). `model_support.py` finds the package from where it lies (the nearest folder above it that holds `app/`), and the real course table in `data/course/` next to the package or at the workspace's root, so they run unchanged in both places. Run them with the helper's tests (`modules/helper/readme.md`, "Run the tests").

## Offline evaluation

```bash
.\modules\helper\.venv\Scripts\python.exe modules/helper/evaluate.py [--part similarity|difficulty|topics]
```

The script measures the models of `app/models/`; it reads the course file with the project's parser through the glue (`glue/course.py`) and draws the random positions of B0 and the random layouts with the project's (`glue/positions.py`). The course table can replace the first, a draw of the models' own the second ([roadmap.md](roadmap.md), step 21).

`--part topics` measures the topic model T1 against the scripted topics of `data/topics.json`: the agreement of the nearest centroid rule next to the neighbour rule, a logistic regression per deck and chance, over the course, by stack and per deck; the margin table (the cards kept at a threshold and their agreement), the coherence per topic, the moved cards and the emptied topics. One row per model and scope goes to `results/topics.csv`.

`--part difficulty` measures only the difficulty model D1. It appends one row per feature set and label rule, plus the two baselines, to `results/difficulty.csv`: out of fold on the rated quizzes, as mean ± sd over the repeats, and the predicted levels of the unrated quizzes.

`--part similarity` measures the random floor (B0), the TF-IDF variants (same-deck precision@5, coverage, distinct recommended cards) and the chosen model's map (trustworthiness and neighbour overlap of the deck layout, the within-topic overlap of the card layouts over the model's topics), appends one row per model and variant to `results/results.csv`, and writes the hand-check candidates for 20 NLP cards to `results/hand_check.md`, to be judged by hand: mark every relevant candidate with `x`, or the `none` line when nothing fits. Later runs keep the marks and add every variant's hand-judged precision@5 over the judged cards (`hand_precision` in the results). Without `--part` all three parts run. The rows note the course's decks ("decks 01 to 24"); older rows of the 48-deck course stay above them. The numbers are in `docs/draft/helper/concept.md`.
