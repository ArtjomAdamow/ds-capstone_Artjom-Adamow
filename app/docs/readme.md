# app: guided learning

The capstone's main app: the `models/`, the course table they read, the standalone page that shows them, the glue that puts them behind the optional viewer's Sputnik **Helper** button (`glue/`, the only part that imports the viewer), the launcher's mode question, the launcher `start.py` and their tests. What comes next is in [roadmap.md](roadmap.md).

Development arrives in trains: `app/`, `start.py`, `tests/app/` and the course data in `data/` are replaced with each train, so they are never edited here; a fix comes with a later train.

Run the commands below from the capstone's root.

## Structure

```text
├── start.py                   # the launcher (below)
├── evaluate.py                # the offline evaluation of the models (below)
├── app/
│   ├── cards.py               # the card record the models read: the seven fields of Flashcard and a quiz's answer options, the stacks
│   ├── course_table.py        # the course table: cards.csv, decks.csv, topics.csv, read with the csv module
│   ├── figures.py             # the plotly figures the page and the glue's card panel share (plotly only)
│   ├── launch.py              # the mode question: full (with the viewer) or standalone
│   ├── models/                # nothing of the viewer: numpy, scipy, scikit-learn; sentence-transformers and transformers where installed (S2, D2)
│   │   ├── similar.py              # card text, vectoriser, unit vectors, the own minimum, neighbours, deck vectors
│   │   ├── embeddings.py           # S2: the sentence encoder, the embeddings' cache, the hybrid with TF-IDF, the app's vectoriser
│   │   ├── topic_model.py          # T1: nearest centroid over the scripted topics (TopicSeed, TopicModel)
│   │   ├── layout.py               # the layout step: SVD start, force layout on the sphere, measures
│   │   ├── clustering.py           # connections from scipy's clustering, in merge order; the cut-off preset's rank
│   │   ├── course_model.py         # the fitted model: deck points and connections, the topics' layouts and connections
│   │   ├── difficulty_model.py     # D1: the difficulty of the unrated quizzes, measured out of fold; D2 alone judged the same way
│   │   ├── zero_shot.py            # D2: the zero-shot difficulty tags of the quizzes by an NLI model, cached like the embeddings
│   │   └── evaluation.py           # the offline measures and the judges of the topic model, for evaluate.py
│   ├── standalone/            # the models' own page: Dash and plotly, nothing of the viewer or the glue
│   │   ├── __init__.py             # main(course_dir, results_dir): the page on the course table and the results
│   │   ├── app.py                  # Page (the fitted models, what each view and panel shows), the app and its one callback
│   │   ├── layout.py               # the map figure, the dendrogram's window and the model panels
│   │   ├── results.py              # the results files read for the panels: the newest run, the tables, the glossary
│   │   └── assets/                 # style.css: the viewer's theme tokens, written anew; map_side.js: the map's side; measure_rows.js: a results row's click
│   ├── glue/                  # the models on sputnik's map, over its public API: the Helper button and its dialog
│   └── docs/                  # this readme, the roadmap and columns.md, what the results' columns mean
├── tests/app/                 # the model tests, the page's, the launchers' and the evaluation's tests on model_support.py
├── results/                   # hand_marks.csv and hand_labels.csv come with the trains; evaluate.py writes results.csv, difficulty.csv, topics.csv
└── data/                      # git-ignored: topics.json, course/, images/, jupyter/, the app's state in state/, S2's embeddings and D2's tags in cache/
```

## The models

- **The similarity model S1** is one TF-IDF over the course's card texts (the question, the answer and the explanation), unit rows and the cosine: every card's five nearest cards above the index's own minimum score (the 10th percentile of the fifth neighbour's cosine), and the vectors every other model reads. **S2** is a sentence encoder in the same slot (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`), alone or hybrid with S1's TF-IDF (the two halves' unit rows side by side, each at its own minimum, so the cosine ranks pairs as the sum of the two scores): it needs `sentence-transformers` (the `embeddings` extra below) and caches the course's embeddings in `data/cache/embeddings/` (a derived file, keyed by the model and the texts; nothing to back up). Which one the app runs is `APP_VECTORISER` in `app/models/embeddings.py`, set by the hand check; without the library or the model the app and the evaluation run on TF-IDF and say so in one line.
- **The topic model T1** takes the scripted topics as its input and decides every card's topic: the nearest of the deck's topic centroids by cosine, each centroid the mean of the topic's cards' vectors (leave-one-out, so a card never votes for itself). On the course 351 of the 1,064 cards (33 %) get another topic than the script; 2 topics end up empty.
- **The difficulty model D1** rates the flattened quizzes that have no difficulty (56 in the course; flashcards are not rated). The levels are relative to the deck: each deck's quizzes keep the course's shares (8 easy, 3 medium and 1 hard of 12). **D2** tags the same quizzes with an NLI model and no training (`models/zero_shot.py`: the quiz with its options, the correct options' feedback and the explanation against three hypotheses, the tags cached in `data/cache/embeddings/`; it needs the `embeddings` extra below): a judge next to D1 in the evaluation, and a feature set of D1 (`text+tags`) that the hand labels found within noise of D1's order, so D1 runs on its text statistics.
- **The connections and their cut** (`models/clustering.py`): one connection per merge of the hierarchical clustering, in merge order, and `preset_rank`, the last merge whose connected parts stay within a size (five decks, four cards), which the page and the glue both use.

## The course table

The models' input without the viewer's parser, three CSV files in `data/course/`:

- `cards.csv`: every card the parser keeps, in its order: `comb_id`, the deck's id and name, the scripted topic's id, name and rest flag, the stack, the authored difficulty, the question, the answer and the explanation, and a quiz's answer options, their feedback and the indices of the correct ones as JSON lists in three cells (empty for a card without options);
- `decks.csv`: every deck in deck id order with its description;
- `topics.csv`: every deck's scripted topics in file order, the empty ones too (T1's seeds hold them, so an emptied topic keeps its dot).

`course_table.load_table(folder)` reads them with the csv module into what the models take: the card records, the deck ids by name, the scripted topics as `TopicSeed`s, and the deck list. The ids agree with the viewer's, since the table is written with its parser from the same course file `data/topics.json`.

## The standalone page

The models on a page of their own, without the viewer, in its theme. It fits the course model and D1 on the course table at the start (a few seconds on TF-IDF; about ten with S2's hybrid, the embeddings from the cache and the encoder loaded once), reads the results files next to `evaluate.py` and stores nothing.

- **Three views:** the decks, a deck's topics (Open on a selected deck), a topic's cards (Open on a selected topic). A click selects a dot, Open (or a click on a selected deck's or topic's panel) enters it, Back goes up a level with the dot it came from selected.
- **Four model panels,** in a column right of the map, their text at double size: S – Similarity, C – Course model, T – Topics, D – Difficulty, each a thin row until a click expands it (one at a time; its close box collapses it). A panel holds one section per level of the model (S1, C1, T1, D1): the level's live measures of the view (C1's readout, D1's scores, S1's vocabulary and minimum score, T1's agreement, moved cards and margins), then its rows of the newest run of its results file, one row per measure and one column per variant with the chosen one marked, next to its floor of that run (B0 for S1, the random layout for C1, chance for T1, always easy for D1). A click on a row opens its description under it, what the measure is and its range ([columns.md](columns.md)), one row at a time: a click on another row moves it, a click on the open row or its close box closes it. Without a results file the section says how to make one.
- **The map follows the expanded panel.** With every panel collapsed, or S or D expanded, it shows the scripted topics: grey dots at random positions (a fixed seed, the same for every panel), no connections, centroid, axes or colours, as the viewer without Helper; D adds the difficulty colours, as the viewer shows them after Helper: a rated quiz by its level (easy blue, medium beige-brown, hard red-orange), a quiz D1 predicted amber and drawn as a cross, a flashcard grey, with the legend inside the map. C shows the model: T1's topics at C1's positions, the decks and the cards connected gold and dashed as far as the preset keeps them (parts of at most five decks, 9 of the 23 on the course; parts of at most four cards), the centroid of a deck and the axes, x, y and z with ticks at −1, −0.5, 0, 0.5 and 1, the unit sphere's coordinates in which the readout measures a dot's nearest. T shows the same positions with T1's markers: a card coloured by its margin on a teal scale, a cross below the threshold, a violet diamond for a card T1 moved in, an open pink diamond for a card it moved out (placed at the mean direction of its most similar cards of the topic), an emptied topic as an open circle and a topic's dot grown with its share of cards between two topics; the legend is in the panel. The topics view places the topics at random in every state (the model has no topic positions yet) and draws no connections there.
- **The map** is a square of 12.5 % of the window's area (`assets/map_side.js` sets its side on load and on resize, at least 240 px); under it (beside it in portrait) the dendrogram's window: the selected dot's ids and counts (a deck: `deck_id`, topics, cards; a topic: `deck_id`, `topic_id`, the rest flag, its cards by kind; a card: `comb_id`, `deck_id`, `topic_id`, the stack, the authored or predicted difficulty, whether T1 placed it between two topics, and with T expanded its margin and runner-up topic) or, with nothing selected, the dendrogram of the view's membership with the preset's cut-off line, its heights the real merge distances, its leaves by id, in every panel state. Under the dendrogram's window, with D expanded, the difficulty window: D1's confusion matrix and the shares of the levels of the view's part (the course, the entered deck, the entered topic).
- **No learning material, ids only:** no card's question, answer or explanation and no deck's description, anywhere on the page; the map's hover, the line under the top bar and the panels are ids: a deck's `deck_id` (`deck 01` under the top bar), a card's `comb_id`; no deck's name. No topic name either: a topic's label is its `topic_id` and a hash of its name (`02-3fa9c1`), made at runtime and stored nowhere. A results cell that names a deck or a topic of the course table shows the id or the label instead.

Start it with the launcher: answer `n` at "Launch Sputnik? (y/n)"; without sputnik installed it starts without asking.

## The launcher: start.py

It takes no arguments; the folders and the commands are written in it:

```bash
uv run start.py
```

- **It searches for sputnik** (the viewer extra, imported as `project`). Found, it asks "Launch Sputnik? (y/n)": `y`, or no input, starts the full app, sputnik's map with the Helper button; `n` the standalone page. Not found, it starts the standalone page without asking and prints the command that installs the viewer (`uv sync --all-groups --extra viewer`). The question is `launch.py`'s (`choose_mode`).
- **The standalone page** reads the course table in `data/course/` and stores nothing.
- **It stops with the command that fixes it** when a library is missing (`uv sync --all-groups`), the course data is missing, or the installed sputnik is older than `launch.VIEWER_RELEASE`, the release whose hooks the glue uses.
- **S2's encoder is optional:** an extra `embeddings` of the capstone's `pyproject.toml` with `sentence-transformers` (it brings torch and transformers; on Linux pin torch to the CPU index, `https://download.pytorch.org/whl/cpu`, with an `[[tool.uv.index]]` entry marked `explicit = true` and a `[tool.uv.sources]` line for torch, since the Linux wheel pulls the CUDA libraries). The models are fetched once from the Hugging Face hub into its cache (`HF_HOME`), D2's NLI model with them (`app/models/zero_shot.py`, `NLI_MODEL`; `evaluate.py` prints the fetching command when it is missing); set `HF_HUB_OFFLINE=1` afterwards. Without the extra both modes and `evaluate.py` run on TF-IDF, the S2 and D2 rows are skipped with one line each and the encoder and tagger tests skip.
- Until the models are needed it imports only the standard library and `launch.py`, so it starts on any interpreter.
- **In the flashcards workspace** the launcher there (`modules/helper/helper.py`) asks no question: the viewer is always there, so it starts the full app and the standalone page together, the page as a process of its own on port 8051 (`standalone.main` takes the port), prints its URL and puts a Models button into the viewer's top bar that opens the page. `start.py` keeps the question and its two modes: one process here, and no Models button.
- **The viewer's newer hooks:** the glue uses what the installed viewer offers (`glue/composition.py` reads `compose`'s signature): with sputnik 0.3.0 the back of the card panel holds D1's scores, confusion matrix and shares, and the predicted quizzes are amber circles; a newer sputnik shows the matrix and the shares in a window under the card's window, the predicted quizzes as amber crosses, and the groups' legend inside the map.

## The offline evaluation

`evaluate.py` measures the models on the course table and appends one row per model and variant to `results/` next to it (`--results` names another folder; `--table` another course table; `--cache` the cache of S2's embeddings and D2's tags, `data/cache/` next to the package by default): `--part similarity` the similarity model's vectoriser variants, S1's TF-IDF settings and, with the `embeddings` extra, S2's encoders and their hybrids with S1's chosen settings (same-deck precision@5, coverage, each variant's own minimum score), the random floor of random positions on the sphere and the course model's map on the app's vectoriser (trustworthiness and neighbour overlap of the deck layout and of the card layouts per topic, at a fixed small k with the lift over random) into `results.csv`; `--part difficulty` D1 out of fold with its baselines, every feature set's within-deck order on the hand labels paired against the chosen set with its bootstrap interval, and with the extra D2's zero-shot tags, both templates alone and as D1's feature set `text+tags`, into `difficulty.csv`; `--part topics` T1 against the scripted topics into `topics.csv`, both on the app's vectoriser. Without `--part` all three run; every row of a run carries its start time (`run`), and a results file whose columns grew is written again with its rows kept. What the columns mean: [columns.md](columns.md).

```bash
uv run evaluate.py --part similarity
```

The hand-judged measures (the relevance of the neighbours, the difficulty levels of the unrated quizzes) are computed from `results/hand_marks.csv` and `results/hand_labels.csv`, which come with the trains without their notes; their judging forms stay in the workspace the trains come from, since they hold card text, so here nothing is written back to the two files and no form is written. It imports nothing of the viewer. What is measured and what the numbers say: [roadmap.md](roadmap.md), "The offline evaluation".

## The tests

The model tests (`test_similar.py`, `test_layout.py`, `test_clustering.py`, `test_topic_model.py`, `test_course_model.py`, `test_difficulty_model.py`, `test_embeddings.py`, `test_zero_shot.py`, `test_models_package.py`) are plain `unittest.TestCase` classes. They import `tests/app/model_support.py` only: the package on the path, the test course as a table of card records (three decks, 13 cards in seven scripted topics, one of them empty and one the rest topic; T1 keeps its scripted topics, so `moving_cards()` adds a card that makes T1 move another), a course for the difficulty model (`difficulty_course()`, three decks of 12 rated quizzes with their unrated quizzes' options, and `difficulty_report()`), and the stand-ins for S2's encoder and D2's pipeline (`FakeEncoder`, `FakeClassifier`), so the tests of both run without torch; the real models' tests skip without the library or the cached model. They run in an env without the viewer; `test_models_package.py` checks that no model source imports it and runs the models in a process where it cannot be imported.

The page's tests (`test_figures.py`, `test_standalone.py`) import `model_support.py` too, and nothing of the viewer: `test_standalone.py` writes the test course as a course table, drives the three views through the page's callback, runs the page in a process where the viewer cannot be imported (and checks that nothing of it or of the glue is loaded), and, when `data/course/` is there, serves the real course table.

The launchers' tests import `model_support.py` too: `test_launch.py` the mode question (`launch.py`) with the viewer passed in or faked, `test_start.py` the launcher with its two runs mocked (the question only with sputnik found, the reloader's child, the stops with their commands, and a process without the models' libraries in which it loads and stops).

The evaluation's tests (`test_evaluate.py`) import `model_support.py` and `evaluate.py` next to the package: the results files, the hand marks and labels with their forms on the fixtures, the measures on a hand-made layout, the rows of the similarity part, D2's rows and the feature set `text+tags` with the fake pipeline, and a run on the test course's table, with and without the hand files, in a process where the viewer cannot be imported.

`model_support.py` finds the package from where it lies (the nearest folder above it that holds `app/`) and the real course table in `data/course/` next to it. None of the tests needs the course data or the viewer. Run them under either runner:

```bash
uv run python -m pytest tests/app
uv run python -m unittest discover -s tests/app
```
