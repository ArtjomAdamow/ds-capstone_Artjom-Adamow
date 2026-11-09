# app: guided learning

The capstone's main app: the `models/`, the course table they read, the dashboard that shows one card at a time with what a learner sees of them, the launcher's part, the launcher `start.py` and their tests. Nothing of the package imports a viewer.

Development arrives in trains: `app/`, `start.py`, `tests/app/` and the course data in `data/` are replaced with each train, so they are never edited here; a fix comes with a later train. The course data is git-ignored and never presented online. Deck 00, `data/demo/`, is the one part of `data/` in the repository: a synthetic deck, "How to use", that the dashboard's Demo shows.

Run the commands below from the capstone's root.

## Structure

```text
├── start.py                   # the launcher: the dashboard (below)
├── evaluate.py                # the offline evaluation of the models (below)
├── app/
│   ├── cards.py               # the card record the models read: the seven fields of Flashcard and a quiz's answer options, the stacks
│   ├── course_table.py        # the course table: cards.csv, decks.csv, topics.csv, read with the csv module
│   ├── figures.py             # the dashboard's plotly figures (plotly only)
│   ├── headlines.py           # each generation's two numbers from the results rows, with the verdict rule and its words
│   ├── launch.py              # the launcher's part: the port argument and the port check, the data folder, the backup question
│   ├── neighbours.py          # the related cards' order (the next card) and labels, the dashboard's
│   ├── panels.py              # the model panels' names and letters, the cut labels, the slot's helpers, the app's variant labels
│   ├── results.py             # the results files: each generation's newest run, the tables, the glossary
│   ├── models/                # nothing of the viewer: numpy, scipy, scikit-learn; sentence-transformers and transformers where installed (S2, D2)
│   │   ├── similar.py              # card text, vectoriser, unit vectors, the own minimum, neighbours, deck vectors
│   │   ├── embeddings.py           # S2: the sentence encoder, the embeddings' cache, the hybrid with TF-IDF, the app's vectoriser
│   │   ├── topic_model.py          # T1: nearest centroid over the scripted topics (TopicSeed, TopicModel)
│   │   ├── layout.py               # the layout step: SVD start, force layout on the sphere, measures
│   │   ├── clustering.py           # connections from scipy's clustering, in merge order; the cut-off preset's rank
│   │   ├── course_model.py         # the fitted model: deck points and connections, the topics' layouts and connections
│   │   ├── difficulty_model.py     # D1: the difficulty of the unrated quizzes, measured out of fold; D2 alone judged the same way
│   │   ├── zero_shot.py            # D2: the zero-shot difficulty tags of the quizzes by an NLI model, cached like the embeddings, read back alone
│   │   ├── prompt_model.py         # P1: a request scored against the topics by three routes at their self-set floors, a target or "no match"
│   │   └── evaluation.py           # the offline measures and the judges of the topic model, for evaluate.py
│   ├── dashboard/             # the dashboard: one card at a time, nothing of the viewer
│   │   ├── __init__.py             # main(course_dir, results_dir, state_dir, demo_dir): the dashboard on the course table, the results and its state file, its Demo on deck 00
│   │   ├── app.py                  # Dashboard (the fitted models, the state, what the regions show), DemoBoard (deck 00's), Boards (the one the view names), the app and its one callback
│   │   ├── layout.py               # the top bar with the request box, Back and Demo, the diagram, the card with its options and its Related list, the tiles and the model's panel
│   │   ├── panel_figures.py        # the panels' figures of the whole course: where the related cards lie, the decks on the sphere, P1's judged requests
│   │   ├── state.py                # the state file dashboard_state.json: the read and explored flags, the shown card, the trail
│   │   ├── quiz.py                 # a quiz's judgement as plain functions
│   │   └── assets/                 # style.css: the theme's tokens and the type scale, written anew; keys.js: Enter into the hidden input
│   └── docs/                  # this readme and columns.md, what the results' columns mean
├── tests/app/                 # the model tests, the shared parts', the dashboard's, the launchers' and the evaluation's tests on model_support.py
├── results/                   # hand_marks.csv, hand_labels.csv and prompt_requests.csv come with the trains; evaluate.py writes results.csv, difficulty.csv, topics.csv, prompt.csv
└── data/                      # git-ignored: topics.json, course/, images/, jupyter/, the app's and the dashboard's state in state/, S2's embeddings and D2's tags in cache/;
                               # tracked: demo/, deck 00 (dashboard.json and its table dashboard/)
```

## The models

- **The similarity model S1** is one TF-IDF over the course's card texts (the question, the answer and the explanation), unit rows and the cosine: every card's five nearest cards above the index's own minimum score (the 10th percentile of the fifth neighbour's cosine), and the vectors every other model reads. **S2** is a sentence encoder in the same slot (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`), alone or hybrid with S1's TF-IDF (the two halves' unit rows side by side, each at its own minimum, so the cosine ranks pairs as the sum of the two scores): it needs `sentence-transformers` (the `embeddings` extra below) and caches the course's embeddings in `data/cache/embeddings/` (a derived file, keyed by the model and the texts; nothing to back up). Which one the app runs is `APP_VECTORISER` in `app/models/embeddings.py`, set by the hand check; without the library or the model the app and the evaluation run on TF-IDF and say so in one line.
- **The topic model T1** takes the scripted topics as its input and decides every card's topic: the nearest of the deck's topic centroids by cosine, each centroid the mean of the topic's cards' vectors (leave-one-out, so a card never votes for itself). On the course 351 of the 1,064 cards (33 %) get another topic than the script; 2 topics end up empty.
- **The difficulty model D1** rates the flattened quizzes that have no difficulty (56 in the course; flashcards are not rated). The levels are relative to the deck: each deck's quizzes keep the course's shares (8 easy, 3 medium and 1 hard of 12). **D2** tags the same quizzes with an NLI model and no training (`models/zero_shot.py`: the quiz with its options, the correct options' feedback and the explanation against three hypotheses, the large model of the ModernBERT family, the tags cached in `data/cache/embeddings/`; it needs the `embeddings` extra below): a judge next to D1 in the evaluation, built on D1's frame and supplying no feature to it; on the hand labels it orders the extra quizzes within noise of D1 and below it, so D1 runs the app on its text statistics (the answer options' statistics add nothing D1 can learn from the rated quizzes).
- **The connections and their cut** (`models/clustering.py`): one connection per merge of the hierarchical clustering, in merge order, and `preset_rank`, the last merge whose connected parts stay within a size (five decks, four cards), which the dashboard uses. A deck's topics are clustered the same way on their vectors (`course_model.topic_links`), 2 to 6 merges per deck, the tree of the topics view's dendrogram.

## The course table

The models' input without the viewer's parser, three CSV files in `data/course/`:

- `cards.csv`: every card the parser keeps, in its order: `comb_id`, the deck's id and name, the scripted topic's id, name and rest flag, the stack, the authored difficulty, the question, the answer and the explanation, and a quiz's answer options, their feedback and the indices of the correct ones as JSON lists in three cells (empty for a card without options);
- `decks.csv`: every deck in deck id order with its description;
- `topics.csv`: every deck's scripted topics in file order, the empty ones too (T1's seeds hold them, so an emptied topic keeps its dot), with the course file's headings and sources, joined by ` | ` in one cell each (read from the course file with the json module, since the parser keeps neither).

`course_table.load_table(folder)` reads them with the csv module into what the models take: the card records, the deck ids by name, the scripted topics as `TopicSeed`s, and the deck list. The ids agree with the viewer's, since the table is written with its parser from the same course file `data/topics.json`.

## The dashboard

The learner's page without the viewer: one card at a time with what a learner sees of the models, in a slides look, landscape only, in plain HTML. It fits the course model with T1, D1 and P1 on the course table at the start, and the same models on deck 00 for its Demo, reads the results files next to `evaluate.py`, and keeps its own state file. It imports nothing of the viewer.

```bash
uv run start.py
```

- **The card,** the right column, as the viewer shows it: the front the question, a click on the card or Enter turns it to the answer; a quiz's front the question with its options and Confirm under it, Confirm disabled while nothing is checked and enabled by the first check; Confirm or Enter turns it to its back, "Correct answer" with the card's feedback, or "Wrong answer" with every checked option's own feedback (right only when exactly the correct options are checked); Enter on the back turns it to the front with the checks cleared. The turn marks the card read, a quiz once judged. A quiz too tall for the card's frame scrolls inside it. The title row names the deck and the topic in full, over as many lines as they need; nothing of the models stands on the card: no level, no margin, no score. With no card shown the card's place says how to begin.
- **The request box,** in the top bar at all times, the one way in: a request in the learner's words, Go or Enter; P1's answer shows the topic's card where it navigates (the selected card, else the topic's nearest), or "No match in the course." with up to three entries under the box, the nearest topics' cards named by the deck, the topic and the question, each a click that shows the card and closes the answer; an empty request does nothing.
- **Related,** a button in the card's title row, lists the shown card's five related cards of the index, the next card first and marked, and again in its place; each the card's question shortened and where it lies (this topic, this deck, another deck by name); every entry a click that shows the card. The next card is the viewer's rule without the confidence: the unread cards first, the quizzes D1 predicted in D1's order among them, the other unread ones in the neighbours' order, the read ones after them. Nothing is explored when the list is drawn.
- **Back,** in the top bar, returns to the card shown before (the trail of the cards shown), disabled on an empty trail.
- **The diagram,** the left column: decks → topics → cards, three rows in plain HTML, each item a dot with its name (the deck's name, the topic's name, the question shortened): the explored decks, the open deck's topics that hold an explored card, the open topic's explored cards; nothing unexplored. The open deck and topic are the shown card's, framed in gold, with the shown card's deck, topic and card highlighted. A click on a card shows it; on a topic the cards row lists that topic's cards; on a deck the topics row lists its topics and the cards row empties; a click on a deck or a topic leaves the card as it is, and any selection opens the shown card's deck and topic again. A row too full scrolls inside itself. A fresh start shows it empty, and it fills as the learner explores.
- **The slide row,** along the bottom: five tiles, S, C, T, D and P, each the model's headline and its two numbers with their verdict words from the results files, for the generation the app runs (S2 with the encoder, else S1; C1, T1, D1, P1); one line where a results file is missing. A click on a tile opens the model's panel over the two columns, opaque, the tiles staying visible under it: on its left the headline and the two numbers one under the other, each with its label, its contrast, its sentence, its verdict word and its explanation; on its right the model's figure, the whole course's whatever card is shown: S where the related cards lie, seen from their card (this topic, this deck, another deck, with the share outside the card's topic), C the decks on the sphere with the connections below the preset and the shown card's deck gold and turned to the front, T the margins with the threshold and the line of the 25 % most confident, D D1's confusion matrix on the rated quizzes, P P1's judged requests on and off the course by where it took them (the counts `evaluate.py --part prompt` writes; a line until then). A panel too low folds: the sentences and the hint go first, then the explanations. Another tile shows that model in its place, a second click on the open tile closes it; Enter does nothing while it is open.
- **What explores a card:** a selection in any way, Go's answer that shows a card, a click on a Related entry, on one of Go's entries or on a card in the diagram, Back; nothing else and nothing ahead. Where a card lies is T1's topic, the model's, as the full app gives every deck from the start.
- **The state file** `data/state/dashboard_state.json`: the read and explored flags per card, the shown card and the trail, saved at every change; the closed start: a fresh start shows nothing, a restart shows the explored cards and the shown card again. The flip, Related drawn, the last request, the open tile and the diagram's open deck and topic live in the browser alone. The backup question covers the file; a file of another version is refused, and y at the question starts fresh. No episodes' log.
- **The Demo,** in the top bar after Back: it puts the course aside and shows deck 00, "How to use" (`data/demo/dashboard/`, 32 cards in six topics on the page's parts, a quarter of them rated quizzes), from its first card; the button reads "End demo", filled gold, while it runs. Deck 00 has its own board (`DemoBoard`), fitted at the start and kept in memory: the course model over its cards with its topics as written (T1 moves no card), P1 on its own floors, no D1, the app's kind of vectoriser on the encoder already loaded and without the embeddings' cache. While the demo runs the request box, Related, Back and the diagram answer from deck 00 alone; the tiles and their panels stay the course's. End demo, a reload or a restart drops the demo's state and brings back the course's card, trail and explored cards; nothing of the demo is saved.
- **Without course data,** as in a clone of this repository: the dashboard starts with deck 00 alone. Outside the demo the request box says "No data loaded", the box and Go are disabled, the diagram is empty and the card's place points to Demo; the tiles show the results files' newest runs, and the panels' figures but P's say they are drawn from the course. The state file is neither read nor written.
- **Large type:** the panel's numbers 3rem, headings and the tiles' numbers 1.5rem, the card's text between 1.35 and 2rem, nothing below 16 px; under 1400 px (a projector) a step smaller. Landscape only, no stacked layout.

## The launcher: start.py

It takes no arguments; the folders and the commands are written in it:

```bash
uv run start.py
```

- **It starts the dashboard,** with no question but the backup question, and needs no viewer.
- **The port is checked before anything serves** (`launch.choose_port`): a port is held when a server answers on it. The dashboard takes the next free port from Dash's default (8050) and prints the URL it serves on; `--port N` names the port and stops when it is held, naming the process that holds it: "Port 8060 is already in use (PID …, command line …). Stop it or use --port N".
- **The backup question,** after the port check: with `dashboard_state.json` in `data/state/`, it asks whether to back it up (`dashboard_state.json_01.bak`, `_02.bak`, …) and start fresh; `n` or no input keeps it (`launch.prepare_state`).
- **The dashboard** runs on the course table in `data/course/`, the results next to `evaluate.py` and its state file in `data/state/`, with its Demo on deck 00's table in `data/demo/dashboard/`, without the reloader and with no process next to it (`run_dashboard`). Without the course table it starts empty, with the Demo, and says so; without deck 00's table either it stops.
- **The data folder** is `data/` next to `start.py`; in the flashcards workspace, where `start.py` lies one level below the repository root, the root's (`launch.data_dir`). `evaluate.py` and `model_support.py` take it by the same rule.
- **It stops with the command that fixes it** when a library is missing (`uv sync --all-groups`) or the course data is missing.
- **S2's encoder is optional:** an extra `embeddings` of the capstone's `pyproject.toml` with `sentence-transformers` (it brings torch and transformers; on Linux pin torch to the CPU index, `https://download.pytorch.org/whl/cpu`, with an `[[tool.uv.index]]` entry marked `explicit = true` and a `[tool.uv.sources]` line for torch, since the Linux wheel pulls the CUDA libraries). The models are fetched once from the Hugging Face hub into its cache (`HF_HOME`), D2's NLI models with them (`app/models/zero_shot.py`, `NLI_MODELS`: the base and the large model, both read by `evaluate.py`, the large one by the app, `NLI_MODEL`; `evaluate.py` prints the fetching command for a missing one and skips its rows); set `HF_HUB_OFFLINE=1` afterwards. Without the extra the dashboard and `evaluate.py` run on TF-IDF, the S2 and D2 rows are skipped with one line each and the encoder and tagger tests skip. **On an NVIDIA GPU** the extra `embeddings-gpu` takes torch from the CUDA index instead (`uv sync --all-groups --extra embeddings-gpu`; 3 GB instead of 0.5 GB, a driver for CUDA 13.0 or newer; the two extras conflict, so every sync names one): the encoder and D2's model run on the GPU where torch sees one, else on the CPU. `APP_DEVICE` (`auto`, the default, `cpu` or `cuda`) chooses; the S panel names the device next to the encoding time and the evaluation's S2 rows carry it in `params`. fp32 on the GPU gives the CPU's vectors, so the cache serves both.
- Until the models are needed it imports only the standard library and `launch.py`, so it starts on any interpreter.

## The tests

The model tests (`test_similar.py`, `test_layout.py`, `test_clustering.py`, `test_topic_model.py`, `test_course_model.py`, `test_difficulty_model.py`, `test_embeddings.py`, `test_zero_shot.py`, `test_models_package.py`) are plain `unittest.TestCase` classes. They import `tests/app/model_support.py` only: the package on the path, the test course as a table of card records (three decks, 13 cards in seven scripted topics, one of them empty and one the rest topic; T1 keeps its scripted topics, so `moving_cards()` adds a card that makes T1 move another), a course for the difficulty model (`difficulty_course()`, three decks of 12 rated quizzes with their unrated quizzes' options, and `difficulty_report()`), and the stand-ins for S2's encoder and D2's pipeline (`FakeEncoder`, `FakeClassifier`), so the tests of both run without torch; the real models' tests skip without the library or the cached model. They run in an env without the viewer; `test_models_package.py` checks that no source of the package imports it and runs the models in a process where it cannot be imported.

The tests of the dashboard's shared parts import `model_support.py` too, and nothing of the viewer: `test_figures.py` (the figures, the connections, T1's margins), `test_results.py` (the results files, the tables, the glossary; it writes the small results files the dashboard's tests read), `test_headlines.py` (the verdict rule, each generation's two numbers) and `test_panels.py` (the panels' names, the cut labels, the slot's helpers, the variant labels).

The dashboard's tests (`test_dashboard.py`) import nothing of the viewer either: the closed start and a restart from its state file, what explores a card, the card's two sides and a quiz's judgement with Confirm's own callback, Related's order and places, Back on the trail, a request's answer and "no match" with its entries, the diagram's names and clicks, the tiles on the results fixtures and their panel with its figure, the Demo beside the course (deck 00 from card 001, End demo and a reload bringing back the course as it was, Related and the request box from deck 00 alone, the state file and the embeddings' cache untouched) and without course data, the layout's regions, the judgement's functions, a process with the viewer blocked (nothing of it or the page loaded), and a request on the real course table when it is there. `model_support.py` holds a deck 00 of twelve cards as a table for them (`write_demo_table`).

The launcher's tests import `model_support.py` too: `test_launch.py` the launcher's part (`launch.py`): the option reader and the port argument, the port check (a socket the test binds, the scan upwards, the holder from faked `netstat` and `lsof` output, the hint), the data folder and the backup question on a temporary folder with the state files; `test_start.py` the launcher with its run and the backup question mocked: the dashboard alone with no mode question, the port check before the backup question (the next free port with the printed URL, a held `--port` stopping it), the backup question over the dashboard's state file alone, the folders by the data folder's rule, the stops with their commands, and a process without the models' libraries in which it loads and stops.

The evaluation's tests (`test_evaluate.py`) import `model_support.py` and `evaluate.py` next to the package: the results files, the hand marks and labels with their forms on the fixtures, the measures on a hand-made layout, the rows of the similarity part, D2's rows per model and template with their deck rows with the fake pipeline, and a run on the test course's table, with and without the hand files, in a process where the viewer cannot be imported.

`model_support.py` finds the package from where it lies (the nearest folder above it that holds `app/`) and the real course table in `data/course/` next to it. None of the tests needs the course data or the viewer. Run them under either runner:

```bash
uv run python -m pytest tests/app
uv run python -m unittest discover -s tests/app
```
