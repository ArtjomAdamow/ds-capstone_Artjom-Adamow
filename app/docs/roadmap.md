# app: roadmap

The open points and the planned work of the helper's draft documents (its roadmap, concept and architecture) that do not depend on the viewer, relocated from them: the models, their evaluation, the data package and the move into the capstone. The points that need the viewer (the glue over its hooks, the viewer's own features) are in the helper's draft roadmap of the flashcards workspace (`docs/draft/helper/roadmap.md`), what is built in its archived roadmap (`archive/helper/roadmap.md`). Step numbers are the helper roadmap's.

## The hand check of M1 (pending)

- **Pending:** the marks of `results/hand_check.md` (20 cards of decks 17 and 24 with the candidates of all variants; the rule in the helper's concept, "M1 numbers").
- **Done when** most top-5 neighbours in the hand check are relevant. That result is the baseline M2 has to beat.

## 11. A hand-labelled sample (next)

- **What:** a sample of unrated quizzes labelled easy, medium or hard by hand, starting with the 56 flattened quizzes, in a checklist file like `results/hand_check.md`. `evaluate.py` keeps the marks across runs and scores D1's predictions on the judged cards: accuracy, macro-F1 and weighted kappa, for the deck shares and for the cut points of the rated quizzes.
- **Why:** it is the only test score D1 can get, and it measures the absolute level of the unrated quizzes, which the deck shares leave open.
- **It also decides D1's feature set:** `text` (the better weighted kappa out of fold, 0.413) against `text+course` (0.369, the better macro-F1), judged on the hand labels. `text+course` stays the chosen set until then.
- **Later:** a sample of plain flashcards.

## 12. Zero-shot difficulty tags, M3 (after step 11)

- **How:** Hugging Face zero-shot classification (natural language inference, NLP_2 and NLP_3). The levels easy, medium and hard become hypotheses ("This question is {}.", the template to be tuned), with no training.
- **Judged like D1:** on the rated quizzes, with the same metrics and the deck shares on its scores, and on the hand-labelled sample of step 11. Nothing is fitted, so it needs no cross-validation. Compared with D1 first, then tried as an extra feature of D1.
- **Open:** the dependency (`transformers` with `torch` is a large install; the launcher's env check asks before installing) and the NLI model (for example `facebook/bart-large-mnli`).

## 15. The quizzes' answer options in D1 (later)

- **The data:** every quiz in `data/topics.json` carries `options` (the option texts), `correct` (the index of the correct option, or a list) and `answers` (one feedback text per option, the wrong ones included). The rated quizzes have four options and one correct one; only the unrated quizzes are many-choice: 3 to 8 options, 29 of the 56 with 2 to 5 correct ones. Today's flattened cards carry the correct options' text in the feedback and no wrong option.
- **Why D1 has to change:** D1 sees neither how many options a quiz has nor how close the wrong ones are to the right one, which is part of what makes a quiz hard; and the unrated quizzes, the ones D1 predicts, are exactly the many-choice ones.
- **Plan:**
  1. **The card carries its options:** the parser reads `options`, `correct` and `answers` into the card (a change of the project, proposed as a patch); the helper follows it. The similarity index may take the option texts too, as a vectoriser variant measured like the others (the unrated quizzes agree least with their topics: 0.48).
  2. **D1's features:** the number of options, the number of correct ones (many-choice or not), the length of the options, the cosine of the wrong options to the correct one (the closest distractor) and the length of the per-option feedback. Never the order of the options.
  3. **Many-choice stays a feature,** not a model of its own (29 cards are too few); the hand-labelled sample judges the predictions.
  4. **Later, a second model:** correct or wrong per option.
- **Done when:** D1's rows with the option features stand next to the rows without them in `difficulty.csv`, on the same cross-validation.

## 17. The card layouts measured with a fixed small k (next, with the next evaluation change)

- **Why:** with k = min(5, n − 1) on topics of about 8 cards, random directions reach 0.76 within-topic overlap (the model 0.92, the SVD start 0.87), so the measure separates the layouts by 0.16 at most.
- **What:** `evaluate.py --part similarity` records the within-topic overlap at k = 2 and k = 3 (topics with fewer than k + 1 cards left out) for the model, the SVD start and random directions, with a column for the lift over random ((layout − random) / (1 − random)); the k = min(5, n − 1) column stays for the record. The topic layouts' trustworthiness (already computed per `TopicLayout`) is recorded next to it. The within-deck neighbours are not the measure: each topic has its own sphere, so a neighbour in another topic has no map distance.
- **Checked when built:** the new columns in `results.csv`, the random floor well below the model at k = 2, a test of the fixed k and the lift on a hand-made layout.

## What would improve D1's predicted groups

The predicted groups are a tentative order within a deck, not a level (the helper's concept, "Quality of the predicted groups").

- **Code features:** lines of code, nesting depth, operators, slicing, the number of answer options.
- **Zero-shot tags** (step 12).
- **After the hand labels:** calibrating the levels across decks instead of forcing the shares on 1 to 7 extra quizzes.

## T1's upgrade path

- Nearest centroid is one k-means step from the scripted seed: the upgrade path is to iterate the centroids (spherical k-means, the scripted topics still never stored), and the zero-shot tags of M3 are the judge that does not learn from the script.

## 21. The split for the capstone: the side without the viewer

- **Built** (the helper roadmap's steps 21.1 and 22): the card record (`cards.py`, the seven fields of `Flashcard` as a protocol), the models importing nothing of the viewer, the model tests on a table fixture (`tests/model_support.py`; `difficulty_course()` builds its cards by hand); the course table (`course_table.py`, written by `modules/helper/export_course.py` with the viewer's parser into `data/course/`: `cards.csv`, `decks.csv`, `topics.csv`); the shared figures (`figures.py`); the launcher's mode question (`launch.py`); and mode (a), the standalone page (`standalone/`, a Dash page on the course table), whose tests import nothing of the viewer.
- **The course data in the capstone** (decided 2026-10-29): it travels from this workspace with every train of the migration (`scripts/migrate_helper.py`, `docs/draft/helper/protocol_export.md`). The course file `topics.json`, the course table, the images and the notebooks go into the capstone's git-ignored `data/` whenever they change. The migration checks that the table equals an export of the course file. A data package of its own is no longer needed; whether it still comes is open (below). New columns come when the dataset grows (the quiz options first, step 15). The models read the table, so they need no parser; the viewer reads the raw file, and the ids agree because both files come from the same parser version. The capstone's tests run on fixtures without the data.
- **The capstone's launcher** (built 2026-10-29): `start.py`, developed here next to the package and exported with it (readme, "The capstone's launcher"): no arguments, the course table's folder for `standalone.main` and the state folder for mode (b) written in it, `choose_mode` from `launch.py` asked only when sputnik is found, a stop with the command for a missing library, missing data or a sputnik older than `launch.VIEWER_RELEASE`; none of this workspace's delta layer, `temp/` and redirects.
- **The export** (`docs/draft/helper/protocol_export.md`): the package (renamed from `guided_learning` to `app` on 2026-10-29), `start.py`, the tests without the viewer and the changed course data go to the capstone with each train of `scripts/migrate_helper.py`, together with `migrate_train.md`, the import side's instructions; the export side owns them, the capstone adapts its own files and makes its own commits.
- **The capstone's dependencies:** scipy, dash and plotly next to scikit-learn in its `pyproject.toml` (the module env has dash and plotly through the viewer package).
- **Tests:** the model tests, the standalone page's and the launchers' tests on fixtures everywhere (the capstone's CI has no private access).
- **Step 3 of the order, the move:** each train drops the package at the capstone's root (`app/`, next to `start.py`), with the course data when it changed.
- **Done when** (step 3, in the capstone): `uv sync` needs no private access and the model tests pass; with the course data, mode (a) serves the page and `evaluate.py` appends the results rows.
- **Open:**
  - **B0 without the viewer:** the random floor B0 and the random layouts of `evaluate.py` use the project's `surface_positions` (through `glue/positions.random_points`); mode (a) needs its own uniform draw on the sphere (the same distribution, other digits) or records B0 from mode (b).
  - **`evaluate.py` on the table:** it reads the course file with the viewer's parser through the glue (`glue/course.py`); `course_table.load_table` gives the same cards, deck ids and seeds, so the switch is one function once B0 has its own draw.
  - **The data package:** whether it still comes, now that the data travels with the migration; if so, its name, and whether it is an extra.
