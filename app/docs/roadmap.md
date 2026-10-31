# app: roadmap

The open points and the planned work of the models, their evaluation and the split for the capstone (step 21). How each model works, how it is judged, its numbers and the analysis behind the steps are in `docs/draft/helper/models/`, one file per model: `similarity.md` (S1, S2), `course.md` (C1), `topic.md` (T1), `difficulty.md` (D1, D2) and `prompt.md` (the prompt side). Note: the documentation will be updated and included in the final docs.

## The hand check of S1 (done 2026-10-30)

- **Done:** all 20 cards of `results/hand_check.md` are judged, 96 marks over the 173 pooled candidates. The chosen variant's hand-judged precision@5 is 0.68 (p@1 0.85), what S2 (step 23) has to beat, by about 0.08 on these 20 cards, or by less on the grown sample. The rule, the noise, the per-setting evidence and the improvement workflow: `similarity.md`.
- **The marks live in `results/`:** `hand_check.md` holds the marks of the variants' candidates, `jaccard_extra_marks.csv` the 30 Jaccard candidates judged beyond them. `results/` is tracked: its result rows are derived course data, its marks are hand labour.
- **Open, the name:** the similarity model is S1 since 2026-10-30, and M2 is S2. `evaluate.py`'s rows and the model column of `results/results.csv` (and of its copy `scripts/input/data.csv`) say M1 until the rename, step 0 of the workflow in `similarity.md`.

## 11. A hand-labelled sample (next)

- **What:** a sample of unrated quizzes labelled easy, medium or hard by hand, starting with the 56 flattened quizzes, in a checklist file like `results/hand_check.md`. `evaluate.py` keeps the marks across runs and scores D1's predictions on the judged cards: accuracy, macro-F1 and weighted kappa, for the deck shares and for the cut points of the rated quizzes.
- **Why:** it is the only test score D1 can get, and it measures the absolute level of the unrated quizzes, which the deck shares leave open (`difficulty.md`, "Quality of the predicted groups"). The levels are a quota of the writing, so the sample needs a criterion of its own (`difficulty.md`, "The data").
- **It also decides D1's feature set:** `text` against `text+course` (`difficulty.md`, "The numbers"), judged on the hand labels; `text+course` stays the chosen set until then, and the decision is made again if S2 takes over the vectors (step 23).
- **The labels file:** the labels are kept the way the similarity part keeps its marks (step 17: a file of their own in `results/`, read before a rewrite and written after it, tracked), so a changed sample loses none.
- **Later:** a sample of plain flashcards.

## 12. Zero-shot difficulty tags, D2 (after step 11)

- **How:** zero-shot classification with an NLI model, the levels as hypotheses, no training (`difficulty.md`, "D2").
- **Judged like D1:** on the rated quizzes, with the same metrics and the deck shares on its scores, and on the hand-labelled sample of step 11. Nothing is fitted, so it needs no cross-validation. Compared with D1 first, then tried as an extra feature of D1.
- **The dependency comes with step 23:** `sentence-transformers` installs `transformers` and `torch`, so D2 adds only its NLI model (for example `facebook/bart-large-mnli`, 1.6 GB) to the same install. **Open:** the model and the template.

## 15. The quizzes' answer options in D1 (later)

- **Why:** D1 sees neither how many options a quiz has nor how close the wrong ones are to the right one, and the quizzes it predicts are exactly the many-choice ones (`difficulty.md`, "The data", answer options, and "What would improve the predicted groups").
- **Plan:**
  1. **The card carries its options:** the card record and the course table (new columns of `cards.csv`) carry `options`, `correct` and `answers`, so the models can read them; further columns come when the dataset grows. The viewer's parser reads them already, apart from the card (`CardsFile.quiz`: a `Quiz` with `options`, `answers` and `correct`), so no change of the viewer is needed: `export_course.py` writes them into the table, and `glue/course.py` hands them to the card record in mode (b). The similarity index may take the option texts too, as a vectoriser variant measured like the others (the unrated quizzes agree least with their topics: 0.48).
  2. **D1's features:** the number of options, the number of correct ones (many-choice or not), the length of the options, the cosine of the wrong options to the correct one (the closest distractor) and the length of the per-option feedback. Never the order of the options.
  3. **Many-choice stays a feature,** not a model of its own (29 cards are too few); the hand-labelled sample judges the predictions.
  4. **Later, a second model:** correct or wrong per option (F2).
- **Done when:** D1's rows with the option features stand next to the rows without them in `difficulty.csv`, on the same cross-validation.

## 17. The card layouts measured with a fixed small k, with the evaluation's plumbing for S2 (next)

- **Why:** the within-topic overlap at k = min(5, n − 1) separates the layouts by 0.16 at most, random included (`course.md`, "The numbers").
- **What:** `evaluate.py --part similarity` records the within-topic overlap at k = 2 and k = 3 (topics with fewer than k + 1 cards left out) for the model, the SVD start and random directions, with a column for the lift over random ((layout − random) / (1 − random)); the k = min(5, n − 1) column stays for the record. The topic layouts' trustworthiness (already computed per `TopicLayout`) is recorded next to it.
- **With it, the evaluation changes S2 needs** (`evaluate.py` and `app/models/evaluation.py`), so S2 is judged on the map as well as on the marks (`similarity.md`, "Improvement workflow"):
  1. **A marks file of its own:** `results/hand_marks.csv` (card, candidate, relevant, judged, judge, note: the shape of `results/jaccard_extra_marks.csv`, whose rows merge into it), read before `write_hand_check` rewrites the candidate lists and written after it, so removing a variant or changing the sample loses no judgement. `hand_check.md` stays the judging form. Back the files up before the first run.
  2. **The paired difference:** the per-card difference of hand p@5 to the chosen variant, bootstrapped over the cards (5,000 draws), as a function next to `same_deck_precision` and a column of `results.csv`; hand p@1 and p@3 next to p@5.
  3. **The minimum score per vectoriser:** the percentiles of the fifth neighbour's cosine printed for every variant, not only the chosen one, and each variant's minimum set at the 10th percentile of its own (`SimilarityIndex.fit(..., min_score=...)`); dense vectors sit higher than TF-IDF's, so the fixed 0.1 cuts nothing for them.
  4. **A variant carries a vectoriser factory,** not TF-IDF options only, since `SimilarityIndex.fit` takes anything with `fit_transform`; every existing variant stays, because the chosen one is the reference every interval is paired against.
- **Checked when built:** the new columns in `results.csv`, the random floor well below the model at k = 2, a test of the fixed k and the lift on a hand-made layout; the 96 marks of 2026-10-30 unchanged after a rewrite with a variant removed (on a copy of the files), the chosen variant's interval against itself at zero, and every variant's minimum printed.

## 21. The split for the capstone: the rest of step 3 (to check in the capstone)

- **Built upstream** (spatial_study, `archive/helper/done_roadmap.md`, steps 21 and 21.3): the split into the models and the glue, the viewer part merged into sputnik (v0.2.0, the Helper dialog's part v0.3.0), the two modes behind one launcher (`app/launch.py`, `start.py`), the state files in `data/state/`, the course table and the tests on fixtures.
- **Done when,** checked in the capstone:
  1. `uv sync` needs no private access, and the tests on fixtures pass (the models', the standalone page's and the launchers').
  2. With the course data, mode (a) serves the standalone page.
  3. With the course data, `evaluate.py` appends the results rows. Open upstream: it does not travel yet ("The offline evaluation", "Open").
  4. With the viewer extra, mode (b) shows the Helper button and its dialog, with no delta hook and no launcher redirects.
- **The quiz options in the course table** (with step 15): step 15, point 1.

## 23. S2: sentence embeddings, hybrid with TF-IDF (after step 17)

- **Decided 2026-10-30:** semantic textual similarity with torch: `sentence-transformers` encodes each card's text once, `all-MiniLM-L6-v2` first and `all-mpnet-base-v2` as the quality check, and the hybrid sums S1's score and the embedding score, each normalised at its own minimum (step 17, point 3). A `fit_transform` wrapper plugs the encoder into `SimilarityIndex.fit` as one more variant of `evaluate.py`. Why, and the measures it was chosen over: `similarity.md`, "S2" and "The measures of the list".
- **Judged** through the pool-judge-compare loop (`similarity.md`, "Improvement workflow"): the new candidates judged by the rule first, then the paired difference to S1 with its interval, hand p@1 kept, and the misses tagged (lexical trap, shared dataset, generic term) so the gain shows where it was expected.
- **Grow the sample with it:** `HAND_CHECK_DECKS` at step 1 for decks 17 and 24 (70 cards, the interval about half of today's; step 1 keeps every judged card, step 2 would drop four of deck 24's), and deck 18 (AI Agents & RAG Systems, 42 cards) as the third deck, for transfer. 20 cards resolve about 0.08 and nothing smaller.
- **Dependencies:** `sentence-transformers` 6.2.0 with `torch` 2.15.0, 46 packages in all, none of them in the module env today (the versions, the sizes and the models' downloads: `similarity.md`, "S2").
  - **Where declared:** here, a dependency group `embeddings` in `modules/helper/pyproject.toml` (today: scikit-learn, scipy and the project), listed in `[tool.uv] default-groups` so the launcher's `uv sync --locked --check` keeps passing with it installed, after `uv lock` has updated `uv.lock`. In the capstone an extra (`app[embeddings]`) with the CPU index, so `uv sync` without it stays small and the model tests without torch still run on fixtures; a test that needs the encoder skips without it.
  - **Torch on Linux:** the capstone's CI and any Linux machine pin torch to the CPU index (`https://download.pytorch.org/whl/cpu`; with uv an `[[tool.uv.index]]` entry with `explicit = true` and a `[tool.uv.sources]` line for torch), since the Linux wheel pulls the CUDA libraries.
  - **Open:** the start-up cost: encoding the 1,064 cards at every start, next to the 1.7 s fit, or a cache of the embeddings as a derived file of the course.
- **Done when:** the hybrid's paired interval against S1 excludes zero on the grown sample without losing the first neighbour (hand p@1), with every variant's minimum at its own 10th percentile and the misses tagged. Then the draft roadmap's sentence holds: the better one provides the neighbours and the positions, so D1 is refitted (`--part difficulty`; its course features are the index's) and T1 and the map measures are recorded again (`--part topics`; step 17's fixed-k measure) in the same run.
- **Reuse:** the encoder is the content judge of the image texts and the request half of the navigation prompt (`prompt.md`).

## The offline evaluation

`evaluate.py [--part similarity|difficulty|topics]` measures the models of `app/models/` and appends one row per model and variant to `results/`: `--part similarity` S1's variants, the random floor B0 and C1's map (`results.csv`, and the hand-check candidates in `hand_check.md`), `--part topics` T1 (`topics.csv`), `--part difficulty` D1 (`difficulty.csv`); what each part measures is in the model's file, "How it is judged" or "The measurement". Without `--part` all three parts run. The rows note the course's decks ("decks 01 to 24"); older rows of the 48-deck course stay above them. `results/` is tracked: the result rows are derived course data, the hand marks (`hand_check.md`, `jaccard_extra_marks.csv`, with step 17 `hand_marks.csv`) are hand labour.

- **Open (step 21): it is not part of the app yet.** It reads the course file with the viewer's parser through the glue (`glue/course.py`), and the random floor B0 and the random layouts use the viewer's `surface_positions` (through `glue/positions.random_points`). It comes into the package once it draws them itself, a uniform draw on the sphere (the same distribution, other digits), or B0 is recorded from mode (b); the switch to the course table is then one function (`course_table.load_table` gives the same cards, deck ids and seeds).

## Later

- **D1's predicted groups:** code features, the zero-shot tags (step 12) and, after the hand labels, calibrating the levels across decks instead of forcing the shares on 1 to 7 extra quizzes (`difficulty.md`, "What would improve the predicted groups").
- **T1's upgrade path:** iterating the centroids (spherical k-means from the scripted seed), a zero-shot judge that does not learn from the script, and S2's vectors (`topic.md`, "The upgrade path").
- **The prompt side** (`prompt.md`, a note, not yet a concept): a language model writes the "The image shows …" texts for the 17 image cards without one, and a prompt navigates a request to a topic (T1's topics).
- **Retrieval over the notebooks and slides,** from a card to the cell or slide that teaches it, where the lexical measures have their jobs; the hand-check method carries over unchanged (`similarity.md`, "The measures of the list").
