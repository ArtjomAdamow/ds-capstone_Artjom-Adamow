# app: roadmap

The open points and the planned work of the models and their evaluation.

## The hand check of S1 (done 2026-10-30)

- **Done:** all 20 cards of `results/hand_check.md` are judged, 96 marks over the 173 pooled candidates of the seven variants. The chosen variant's hand-judged precision@5 is 0.68 (p@1 0.85; the pool's ceiling is 0.81), so most top-5 neighbours are relevant, as this section asked. The rule and the result are in the helper's concept (`docs/draft/helper/concept.md`, "S1 hand check"); the instruction, the noise and the per-setting evidence in `app/docs/S1_handcheck_analysis.md`.
- **The baseline:** 0.68 is what S2 (step 23) has to beat, by about 0.08 on these 20 cards, or by less on the grown sample.
- **The marks live in `results/`:** `hand_check.md` holds the marks of the variants' candidates, `jaccard_extra_marks.csv` the 30 Jaccard candidates judged beyond them (card, candidate, relevant, judged, judge, note). `results/` is tracked: its result rows are derived course data, its marks are hand labour.
- **The name:** the similarity model is S1 since 2026-10-30, and M2 becomes S2. `evaluate.py`'s rows, the model column of `results/results.csv` (and of its copy `scripts/input/data.csv`) and the drafts' older sections say M1 until the rename, step 0 of the workflow in `S1_handcheck_analysis.md`; the sections written since say S1.

## 11. A hand-labelled sample (next)

- **What:** a sample of unrated quizzes labelled easy, medium or hard by hand, starting with the 56 flattened quizzes, in a checklist file like `results/hand_check.md`. `evaluate.py` keeps the marks across runs and scores D1's predictions on the judged cards: accuracy, macro-F1 and weighted kappa, for the deck shares and for the cut points of the rated quizzes.
- **Why:** it is the only test score D1 can get, and it measures the absolute level of the unrated quizzes, which the deck shares leave open.
- **It also decides D1's feature set:** `text` (the better weighted kappa out of fold, 0.413) against `text+course` (0.369, the better macro-F1), judged on the hand labels. `text+course` stays the chosen set until then. The course features are S1's (the centrality, the first and fifth neighbour), so the decision is made again if S2 takes over the vectors (step 23).
- **The labels file:** the labels are kept the way the similarity part keeps its marks (step 17: a file of their own in `results/`, read before a rewrite and written after it, tracked), so a changed sample loses none.
- **Later:** a sample of plain flashcards.

## 12. Zero-shot difficulty tags, M3 (after step 11)

- **How:** Hugging Face zero-shot classification (natural language inference, NLP_2 and NLP_3). The levels easy, medium and hard become hypotheses ("This question is {}.", the template to be tuned), with no training.
- **Judged like D1:** on the rated quizzes, with the same metrics and the deck shares on its scores, and on the hand-labelled sample of step 11. Nothing is fitted, so it needs no cross-validation. Compared with D1 first, then tried as an extra feature of D1.
- **The dependency comes with step 23:** `sentence-transformers` installs `transformers` and `torch`, so M3 adds only its NLI model (for example `facebook/bart-large-mnli`, 1.6 GB) to the same install. **Open:** the model and the template.

## 15. The quizzes' answer options in D1 (later)

- **The data:** every quiz in `data/topics.json` carries `options` (the option texts), `correct` (the index of the correct option, or a list) and `answers` (one feedback text per option, the wrong ones included). The rated quizzes have four options and one correct one; only the unrated quizzes are many-choice: 3 to 8 options, 29 of the 56 with 2 to 5 correct ones. Today's flattened cards carry the correct options' text in the feedback and no wrong option.
- **Why D1 has to change:** D1 sees neither how many options a quiz has nor how close the wrong ones are to the right one, which is part of what makes a quiz hard; and the unrated quizzes, the ones D1 predicts, are exactly the many-choice ones.
- **Plan:**
  1. **The card carries its options:** the card record and the course table (new columns of `cards.csv`) carry `options`, `correct` and `answers`, so the models can read them. The similarity index may take the option texts too, as a vectoriser variant measured like the others (the unrated quizzes agree least with their topics: 0.48).
  2. **D1's features:** the number of options, the number of correct ones (many-choice or not), the length of the options, the cosine of the wrong options to the correct one (the closest distractor) and the length of the per-option feedback. Never the order of the options.
  3. **Many-choice stays a feature,** not a model of its own (29 cards are too few); the hand-labelled sample judges the predictions.
  4. **Later, a second model:** correct or wrong per option.
- **Done when:** D1's rows with the option features stand next to the rows without them in `difficulty.csv`, on the same cross-validation.

## 17. The card layouts measured with a fixed small k, with the evaluation's plumbing for S2 (next)

- **Why:** with k = min(5, n − 1) on topics of about 8 cards, random directions reach 0.76 within-topic overlap (the model 0.92, the SVD start 0.87), so the measure separates the layouts by 0.16 at most.
- **What:** `evaluate.py --part similarity` records the within-topic overlap at k = 2 and k = 3 (topics with fewer than k + 1 cards left out) for the model, the SVD start and random directions, with a column for the lift over random ((layout − random) / (1 − random)); the k = min(5, n − 1) column stays for the record. The topic layouts' trustworthiness (already computed per `TopicLayout`) is recorded next to it. The within-deck neighbours are not the measure: each topic has its own sphere, so a neighbour in another topic has no map distance.
- **With it, the evaluation changes S2 needs** (`evaluate.py` and `app/models/evaluation.py`), so S2 is judged on the map as well as on the marks:
  1. **A marks file of its own:** `results/hand_marks.csv` (card, candidate, relevant, judged, judge, note: the shape of `results/jaccard_extra_marks.csv`, whose rows merge into it), read before `write_hand_check` rewrites the candidate lists and written after it, so removing a variant or changing the sample loses no judgement. `hand_check.md` stays the judging form. Back the files up before the first run.
  2. **The paired difference:** the per-card difference of hand p@5 to the chosen variant, bootstrapped over the cards (5,000 draws), as a function next to `same_deck_precision` and a column of `results.csv`; hand p@1 and p@3 next to p@5.
  3. **The minimum score per vectoriser:** the percentiles of the fifth neighbour's cosine printed for every variant, not only the chosen one, and each variant's minimum set at the 10th percentile of its own (`SimilarityIndex.fit(..., min_score=...)`); dense vectors sit higher than TF-IDF's, so the fixed 0.1 cuts nothing for them.
  4. **A variant carries a vectoriser factory,** not TF-IDF options only, since `SimilarityIndex.fit` takes anything with `fit_transform`; every existing variant stays, because the chosen one is the reference every interval is paired against.
- **Checked when built:** the new columns in `results.csv`, the random floor well below the model at k = 2, a test of the fixed k and the lift on a hand-made layout; the 96 marks of 2026-10-30 unchanged after a rewrite with a variant removed (on a copy of the files), the chosen variant's interval against itself at zero, and every variant's minimum printed.

## 23. S2: sentence embeddings, hybrid with TF-IDF (after step 17)

- **Decided 2026-10-30:** semantic textual similarity with torch. `sentence-transformers` encodes each card's text (question, answer and explanation) once; the cosine of the unit rows is the similarity as in S1, and the hybrid sums S1's score and the embedding score, each normalised at its own minimum (step 17, point 3). Models: `all-MiniLM-L6-v2` first (384 dimensions, 91 MB, fast), `all-mpnet-base-v2` as the quality check (768 dimensions, 438 MB, about five times the cost). A `fit_transform` wrapper plugs the encoder into `SimilarityIndex.fit` as one more variant of `evaluate.py`.
- **Why:** S1's 30 misses are lexical traps, a shared dataset and generic terms, shared words without the concept or one concept in other words, which is what sentence encoders are trained to separate; no lexical measure of the list reached S1 (the concept's "The measures of the list"). The TF-IDF half keeps the exact-term matches (code names such as `word_tokenize`, `doesnt_match`), which embeddings match weakly; LSA showed the risk of smoothing toward deck-level topics (same-deck up, hand check down).
- **Judged** through the pool-judge-compare loop (`S1_handcheck_analysis.md`, "Improvement workflow"): the new candidates judged by the rule first, then the paired difference to S1 with its interval, hand p@1 kept, and the misses tagged (lexical trap, shared dataset, generic term) so the gain shows where it was expected. The same-deck precision serves the sweeps over settings; the hand check decides; a disagreement between the two is a finding, not a tie.
- **Grow the sample with it:** `HAND_CHECK_DECKS` at step 1 for decks 17 and 24 (70 cards, the interval about half of today's; step 1 keeps every judged card, step 2 would drop four of deck 24's), and deck 18 (AI Agents & RAG Systems, 42 cards) as the third deck, for transfer. 20 cards resolve about 0.08 and nothing smaller.
- **Dependencies** (resolved on 2026-10-30 with `uv pip compile` for Python 3.13 on Windows, nothing installed): `sentence-transformers` 6.2.0 brings `torch` 2.15.0, `transformers` 5.22.1, `tokenizers` 0.23.3, `safetensors` 0.8.0, `huggingface-hub` 2.4.0 and about 35 small packages, 46 in all; scikit-learn, scipy and numpy stay the env's versions. None of them is in the module env today.
  - **Sizes:** the Windows wheel of torch from PyPI is CPU-only, 127 MB. The Linux wheel is 539 MB and pulls the CUDA libraries (`nvidia-*`, several GB), so the capstone's CI and any Linux machine pin torch to the CPU index (`https://download.pytorch.org/whl/cpu`; with uv an `[[tool.uv.index]]` entry with `explicit = true` and a `[tool.uv.sources]` line for torch). transformers is 13 MB, sentence-transformers 1 MB.
  - **Models:** downloaded once from the Hugging Face hub into its cache (`HF_HOME`): MiniLM 91 MB, mpnet 438 MB, step 12's `bart-large-mnli` 1,629 MB. With `HF_HUB_OFFLINE=1` nothing is fetched again. The encoder runs locally, so the course data never leaves the machine.
  - **Where declared:** here, a dependency group `embeddings` in `modules/helper/pyproject.toml` (today: scikit-learn, scipy and the project), listed in `[tool.uv] default-groups` so the launcher's `uv sync --locked --check` keeps passing with it installed, after `uv lock` has updated `uv.lock`. In the capstone an extra (`app[embeddings]`) with the CPU index, so `uv sync` without it stays small and the model tests without torch still run on fixtures; a test that needs the encoder skips without it.
  - **Open:** the start-up cost: encoding the 1,064 cards at every start, next to the 1.7 s fit, or a cache of the embeddings as a derived file of the course.
- **Done when:** the hybrid's paired interval against S1 excludes zero on the grown sample without losing the first neighbour (hand p@1), with every variant's minimum at its own 10th percentile and the misses tagged. Then the draft roadmap's sentence holds: the better one provides the neighbours and the positions, so D1 is refitted (`--part difficulty`; its course features are the index's) and T1 and the map measures are recorded again (`--part topics`; step 17's fixed-k measure) in the same run.
- **Reuse:** the encoder is the content judge of the image texts and the request half of the navigation prompt; what transfers is noted apart, in `docs/draft/helper/prompt.md` (not yet a concept).

## What would improve D1's predicted groups

The predicted groups are a tentative order within a deck, not a level.

- **Code features:** lines of code, nesting depth, operators, slicing, the number of answer options.
- **Zero-shot tags** (step 12).
- **After the hand labels:** calibrating the levels across decks instead of forcing the shares on 1 to 7 extra quizzes.

## T1's upgrade path

- Nearest centroid is one k-means step from the scripted seed: the upgrade path is to iterate the centroids (spherical k-means, the scripted topics still never stored), and the zero-shot tags of M3 are the judge that does not learn from the script.
- The vectors are the second lever: T1's centroids are means of the similarity model's unit rows, so S2's vectors (step 23) give other centroids. `--part topics` measures the agreement for either vectoriser, and the hand-check decks 17 and 24 stay the evaluation set.

## The offline evaluation

`evaluate.py [--part similarity|difficulty|topics]` measures the models of `app/models/`. It is not part of the app yet: it comes once it reads the course table (`course_table.load_table`, the same cards, deck ids and seeds) and draws the random floor B0 and the random layouts itself, a uniform draw on the sphere (the same distribution, other digits). What it measures:

- `--part topics` measures T1 against the scripted topics of `data/topics.json`: the agreement of the nearest centroid rule next to the neighbour rule, a logistic regression per deck and chance, over the course, by stack and per deck; the margin table (the cards kept at a threshold and their agreement), the coherence per topic, the moved cards and the emptied topics. One row per model and scope goes to `results/topics.csv`.
- `--part difficulty` measures only D1. It appends one row per feature set and label rule, plus the two baselines, to `results/difficulty.csv`: out of fold on the rated quizzes, as mean ± sd over the repeats, and the predicted levels of the unrated quizzes.
- `--part similarity` measures the random floor (B0), the TF-IDF variants (same-deck precision@5, coverage, distinct recommended cards) and the chosen model's map (trustworthiness and neighbour overlap of the deck layout, the within-topic overlap of the card layouts over the model's topics), appends one row per model and variant to `results/results.csv`, and writes the hand-check candidates for 20 NLP cards to `results/hand_check.md`, to be judged by hand: mark every relevant candidate with `x`, or the `none` line when nothing fits, by the rule in the helper's concept ("S1 hand check"). Later runs keep the marks and add every variant's hand-judged precision@5 over the judged cards (`hand_precision` in the results); the check of 2026-10-30 is complete (0.68 for the chosen variant). The same-deck precision serves the sweeps over settings, the hand check decides; a disagreement between the two is a finding (LSA: same-deck up, hand check down).
- Without `--part` all three parts run. The rows note the course's decks ("decks 01 to 24"); older rows of the 48-deck course stay above them. `results/` is tracked: the result rows are derived course data, the hand marks (`hand_check.md`, `jaccard_extra_marks.csv`, with step 17 `hand_marks.csv`) are hand labour.

## Later

- **The prompt side** (`docs/draft/helper/prompt.md`, a note, not yet a concept): a language model writes the "The image shows …" texts for the 17 image cards without one, with S1's index as the example retriever and the hand-check method as the evaluation, and a prompt navigates a request to a topic (T1's topics). S2's encoder (step 23) is the content judge of the texts and the request half of the retriever; the libraries that serve both are listed there.
- **Retrieval over the notebooks and slides,** from a card to the cell or slide that teaches it: where the lexical measures have their jobs (fuzzy matching of headings to topic names, TF-IDF or BM25 as the lexical half of a hybrid retriever over chunks, ROUGE, BLEU and METEOR for generated answers against card answers). The hand-check method carries over unchanged: pooled candidates, the rule, hand p@5.
