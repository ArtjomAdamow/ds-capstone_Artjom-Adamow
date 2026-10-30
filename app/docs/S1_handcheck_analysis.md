# Similarity model S1

S1 is the helper's similarity model: one TF-IDF over all cards of the course (the question, the answer and the explanation; English stop words, unigrams and bigrams in at least two cards, sublinear term frequency), unit rows, and the cosine as the similarity, which gives every card its five nearest cards above a minimum score of 0.1 (`app/models/similar.py`). Its vectors are the input of every other model: T1 forms its topic centroids from them, the course model's deck vectors, layouts and clustered connections come from them, and D1's course features (the centrality and the first and fifth neighbour's cosine) are read off S1's neighbours, so a change of S1 moves all of them and is measured first.

Until step 0 of the workflow below is done, S1 is called M1 in `evaluate.py`, in `results/results.csv` and in the drafts. The numbers in this document are from the first full judging of the hand check on 2026-10-30 (the rows of that date in `results/results.csv`, the course of decks 01 to 24).

## Handcheck

The hand check is the independent judge of S1's vectoriser. The same-deck precision@5 over the course says whether the vectors capture decks; the hand check says whether a learner accepts the neighbours. Do it in this order.

**1. Get the candidates.** Run the similarity part of the evaluation in the module env, from the repository root:

```bash
modules/helper/.venv/Scripts/python.exe modules/helper/evaluate.py --part similarity
```

It writes `results/hand_check.md`: every fourth card of deck 17 and every third card of deck 24 (`HAND_CHECK_DECKS`, 20 cards) and, under each, the union of all variants' top-5 neighbours above the minimum score (pooling, as in TREC: 173 candidates in all), each with the cosine of every variant that listed it.

**2. Judge every card** by the rule written down in the workflow below ("Write the judging rule down"): put `x` into the `[ ]` of every candidate a learner would accept, or mark the `none` line when nothing fits. A card without any mark counts as not judged; an unmarked candidate counts as irrelevant for every variant. Judge on the full texts of the card and its candidates (question, answer and explanation, in `data/course/cards.csv`), not on the shortened questions in the file: the answer carries the concept. On 2026-10-30 all 20 cards were judged: 96 of the 173 candidates marked, 2 to 7 per card, no card without a relevant candidate.

**3. Run the evaluation again.** It keeps the marks, rewrites the candidate lists and appends `hand_precision` (the hand-judged precision@5 over the judged cards, with the minimum score applied) and `judged_cards` to every variant's row in `results/results.csv`.

**4. Read the numbers against the noise.** The seven variants of `evaluate.py`; the last column is the paired bootstrap of the difference to the chosen variant over the 20 cards (5,000 draws, a scratch script, not in `evaluate.py` yet):

| Variant | Hand p@5 | Hand p@1 | Same-deck p@5, min score | Difference to chosen, 95 % interval |
|---|---|---|---|---|
| unigrams, min_df=2 | 0.610 | 0.80 | 0.636 | −0.07 [−0.16, +0.01] |
| uni+bigrams, min_df=1 | 0.560 | 0.85 | 0.480 | −0.12 [−0.23, −0.02] |
| uni+bigrams, min_df=2 | 0.630 | 0.80 | 0.636 | −0.05 [−0.14, +0.03] |
| uni+bigrams, min_df=3 | 0.620 | 0.85 | 0.641 | −0.06 [−0.19, +0.06] |
| uni+bigrams, min_df=2, explain | 0.660 | 0.90 | 0.665 | −0.02 [−0.09, +0.04] |
| uni+bigrams, min_df=2, sublinear | 0.670 | 0.85 | 0.644 | −0.01 [−0.10, +0.07] |
| **chosen:** uni+bigrams, min_df=2, sublinear, explain | **0.680** | 0.85 | 0.679 | — |

- **The resolution:** one slot of one card is 0.01 of hand p@5, one whole card 0.05. The standard error of the chosen variant over the 20 cards is 0.06, its 95 % interval [0.56, 0.80].
- **The ceiling is 0.81, not 1:** eight of the 20 cards have fewer than five relevant candidates in the whole pool (2 to 4), so a perfect ranking cannot fill their five slots.
- **Only a difference whose interval excludes zero counts.** Here that is `min_df=1` alone (and the question weighting of the next step). Everything else is within noise. A new model has to beat the chosen variant by about 0.08 on these 20 cards to show anything.
- **The chosen variant is even across the two decks** (deck 17 0.67, deck 24 0.69); the variant with the explanation but without sublinear tf is 0.60 / 0.73.

**5. Keep the chosen settings; one simplification is open.** What the hand check says about each setting (the variants marked "scratch" were computed outside `evaluate.py` with the same neighbours and minimum score):

| Setting | Evidence | Decision |
|---|---|---|
| the explanation in the text | +0.03 without sublinear tf, +0.01 with it; the best hand p@1 (0.90) | keep |
| sublinear tf | +0.04 without the explanation, +0.02 with it | keep |
| min_df=2 | min_df=1: −0.12, the one significant loss (one-off terms pull weak matches); min_df=3: no difference | keep |
| bigrams | chosen with unigrams only (scratch): hand 0.690, same-deck 0.687, +0.01 [−0.02, +0.05], about half the vocabulary | no measurable gain: the one simplification the sample supports; decide it on a larger sample |
| the question weighted ×2 / ×3 (scratch) | hand 0.640 / 0.620; ×2 is significant, −0.04 [−0.08, −0.01]; without the explanation and ×2: 0.610 | reject: the answer carries the concept, the question only its wording |
| LSA with 100 / 200 / 300 components on the chosen TF-IDF (scratch) | hand 0.650 / 0.650 / 0.660, same-deck 0.712 / 0.700 / 0.689 | not adopted: the two measures disagree; smoothing toward deck-level topics loses the exact concept matches |
| character 3- to 5-grams within words (scratch) | hand 0.670, hand p@1 0.95 (the best first neighbour), same-deck 0.690 | not adopted alone; the candidate for code names in a hybrid |
| max_df=0.3 (scratch) | hand 0.680, same-deck 0.678 | no effect |

**6. Where S1 fails.** The chosen variant's 100 slots: 68 hits, 30 misses, 2 empty (417029 and 324025 have four neighbours above the minimum). Of the 30 misses 17 come from other decks and 13 from the card's own deck. They are of three kinds:

- **Lexical traps:** the ice-cream overfitting card 408018 (deck 08) for the n-gram card 417009; the neural-network layer cards 115031 and 115032 (deck 15) for how embeddings are learned (417021); the stacking recipe 412028 (deck 12) for the training stages of an LLM (424004); the pooling card 424011 for the stages of one next token (324022).
- **A shared dataset, no shared concept:** notebook 1's spam cards (CountVectorizer 317034, AUC 317041) next to notebook 2's zero-shot card 317037, in both directions.
- **A generic term:** "large language model" pulls the agent-tools card 418006 and the NLP-subfield card 303039 onto both LLM definitions (424001, 324016); "zero-shot" and "few-shot" pull the zero-shot cards (317037, 417030) and the few-shot prompting card 324025 together.

The chosen variant lists 71 % of the 96 relevant pairs in its top 5 (recall@5 0.71, hand p@3 0.78); the other variants found the rest, most of them for 317041 (the evaluation-metric cards of deck 10), 424001 and 324016 (the other LLM cards of deck 24).

**7. Where the relevant pairs sit,** and why the same-deck measure stays a proxy:

| Place of the 96 relevant pairs | Count | Share |
|---|---|---|
| the same scripted topic | 28 | 29 % |
| the same deck, another topic | 32 | 33 % |
| another deck | 36 | 38 % |

The same-deck measure counts the 36 cross-deck pairs as misses: the cosine cards of decks 11 and 18 next to deck 24's, the metric cards of deck 10 next to the AUC card of deck 17, the reinforcement-learning cards of deck 03 next to the RLHF stage card. It still orders the TF-IDF variants like the hand check (Spearman 0.93 over the seven variants) but not the scratch variants (0.64 over all twelve): LSA wins on same-deck and loses on the hand check.

## The measures of the list, judged on the hand check

The list discussed on 2026-10-30: BLEU, ROUGE, METEOR, Jaccard similarity and Levenshtein distance / fuzzy matching as the lexical group; cosine similarity, semantic textual similarity, Euclidean distance and Word Mover's Distance as the semantic group. Every measure that the module env's libraries allow (scikit-learn's count vectors, `difflib`) was computed as a card-to-card score for the 20 judged cards against all other cards, and its top 5 scored against the marks (a scratch script). A candidate nobody judged counts as a miss (pooling), so a new measure's hand p@5 is a lower bound and the share of unjudged slots stands next to it. Same-deck p@5 here is over the 20 cards' top 5 only (S1 0.54), not the course's 0.68.

| Measure | Library, briefly | Measured on the 20 cards (S1: hand p@5 0.68, same-deck 0.54) | Verdict |
|---|---|---|---|
| BLEU | n-gram precision of a hypothesis against a reference with a brevity penalty. `sacrebleu` is the standard, `nltk.translate.bleu_score` the sentence version with smoothing. | BLEU-2 ≥ 0.40, 56 % of the slots unjudged, same-deck 0.33 | No. A generation metric, asymmetric by design; its place is the RAG's generated answer against the card's answer. |
| ROUGE | Recall-oriented n-gram or longest-common-subsequence overlap for summaries. `rouge-score` (Google) gives ROUGE-1/2/L with stemming. | ROUGE-1 F1 ≥ 0.43, 46 % unjudged, same-deck 0.40; ROUGE-2 F1 ≥ 0.40, same-deck 0.26 | No, the same place as BLEU. |
| METEOR | Unigram matching with stemming and WordNet synonyms, plus a fragmentation penalty. `nltk.translate.meteor_score`, with the WordNet corpus. | not measured (no nltk in the env) | No. Synonyms are a small step toward meaning at word level; still a generation metric. |
| Jaccard | Token-set overlap: a set operation, `sklearn.metrics.jaccard_score` on binary vectors, or `textdistance`. | 0.64 after judging its 30 unjudged candidates by the rule (11 relevant, `results/jaccard_extra_marks.csv`), same-deck 0.47 | No. S1 without IDF weighting, within noise and slightly below. Useful for near-duplicates across stacks (the two "What is Hugging Face" cards). |
| Levenshtein, fuzzy | Character edit distance. `rapidfuzz` is the maintained MIT drop-in for fuzzywuzzy/thefuzz; `difflib.SequenceMatcher` in the standard library gives the same ratio slowly. | ratio on the tokens of the full text ≥ 0.20, 78 % unjudged, same-deck 0.17; on the question's characters ≥ 0.15, same-deck 0.16 | No. It ranks by length and boilerplate. For duplicates across stacks and for matching heading titles to topic names. |
| Cosine | S1's measure: the dot product of unit rows, `sklearn.metrics.pairwise.cosine_similarity` elsewhere. | 0.68 | The vectors are the lever, not the measure. |
| Euclidean | `sklearn.metrics.pairwise.euclidean_distances`. On unit rows the squared distance is 2 − 2 cos, the same ranking. | 0.68, the identical top 5 | Nothing by construction. On unnormalised rows a length effect, a regression. |
| Semantic textual similarity | `sentence-transformers`: `all-MiniLM-L6-v2` (384 dimensions, fast), `all-mpnet-base-v2` (768, better, about five times the cost). Encode once, cosine as before; a `fit_transform` wrapper plugs into `SimilarityIndex.fit`. | not measured (needs torch) | The one candidate with an expected gain: S1's three error classes are "shared words, another concept" or "one concept, other words", what these models are trained to separate. Risks: smoothing toward deck level as LSA did, weaker code-name matches; hence a hybrid with TF-IDF. |
| Word Mover's Distance | `gensim`'s `wmdistance` on pretrained word2vec or GloVe vectors from its downloader, with the `POT` optimal-transport solver installed separately. | not measured | Second choice: word-level semantics at cubic cost per pair, code names out of vocabulary, usually behind sentence embeddings on STS benchmarks. Only if torch is unwanted. |

What follows from it:

- **Do not add the lexical measures as variants of S1.** They are overlap without IDF weighting, or generation metrics; all measured below S1 or far below. S1's misses are lexical in nature (shared words, another concept), which more lexical matching cannot fix.
- **Add sentence embeddings as S2,** hybrid with TF-IDF (each score normalised at its own minimum, see "Refit the minimum score per vectoriser"), through the pool-judge-compare loop below. Expected: the lexical traps and the generic terms fixed, the exact-term matches kept by the TF-IDF half. The gain has to reach about 0.08 on 20 cards to show, so grow the sample with it. `all-MiniLM-L6-v2` first for speed, `all-mpnet-base-v2` as the quality check. Decided on 2026-10-30: semantic textual similarity with torch, through `sentence-transformers`; the step, its dependencies (torch's CPU wheel, which step 12 left open) and its done-when are roadmap step 23 (`app/docs/roadmap.md`).
- **The card is not too small for this.** A card is 12 words of question and 38 of answer (medians), the length sentence encoders are trained on, and a pretrained encoder needs no corpus. Small are the judged sample (20 cards) and the corpus for anything that learns on it: fine-tuning, LSA, a stable IDF. Do not extend the content to fix card neighbours.
- **Extend to the notebooks and slides for retrieval,** from a card to the notebook cell or slide that teaches it. There the lexical tools have real jobs: fuzzy matching of H2 headings to topic names and chunk titles, TF-IDF or BM25 as the lexical half of a hybrid retriever over chunks, ROUGE, BLEU and METEOR for generated answers against card answers. The hand-check method carries over unchanged: pooled candidates, the rule, hand p@5. One card-level variant to try then: the card plus its heading's slide text as the document, with the LSA caution that more shared context smooths toward deck level.
- **Keep the extra judgements:** `results/jaccard_extra_marks.csv` (moved there from this folder on 2026-10-30: the marks live in `results/`, tracked) holds the 30 Jaccard candidates (11 relevant) in the shape the step "Protect the marks" proposes; `hand_check.md` cannot hold them, since the rewrite keeps the variants' candidates only.

Library references: [gensim's WMD tutorial](https://nbviewer.org/github/piskvorky/gensim/blob/4.3.3/docs/notebooks/WMD_tutorial.ipynb), [MiniLM against mpnet](https://zilliz.com/ai-faq/what-are-some-popular-pretrained-sentence-transformer-models-and-how-do-they-differ-for-example-allminilml6v2-vs-allmpnetbasev2), [rapidfuzz](https://pypi.org/project/rapidfuzz).

## Improvement workflow

### Step 0
Rename in all sources similarity model -> similarity model (S1) in first occurrence in each document, then S1; M1->S1.

#### Where the similarity model is named M1

docs/draft/helper/concept.md
  - section heading "M1: TF-IDF similar cards (first milestone)"
  - metrics table: "M1 | TF-IDF cosine similarity in the full vector space | The text baseline"
    and "M1-map | TF-IDF positions on the map ... | How much survives the projection onto the map"
  - section "M1 numbers" (the variant table with precision@5, coverage, distinct, vocab)
  - "Map measures (M1-map) of the chosen model"

modules/helper/evaluate.py
  - docstring: "M1, the TF-IDF variants: same-deck precision@5 ..." and
    "M1-map, the chosen model's positions: trustworthiness and top-3 overlap ..."
  - the rows it writes: model column "M1" for every vectoriser variant,
    "M1-map" for the layout measures (visible in results/results.csv)

modules/helper/app/docs/roadmap.md
  - done on 2026-10-30: the hand-check section names S1 and S2 and says where M1 remains

docs/draft/helper/roadmap.md, section "Later"
  - "M2: sentence embeddings as a second vectoriser, measured against M1."

Not in the code: app/models/similar.py calls it the baseline vectoriser /
SimilarityIndex; the standalone page calls it the course model. M1 is the
evaluation's and the drafts' name only, M1-map the same vectors after the layout.

Found on 2026-10-30 in addition (a grep for `M1` outside `.venv`; the notebooks' hits mean other things):
  - docs/draft/helper/architecture.md, the "Related" line: "the measured numbers of M1, D1 and T1"
  - the `model` column of the rows already in results/results.csv (30 rows) and of its copy scripts/input/data.csv, the migration's input: rename the old rows too, the column is the key the drafts filter by
  - the "M2" of the roadmaps becomes S2 with it

### Write the judging rule down

The header `evaluate.py` writes and concept.md's hand-check bullet give the rule in one line (the same concept, a prerequisite, or the natural next card). Write the rule below into concept.md's bullet, with the examples, so the next judge decides the same way; the examples are card ids of the first judging.

- **The frame is the lecture's H2 sequence** (docs/draft/book/topics.md, appendix "H2 candidates per lecture"). The scripted topic names of decks 17 and 24 are those headings; they group the concepts, they are not the concept: "the same concept" is the card's own concept (tokenization, n-grams, IDF) in any stack, and two cards under one heading are not related by that alone (the n-gram card and the stemming card share "Statistical Encodings" and nothing else). "A prerequisite" is an earlier concept the card builds on, "the natural next card" the concept the deck or the lecture goes on to. The headings in scope: the NLP lecture (Working with text data, Converting text into numbers, Statistical Encodings, So.. what do we do with all that?, Word Embeddings, Hugging Face & Transformers, Transformers, Large Language Models), Prompt Engineering, RAG, RNN and transformer, and the adjacent cosine-distance heading of the KNN lecture (deck 11) and the evaluation-metrics lecture (deck 10).
- **Any deck, any stack:** a candidate of another deck counts when it is the same concept (the cosine cards 411016, 411017, 311042 of deck 11 and 418027 of deck 18 for the cosine cards of decks 17 and 24; the reinforcement-learning cards 303033 and 403005 of deck 03 for the RLHF stage card 424004; the metric cards of deck 10 for the AUC card 317041).
- **A prerequisite counts one way:** from the later card to the earlier one. The tokenization card 417001 is a prerequisite of the next-token stages card 324022, so it is marked under 324022; 324022 is not the next card after 417001 and stays unmarked there.
- **The lecture's follow-up counts:** stopwords (317033) to the TF-IDF cards (417007, 417008, 317043), because IDF is the lecture's other answer to uninformative words; the LLM definitions (424001, 324016) to the transformer, the training stages, hallucination and the LLM as an agent's brain (418005).
- **One concept across formulations:** nltk's word tokenization (417001) and the LLMs' subword tokenization (324017) are one concept.
- **Judge the concept, not the wording:** a candidate that shares words but not the concept is unmarked (the ice-cream overfitting card 408018 for n-grams; the stacking recipe 412028 for the training stages; the autoencoder card 422024 for distributed representations; the neural-network layer cards 115031 and 115032 for how embeddings are learned; the agent-tools card 418006 and the NLP-subfield card 303039 for the LLM definition; pooling 424011 for the next-token stages).
- **Judge the concept, not the dataset:** notebook 1's spam cards (317034, 317041) and notebook 2's zero-shot card (317037) share the SMS data and nothing else, unmarked both ways.
- **Zero-shot classification and few-shot prompting are different concepts** (417030 and 317037 against 324025), unmarked both ways.
- **Mark as many as apply,** without a quota per card (2 to 7 in the first judging); mark `none` only after reading every candidate's answer.

### Pool, judge, compare: adding a candidate model

1. **Add the candidate as a variant** of `evaluate.py` (`variants()`). Today a variant is a set of TF-IDF options; let it carry a vectoriser factory instead, since `similar.SimilarityIndex.fit` takes anything with `fit_transform` (a Pipeline of TF-IDF and TruncatedSVD, a sentence-embedding wrapper, a hybrid that sums two scores). Keep every existing variant: the chosen one is the reference every interval is paired against.
2. **Run the similarity part.** The candidate's neighbours that no judged variant listed appear in `hand_check.md` unmarked, with the candidate's column alone, so they are easy to find.
3. **Judge the new candidates first** by the rule above, then run again. Read `hand_precision` only after that: until then the new model is penalised for every card nobody judged, and the comparison is unfair in the baseline's favour.
4. **Compare with the paired difference,** not with the two means: the per-card difference to the chosen variant, bootstrapped over the cards (the scratch script's 5,000 draws). Put that into `app/models/evaluation.py` next to `same_deck_precision`, and a column for the interval into `results.csv`, so the comparison comes with every run.
5. **Judge the ranks too:** hand p@1 and p@3 beside p@5. TF-IDF's first neighbour is right 85 % of the time (character n-grams 95 %); a candidate that lifts p@5 by losing the first neighbour is a worse recommender.

### Protect the marks

`write_hand_check` writes a mark only for a candidate some current variant still lists, and only for the cards `HAND_CHECK_DECKS` samples. Removing a variant or changing the sample loses judgements. Until the marks live in a file of their own (`results/hand_marks.csv` in the shape of `results/jaccard_extra_marks.csv`, whose rows merge into it: card, candidate, relevant, judged, judge, note; read before the rewrite, written after it; roadmap step 17), never remove a variant, and grow the sample only to every card of a deck (step 1), which keeps every sampled card. Back the file up before a run that changes the sample.

### Refit the minimum score per vectoriser

The minimum score 0.1 is TF-IDF's: about the 10th percentile of the fifth neighbour's cosine (0.110, median 0.167; `evaluate.py` prints the percentiles of the chosen model). Dense vectors sit higher: the LSA variants gave the same numbers with and without the minimum at 0.1, so nothing was cut. Before comparing kept precision, coverage and distinct cards, set every vectoriser's minimum at the 10th percentile of its own fifth neighbour (`SimilarityIndex.fit(..., min_score=...)`), and print the percentiles for every variant, not only the chosen one. For a hybrid, normalise each score the same way before summing.

### Two measures side by side

Use the same-deck precision@5 for sweeps over settings: it is free, over 1,064 cards, and orders the TF-IDF variants like the hand check (Spearman 0.93). Let the hand check decide, and treat a disagreement as a finding, not a tie: LSA's same-deck 0.712 against hand 0.650 says the same-deck measure rewards smoothing toward deck-level topics, which the learner does not want. The same-deck measure is blind to the 38 % cross-deck relevant pairs by construction; a model that finds more of them loses on it.

### Grow the sample

20 cards resolve differences of about 0.08 and nothing smaller, so the open settings (bigrams, min_df=3) and any candidate within 0.05 of the chosen variant wait for a larger sample. Every second card of decks 17 and 24 gives 36 cards (the interval shrinks to about 0.75 of today's), all 70 cards of the two decks halve it (0.53). Set `HAND_CHECK_DECKS` to step 1 rather than 2: step 2 drops four of deck 24's judged cards (its sample was every third card), step 1 keeps them all. A third deck tests transfer: deck 18 (AI Agents & RAG Systems, 42 cards) already appears in the pool with its cosine, agent and prompt cards (418027, 418005, 418006, 318038).

### Tag the errors in the H2 scope

Tag every miss of the chosen variant as lexical trap, shared dataset or generic term, and every hit as same heading, prerequisite or next heading (the scripted topic id of `cards.csv` is the heading). Then judge a candidate model on the tags, not on the mean alone: S2 should fix the lexical traps (paraphrases, the same concept in other words) without losing the exact-term matches where TF-IDF is strong (code names such as `word_tokenize`, `doesnt_match`, `CountVectorizer`). A hybrid of the TF-IDF score and an embedding score is the natural candidate; the heading structure is a second one: a bonus for candidates in the same or the following heading, as a re-ranking prior over any vectoriser, measured against the same marks.

### After S1: D1 and the map

D1's course features (centrality, first and fifth neighbour) are S1's, so refit and re-measure D1 (`evaluate.py --part difficulty`) after any change of the vectoriser, and compare its feature sets `text` and `text+course` on the hand-labelled difficulty sample of roadmap step 11 once it exists; the out-of-fold numbers alone leave that choice open. The map measures (M1-map: trustworthiness, neighbour overlap) are of the chosen vectors too and are recorded by the same run.
