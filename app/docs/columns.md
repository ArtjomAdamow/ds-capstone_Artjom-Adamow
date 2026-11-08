# The columns of the results files

What the measure columns of `results/` mean: `results.csv` (B0, S1 and S2, which share the similarity columns, C1), `topics.csv` (T1), `difficulty.csv` (D1 and D2) and `prompt.csv` (P1), written by `evaluate.py`. The standalone page reads this file: a click on a row of a model's results table opens the row's description under it, what the measure is and its range (`app/standalone/results.py`). The bookkeeping columns (`date`, `run`, `model`, `variant`, `scope`, `params`, `features`, `label_rule`, `folds`, `repeats`, `seed`, `seconds`, `notes`) have no row here: they name the run and the row. In `difficulty.csv` a row's `scope` is the course (`all`; empty on rows written before the column) or a deck (`deck 01`): a deck's row carries the within-deck order and its paired difference over that deck's quizzes alone, a readout the standalone page shows for the entered deck; `params` names D2's model and the device that tagged.

Columns of one measure share a row: a value with its sd or its interval, k = 2 with k = 3.

## results.csv

| Column | What it measures | Range |
|---|---|---|
| precision_at_5 | Of each card's five neighbour slots, the share whose card is of the card's own deck, averaged over all cards; an empty slot (no fifth neighbour above the minimum score) counts as a miss. A cheap check that the vectors capture topics; some cross-deck neighbours are good ones, so 1 is not the goal. | 0 to 1, higher is better. B0's row is the floor: the same measure for the five nearest cards at random positions on the sphere, the mean over its seeds. |
| kept_precision | Of the neighbour slots the variant fills, at most five per card and only above its minimum score, the share whose card is of the same deck as the card itself. Unlike precision_at_5, an empty slot is not a miss, so a variant that keeps fewer, surer neighbours scores higher here and lower there. | 0 to 1, higher is better. |
| coverage | The share of cards with at least one neighbour above the variant's minimum score. | 0 to 1, higher is better. |
| distinct | How many different cards appear as some card's neighbour at all; a low count means hub cards that show up everywhere. | 0 to the number of cards, higher is better. |
| min_score | The variant's own minimum score: the 10th percentile of the cosine of every card's fifth neighbour. A neighbour below it is dropped. | 0 to 1; a setting of the variant, not a score. |
| hand_precision, hand_p1, hand_p3 | Hand-judged precision@5, @1 and @3: over the judged cards of the hand check (every card of decks 17, 24 and 18), the mean share of the variant's top 5, 1 or 3 neighbours that were marked relevant by the judging rule; a candidate not judged counts as not relevant. | 0 to 1, higher is better. |
| hand_diff, hand_diff_low, hand_diff_high | The variant's hand precision@5 minus the chosen variant's, paired over the judged cards, with its 95 % bootstrap interval (5,000 draws over the cards); the chosen variant's own row reads 0. An interval that excludes 0 is a difference the sample resolves. | −1 to 1; 0 is the chosen variant. |
| judged_cards | How many hand-judged cards the hand columns are computed over. | A count. |
| trust_decks | scikit-learn's trustworthiness of the deck layout for the deck vectors at k = 3: 1 when none of a deck's three nearest decks on the map is a stranger by text, lower the more strangers intrude. | 0 to 1, higher is better; the random row is the floor. |
| overlap_decks | The share of a deck's three most similar decks by text that are also its three nearest on the map, averaged over the decks. | 0 to 1, higher is better; the random row is the floor. |
| overlap_cards | The within-topic overlap at k = min(5, n − 1), averaged over the model's topics with two cards or more: the share of a card's k most similar cards of its topic that are also its k nearest on the topic's sphere. With k most of a small topic it is high for every layout, random included. | 0 to 1, higher is better. |
| overlap_k2, overlap_k3 | The within-topic overlap at the fixed k = 2 or 3, over the topics with k + 2 cards or more (with k + 1 cards every other card is a neighbour, and the overlap is 1 for any layout). | 0 to 1, higher is better; the random row is the floor. |
| lift_k2, lift_k3 | How far the layout's within-topic overlap at k = 2 or 3 lies above random positions on the way to 1: (overlap − random) / (1 − random). 0 is the random floor, 1 a layout that keeps every card's two or three most similar cards as its nearest on the map. Measured over the topics with k + 2 cards or more, where the overlap is not 1 by construction. | 0 to 1, higher is better; below 0 the layout is worse than random. |
| trust_cards | The mean trustworthiness of the topic layouts at the layout's k, over the topics where scikit-learn defines it (more than 2k cards: 11 or more). | 0 to 1, higher is better; the random row is the floor. |
| overlap_17, overlap_24 | overlap_cards over the topics of deck 17 and of deck 24 alone, the hand-check decks. | 0 to 1, higher is better. |

## topics.csv

| Column | What it measures | Range |
|---|---|---|
| cards | How many cards the row covers: the course, a stack, a deck, or the cards kept at the row's margin. | A count. |
| topics | How many scripted topics the row's scope has, the empty ones too. | A count. |
| agreement | The share of the scope's cards whose model topic is their scripted topic (one minus the Hamming loss). In the rows 1-NN, LR and chance the same share for the neighbour rule (the nearest other card of the deck), a logistic regression per deck (leave-one-out) and an assignment drawn by the topic sizes. The script is the model's input, not the truth, so 1 is not the goal. | 0 to 1, higher means closer to the script; the chance row is the floor. |
| kept_share | The share of the course's cards with a margin of at least the row's threshold (the margin rows and the slider's marks); 1 for the whole course. | 0 to 1. |
| margin | The row's margin threshold: the margin rows' own (`margin >= 0.05`), and for a slider mark (`focus 50 %`, the share of the course's cards in focus) the margin the share stands for, the highest margins first: 0 at 100 %, the eleventh highest at 1 %. The viewer's margin slider filters the cards view at it. | 0 to 1; a setting, not a score. |
| coherence_median | The median over the scripted topics of a topic's coherence: the share of its scripted cards whose nearest centroid is the topic itself. | 0 to 1, higher is better. |
| coherence_below_half | How many scripted topics keep fewer than half of their scripted cards. | A count, lower is better. |
| coherence_at_one | How many scripted topics keep every one of their scripted cards. | A count, higher is better. |
| moved | How many cards the model places in another topic than the script. | A count. |
| emptied | How many scripted topics the model leaves without cards (they keep their dots). | A count. |

## difficulty.csv

| Column | What it measures | Range |
|---|---|---|
| macro_f1, macro_f1_sd | The mean of the three levels' F1 scores on the rated quizzes, out of fold, and its standard deviation over the 3 repeats of the 5-fold cross-validation. Under the deck shares every level keeps its true count, so it equals the balanced accuracy. | 0 to 1, higher is better; the always-easy row is the floor. |
| balanced_accuracy, balanced_accuracy_sd | The mean over the three levels of the share of the level's quizzes predicted at that level, with its sd over the repeats. | 0 to 1, higher is better. |
| kappa, kappa_sd | Cohen's kappa with quadratic weights: the agreement of the predicted with the authored levels beyond chance, an error of one level counted less than one of two; with its sd over the repeats. | −1 to 1; 0 is chance, higher is better. |
| accuracy, accuracy_sd | The share of the rated quizzes whose predicted level is the authored one, with its sd over the repeats. Always easy scores the easy share, 8 of 12. | 0 to 1, higher is better. |
| spearman | Spearman's rank correlation of the out-of-fold score (P(medium) + 2 · P(hard)) with the authored level. | −1 to 1, higher is better. |
| recall_easy, recall_medium, recall_hard | The share of each authored level's quizzes predicted at that level: the diagonal of the confusion matrix. In the hand row: the share of each authored level the hand labels called the same. | 0 to 1, higher is better. |
| rated, unrated | The rated quizzes the model is measured on, and the unrated quizzes it predicts. | Counts. |
| predicted_easy, predicted_medium, predicted_hard | The levels the row's label rule gives the unrated quizzes. | Counts; they sum to unrated. |
| hand_judged | How many of the row's predicted quizzes carry a hand label: the sample the hand columns are computed over. | A count. |
| hand_macro_f1, hand_kappa, hand_accuracy | Macro-F1, weighted kappa and accuracy of the row's predictions with the hand labels as the truth, over the hand-labelled unrated quizzes. In the hand row: the hand labels against the authored levels of the rated quizzes judged blind, the criterion against the quota. | As above. |
| hand_spearman | Spearman's rank correlation of the score with the hand level over the judged quizzes. | −1 to 1, higher is better. |
| hand_within_deck, hand_within_pairs | Of the pairs of a deck's judged quizzes with different hand levels, the share whose scores are in the hand levels' order (a tie counts half), and how many such pairs there are, over every deck on a course row and over one deck on a deck row (`scope` deck). The order the recommender would use. | 0 to 1; 0.5 is chance, higher is better. |
| hand_diff, hand_diff_low, hand_diff_high | The row's within-deck order minus D1's chosen row's (`text`, deck shares), paired over the judged quizzes, with its 95 % bootstrap interval (5,000 draws over the quizzes, the pairs counted anew in each draw); the chosen row's own row reads 0; on a deck row, paired over that deck's quizzes alone. An interval that excludes 0 is a difference the sample resolves: the measure that decides whether a candidate feature set takes over D1's, and whether D2 takes over the app's predicted quizzes. | −1 to 1; 0 is the chosen row. |

## prompt.csv

| Column | What it measures | Range |
|---|---|---|
| self_topic_at1, self_topic_at3 | Over the self set (every card's question alone, its `Source:` line cut, the card held out), the share whose own topic, T1's on the app's vectoriser, ranks first or among the first three by the variant's mean. The cards' own wording, not a learner's: the sweep over every variant, not the verdict. | 0 to 1, higher is better. |
| self_navigated | The share of the self set the rule navigates for, a topic entered or a card selected. The card route's floor is the 10th percentile of the same questions, so it clears about 90 % by construction. | 0 to 1; a readout, not a goal. |
| min_card, min_centroid, min_text | The run's floors, fixed from the self set before any request: the card route's floor the 10th percentile of a question's first neighbour's cosine among the other cards; the centroid route's floor the 10th percentile of a question's cosine to its own topic's leave-one-out centroid; the text route's of its cosine to its own topic's headings. A score clears one only above it; there is no second threshold for the card, the nearest card goes with the entered topic. | −1 to 1; settings of the run, not scores. |
| hand_right | Over the judged requests of decks 17 and 24 (`prompt_requests.csv`; the transfer deck's apart), the share the rule serves right: a navigation into an acceptable topic (the intended card's under T1's membership, the named topic, or an `also` target's), or "no match" on an off-course request. The headline. | 0 to 1, higher is better. |
| hand_wrong | The share navigated to a topic that is not acceptable, or navigated at all on an off-course request: the learner lands somewhere else, the costly error. | 0 to 1, lower is better. |
| hand_missed | The share that got "no match" though the course teaches the request's concept. With hand_right and hand_wrong it sums to 1. | 0 to 1, lower is better. |
| hand_course_right, hand_course_wrong, hand_course_missed, hand_off_right, hand_off_wrong | The same outcomes as counts, apart on and off the course: the in-course requests taken into an acceptable topic (right), into another topic (wrong) or told no match (missed); the off-course requests told no match (right) or taken into any topic (wrong). Together they are judged_requests; the dashboard's P panel draws them as a grid. | Counts. |
| hand_topic_at1, hand_topic_at3, hand_card_at1 | The ranking alone, without the rule: over the in-course requests, an acceptable topic first or among the first three; over the requests with an intended card, the first topic's best card the intended card or an `also` card. | 0 to 1, higher is better. |
| hand_diff, hand_diff_low, hand_diff_high | The variant's hand_right minus its vectoriser's card route's (the character n-grams' minus S1's), paired over the judged requests, with its 95 % bootstrap interval (5,000 draws over the requests); a card route's own row reads 0. The choice rule: the best hand_right is chosen only when its interval excludes 0, else the card route stays. | −1 to 1; 0 is the card route. |
| judged_requests | How many requests the hand columns are over: those of decks 17 and 24 with a target, and the off-course ones. | A count. |
| ms_per_request | The median time of one request through the variant (one transform and the decision), over the requests, or over 55 questions of the self set without them: the dialog's latency. | Milliseconds, lower is better. |
