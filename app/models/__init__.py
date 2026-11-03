"""The models (no Dash, no state, nothing of the viewer): numpy, scipy and scikit-learn only.

* `similar`: card text, a swappable vectoriser, unit rows, neighbours, deck vectors (SimilarityIndex).
* `layout`: the layout step, an SVD start refined by a force layout on the sphere, with its measures.
* `clustering`: one connection per merge of scipy's hierarchical clustering, in merge order; the cut-off preset's
  rank over those pairs (preset_rank).
* `topic_model`: T1, nearest centroid over the scripted topics, leave-one-out (TopicSeed, TopicModel).
* `course_model`: the fitted model of a course: deck points and connections, T1, one layout per topic.
* `difficulty_model`: D1, the difficulty of the unrated quizzes, measured out of fold (DifficultyReport); D2 alone
  judged the same way (judge_tags).
* `zero_shot`: D2, the zero-shot difficulty tags of the quizzes by an NLI model (transformers, imported when the
  model is loaded), cached like the embeddings (ZeroShotTagger, tags_of).
* `evaluation`: the offline measures (the hand precision and its paired bootstrap, the random draw and the random
  floor, the fixed-k overlap of the layouts) and the judges of T1, for evaluate.py next to the package.

The input is a sequence of cards (app.cards.Card), the deck ids by deck name and the scripted topics
(TopicSeed); the glue reads them from the course file with the viewer's parser, the capstone from the data
package's table.
"""
