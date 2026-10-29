"""The models (no Dash, no state, nothing of the viewer): numpy, scipy and scikit-learn only.

* `similar`: card text, a swappable vectoriser, unit rows, neighbours, deck vectors (SimilarityIndex).
* `layout`: the layout step, an SVD start refined by a force layout on the sphere, with its measures.
* `clustering`: one connection per merge of scipy's hierarchical clustering, in merge order; the cut-off preset's
  rank over those pairs (preset_rank).
* `topic_model`: T1, nearest centroid over the scripted topics, leave-one-out (TopicSeed, TopicModel).
* `course_model`: the fitted model of a course: deck points and connections, T1, one layout per topic.
* `difficulty_model`: D1, the difficulty of the unrated quizzes, measured out of fold (DifficultyReport).
* `evaluation`: the offline measures and the judges of T1, for modules/helper/evaluate.py.

The input is a sequence of cards (app.cards.Card), the deck ids by deck name and the scripted topics
(TopicSeed); the glue reads them from the course file with the viewer's parser, the capstone from the data
package's table.
"""
