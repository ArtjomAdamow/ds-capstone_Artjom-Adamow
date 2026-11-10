"""The guided learning app's package: the course table and the dashboard over the models (`models`, next to it).

* `figures`: the models' plotly figures (D1's metrics, the sphere's surface and arcs, the connections, the map's
  colours and sizes, T1's margins, the dendrogram over scipy's coordinates); plotly only, no Dash, and out of
  `models`, so the model tests need no plotly.
* `headlines`, `results`: the results files of the offline evaluation and each generation's two numbers with their
  verdicts, read by the dashboard; nothing of Dash.
* `panels`: the model panels' names and letters, the cut labels, the figure slot's helpers and the app's variant
  labels.
* `course_table`: the course table (cards.csv, decks.csv, topics.csv), the models' input, read and written with the
  csv module.
* `neighbours`: the related cards' order and labels.
* `dashboard`: the dashboard, one card at a time with what a learner sees of the models.
* `launch`: the launcher's part, the standard library only: the port argument and the port check, the data folder
  and the backup question.

The launcher start.py runs the dashboard; evaluate.py, next to it, measures the models.
"""
