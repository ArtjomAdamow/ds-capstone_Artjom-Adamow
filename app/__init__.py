"""The helper's own package: what becomes the capstone's main app.

* `cards`: the card record the models read (the seven fields of the viewer's Flashcard, as a protocol).
* `models`: the similarity pipeline, the layout step, the clustering, the topic model T1, the course model and the
  difficulty model D1, with the offline measures. They import nothing of the viewer (the flashcards package, import
  name `project`), so they run and are tested without it.
* `figures`: the plotly figures the dashboard and the models view draw (D1's
  metrics, the sphere's surface and arcs, the connections, the map's colours and sizes, T1's margins, the dendrogram
  over scipy's coordinates); plotly only, no Dash, nothing of the viewer, and out of `models`, so the model tests
  need no plotly.
* `headlines`, `results`: the results files of the offline evaluation and each generation's two numbers with their
  verdicts, read by the dashboard and the models view; nothing of Dash.
* `panels`: the model panels' names and letters, the cut labels, the figure slot's helpers and the app's variant
  labels, shared by the dashboard and the models view.
* `course_table`: the course table (cards.csv, decks.csv, topics.csv), the models' input without the viewer's
  parser, read with the csv module; the flashcards workspace's export writes it with the parser.
* `launch`: the launcher's part, the standard library only: the port argument and the port check, the data folder
  and the backup question, start.py's and the workspace launchers'.

The models on the viewer (the model state for the whole course with the Helper switch, the position sources, the
cut-off presets, the difficulty groups, the related cards under the card panel, the derived course file and the model
state's file) are the viewer's own plugin, over its public API and hooks: they import the models, the card record,
the neighbours and the course table of this package, and nothing of this package imports the viewer. The launcher
start.py runs the dashboard.
"""
