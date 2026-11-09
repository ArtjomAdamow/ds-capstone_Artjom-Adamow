"""The helper's own package: what becomes the capstone's main app.

* `cards`: the card record the models read (the seven fields of the viewer's Flashcard, as a protocol).
* `models`: the similarity pipeline, the layout step, the clustering, the topic model T1, the course model and the
  difficulty model D1, with the offline measures. They import nothing of the viewer (the flashcards package, import
  name `project`), so they run and are tested without it.
* `glue`: the model state for the whole course with the Helper switch, the position sources, the cut-off presets,
  the difficulty groups, the related cards under the card panel, the derived course file and the model state's file:
  the models on the viewer, over its public API only.
* `figures`: the plotly figures the glue's back of the card panel, the dashboard and the standalone page draw (D1's
  metrics, the sphere's surface and arcs, the connections, the map's colours and sizes, T1's margins, the dendrogram
  over scipy's coordinates); plotly only, no Dash, nothing of the viewer, and out of `models`, so the model tests
  need no plotly.
* `headlines`, `results`: the results files of the offline evaluation and each generation's two numbers with their
  verdicts, read by the dashboard and the standalone page; nothing of Dash.
* `panels`: the model panels' names and letters, the cut labels, the figure slot's helpers and the app's variant
  labels, shared by the dashboard and the standalone page.
* `course_table`: the course table (cards.csv, decks.csv, topics.csv), the models' input without the viewer's
  parser, read with the csv module; the flashcards workspace's export writes it with the parser.
* `launch`: the launcher's mode question, full (the viewer with the models on from the start) or standalone, the
  standalone page as a process next to the viewer, the data folder and the backup question, shared by the launchers.
* `standalone`: the models on a page of their own, without the viewer: the course table in,
  the three views, the dendrogram and D1's metrics in the viewer's theme; nothing stored.

The viewer part of the helper (the navigation rules, the groups apart, the topic reload) was never in here: it was
kept in the flashcards workspace as the project features it is and is merged into the viewer
(sputnik v0.2), where the glue uses it through the viewer's hooks. The launcher start.py runs the package on the
viewer and the standalone page.
"""
