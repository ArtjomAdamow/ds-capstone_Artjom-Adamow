"""The helper's own package (app/docs/roadmap.md, step 21): what becomes the capstone's main app.

* `cards`: the card record the models read (the seven fields of the viewer's Flashcard, as a protocol).
* `models`: the similarity pipeline, the layout step, the clustering, the topic model T1, the course model and the
  difficulty model D1, with the offline measures. They import nothing of the viewer (the flashcards package, import
  name `project`), so they run and are tested without it.
* `glue`: the Helper button per view, the position sources, the cut-off presets, the difficulty groups, the back of
  the card panel, the derived course file and the helper's own state: the models on the viewer, over its public
  API only.
* `figures`: the plotly figures the glue's back of the card panel and the standalone page both draw (D1's metrics,
  the sphere's surface and arcs, the dendrogram over scipy's coordinates); plotly only, no Dash, nothing of the
  viewer, and out of `models`, so the model tests need no plotly.
* `course_table`: the course table (cards.csv, decks.csv, topics.csv), the models' input without the viewer's
  parser, read with the csv module; modules/helper/export_course.py writes it with the parser.
* `launch`: the launcher's mode question, full (the viewer with the Helper button) or standalone.
* `standalone`: the models on a page of their own, without the viewer (step 21's mode (a)): the course table in,
  the three views, the dendrogram and D1's metrics in the viewer's theme; nothing stored.

The viewer part of the helper (the navigation rules, the groups apart, the topic reload) was never in here: it was
kept in modules/helper/delta/ as the project features it is and is merged into the viewer since 2026-10-28
(roadmap step 21.2.1; sputnik v0.2), where the glue uses it through the viewer's hooks. The launcher
(modules/helper/helper.py) runs the package on the viewer, or the standalone page (step 22).
"""
