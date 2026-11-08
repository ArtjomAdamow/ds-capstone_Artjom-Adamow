"""The glue: the models on the viewer (the flashcards package, import name `project`), over its public API only.

* `composition`: `compose_model`, the viewer's `compose` on the derived course file with the model state and the
  Helper's switch; `serving_process`.
* `button`: `Helper`, the models' results for the whole course as the viewer's model state (the decks' and the cards'
  positions; the model's own connections with every view's preset; the difficulty groups) and the switch's label.
* `course`: the course file as the models' input, read with the viewer's parser (the data package's table later).
* `derived`: the derived course file the viewer reads (the model's topics for every deck) and the model state's file.
* `positions`: the viewer's position sources over the model's points, with the viewer's random fallback.
* `cutoff`: the cut-off presets over the model's pairs (models.clustering.preset_rank).
* `difficulty_groups`: the difficulty and predicted groups as generated groups of the card state.
* `related`: the card section under the card panel: the selected card's neighbours with where they lie, the next
  card first by the learner's confidence (the viewer's `card_section` hook).
* `prompt`: the Ask dialog of the top bar: P1's answer to a request as the viewer's targets, the nearest three topics
  of the checked decks with the card where one is selected (the viewer's `prompt` hook).
* `episodes`: the episodes' log: every ended episode of the Ask dialog or the card section, graded by the viewer, as
  a row of episodes.csv next to the model state (the viewer's `after_episode` hook).

What the glue needed of the viewer beyond its hooks (the navigation rules, the groups apart, the switches) is the
viewer's own: the groups' kind and origin, `CardLink.origin`, the module's switch and Display's.
"""
