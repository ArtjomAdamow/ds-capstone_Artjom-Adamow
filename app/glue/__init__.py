"""The glue: the models on the viewer (the flashcards package, import name `project`), over its public API only.

* `composition`: `compose_model`, the viewer's `compose` on the derived course file with the Helper button and the
  card position source; `serving_process`.
* `button`: `Helper`, the Helper button per view with its dialog's checked models (a deck's model topics; the decks'
  and the cards' positions; the model's own connections with their presets; the difficulty groups).
* `dialog`: the Helper dialog's catalogue: the models per view, their labels, the notes, the run order.
* `course`: the course file as the models' input, read with the viewer's parser (the data package's table later).
* `derived`: the derived course file the viewer reads (the model's topics where given) and the model state.
* `positions`: the viewer's position sources over the model's points, with the viewer's random fallback.
* `cutoff`: the cut-off presets over the model's pairs (models.clustering.preset_rank).
* `difficulty_groups`: the difficulty and predicted groups as generated groups of the card state.
* `related`: the card section under the card panel: the selected card's neighbours with where they lie, the next
  card first by the learner's progress (the viewer's `card_section` hook).
* `focus`: the margin slider of the cards view: T1's margins on the share scale (the viewer's `focus` hook).
* `prompt`: the Ask dialog of the top bar: P1's answer to a request as the viewer's targets, the nearest three topics
  of the checked decks with the card where one is selected (the viewer's `prompt` hook).

What the glue needed of the viewer beyond its hooks (the navigation rules, the groups apart, the topic reload) is
the viewer's own: `compose`'s refill_groups and after_save, the groups'
kind and origin, and `TopicIndex.reload`.
"""
