# T1: the two emptied topics

On the course (`data/topics.json`, 24 decks, 1,064 cards, 133 scripted topics) the topic model T1 empties two topics: every scripted card of the topic is nearer to another topic of its deck. An emptied topic keeps its dot, without cards. Both topics had three scripted cards, so each card's own topic was judged by a centroid of the other two cards only, the weakest evidence the leave-one-out rule gives.

**How the numbers are read:**
- **Score:** the cosine of the card's unit TF-IDF row to a topic's centroid in its deck (the normalised mean of the topic's unit rows); for the card's own scripted topic, the centroid without the card.
- **Margin:** the best score minus the second best; below `FOCUS_MARGIN` (0.05) a card sits between two topics.
- **Own rank:** the place of the card's scripted topic among its deck's topics.

## Deck 06 "Exploratory Data Analysis & Visualization", topic 01 "The EDA Process; The EDA Process - Recap"

The deck's topics, scripted cards → the model's: 01 The EDA Process; The EDA Process - Recap (3 → 0), 02 Dos and Don'ts of EDA; Where does EDA belong in the bigger picture … (6 → 3), 03 Multivariate EDA; EDA and why do we do it … (6 → 6), 04 Univariate EDA (16 → 18), 05 Presenting your results (4 → 3), 06 Data Cleaning (16 → 21).

| Card | Stack | Question | Model's topic | Margin | Own rank | Scores |
|---|---|---|---|---|---|---|
| 406016 | 4 | What is a Contingency Table? | 04 Univariate EDA | 0.017 | 2nd | 04 0.085, 01 0.068, 02 0.067, 06 0.057, 03 0.034, 05 0.030 |
| 106033 | 1 | Why are we doing a train/test/validation split? How are we typically doing it? | 06 Data Cleaning | 0.034 | 3rd | 06 0.075, 04 0.040, 01 0.037, 02 0.035, 05 0.006, 03 0.005 |
| 306037 | 3, easy | EDA is often described as "detective work." What is the primary goal of **Exploratory Data Analysis (EDA)**? | 06 Data Cleaning | 0.008 | 3rd | 06 0.103, 03 0.096, 01 0.039, 02 0.037, 04 0.034, 05 0.007 |

## Deck 16 "Image Modeling & Convolutional Neural Networks (CNNs)", topic 02 "Linear models for image modeling"

The deck's topics, scripted cards → the model's: 01 Deep Neural Networks for image modeling; Motivation (8 → 9), 02 Linear models for image modeling (3 → 0), 03 Convolutional layers (22 → 26), 04 Data augmentation and transfer learning (8 → 4), 05 Pretrained Networks (6 → 8).

| Card | Stack | Question | Model's topic | Margin | Own rank | Scores |
|---|---|---|---|---|---|---|
| 416014 | 4 | What is ReLU, and why does it matter for CNNs? | 03 Convolutional layers | 0.031 | 4th of 5 | 03 0.130, 04 0.099, 01 0.085, 02 0.048, 05 0.042 |
| 316038 | 3, easy | On the MNIST Fashion dataset, logistic regression reaches only about 86% accuracy. Why do linear models struggle to do better on image data? | 01 Deep Neural Networks for image modeling; Motivation | 0.055 | 4th of 5 | 01 0.139, 03 0.084, 04 0.056, 02 0.050, 05 0.045 |
| 316040 | 3, medium | The model is compiled with `loss='sparse_categorical_crossentropy'`. According to the notebook, why is the sparse version chosen over regular categorical crossentropy? | 01 Deep Neural Networks for image modeling; Motivation | 0.000 | last | 01 0.047, 04 0.047, 03 0.007, 05 0.006, 02 0.002 |

## Findings

- **All six cards leave, and five of them sit between two topics:** every margin is below 0.05 but 316038's (0.055).
- **The scores are low:** the winning topic scores 0.047 to 0.139, against a median best cosine of 0.272 on the whole course.
- **The scripted topic is close in deck 06 and far in deck 16:** in deck 06 it ranks 2nd, 3rd and 3rd; in deck 16 it ranks 4th, 4th and last.
- **One tie:** 316040 scores 0.047 for topics 01 and 04; topic 01 wins only because it comes first in the course file.

## Reading of the new topics

A judgement by content, still to be checked in the browser:
- **Deck 16 reads plausibly:** ReLU under "Convolutional layers", and the two quizzes under the motivation for deep networks.
- **Deck 06 reads poorly:**
  - A contingency table relates two variables, so "Univariate EDA" is the wrong home; "Multivariate EDA" scores only 0.034 for it.
  - Neither the train/test/validation split (106033) nor the goal of EDA (306037) belongs under "Data Cleaning".
  - For 306037 the runner-up, 03 "Multivariate EDA; EDA and why do we do it …" (0.096 against 0.103), would read better.
