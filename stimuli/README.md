# Experiment stimuli

`f1_items.csv` lists the 480 Family 1 items: 80 held-out English texts for each member of anger-disgust, fear-sadness, and happiness-surprise. The `row` column is the row index in the deduplicated corpus. This file is covered by the preregistration hash.

`f2_disambiguation.py` holds the 120 Family 2 scenarios (40 for each pair), the neutral filler sentences, and the four instruction paraphrases. Evidence sentences were written so they do not contain the emotion label or a direct morphological variant.

`f3_f4_generation.py` holds the 40 neutral facts for Family 3, the 40 anger situations and 40 fear situations for Family 4, and the instruction paraphrases for express, suppress, and neutral.

Families 2 to 4 were written by an AI system under the authors' direction. That is stated in the manuscript and in the preregistration.

`code/derive_map.py` rewrites `f1_items.csv`. Keep a copy of this file before running it again.
