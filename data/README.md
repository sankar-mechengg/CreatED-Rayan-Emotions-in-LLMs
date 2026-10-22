# Analysis data

`emotion_dataset_translated.csv` is the bilingual corpus. Columns are `chinese_text`, `primary_emotion`, `secondary_emotions`, and `english_text`. Texts are wrapped in square brackets by the translation pipeline; `extract_baseline.py` strips those brackets before tokenization. The file has 16,667 data rows. The loader in `code/pipeline/preprocessing.py` removes 772 duplicate Chinese-text and label pairs and leaves 15,895 samples, which is the N in the released baseline results. The class counts match Section 3.1. The manuscript text says 682 duplicates; that count is not what this file produces.

`overlap_matrix_earlier_version.csv` is the uncorrected Jaccard matrix from the earlier pipeline. `code/compare_baselines.py` uses it as the "earlier version" column of Supplementary Table S1.

Gate activation arrays are extracted into `cache/` and are not part of this folder.
