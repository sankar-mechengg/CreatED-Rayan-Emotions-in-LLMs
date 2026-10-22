# Corpus construction

These scripts built `data/emotion_dataset_translated.csv`. The manuscript analyses start from that file. `code/pipeline/preprocessing.py` then drops duplicate Chinese text and primary label, empty text, and labels outside Ekman's six emotions, leaving 15,895 rows.

| Script | What it does |
| --- | --- |
| `emotion_classifier.py` | Assigns a primary Ekman label and secondary labels with GLM-4.5 |
| `filter_empty_emotions.py` | Removes rows whose emotion field is empty |
| `translate_chinese.py` | Writes the English translation column with GLM-4.5 |
| `check_translations.py` | Checks the translation column |
| `emotion_summary.py` | Prints class counts |

`emotion_classifier.py` and `translate_chinese.py` read `GLM_API_KEY` from the environment and stop immediately if it is missing. No key is stored in this repository. The intermediate CSVs these scripts expect are not included; the translated corpus used for every reported statistic is `data/emotion_dataset_translated.csv`.
