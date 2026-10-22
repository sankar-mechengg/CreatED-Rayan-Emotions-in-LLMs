# Activation cache

This directory is empty in the release. `code/extract_baseline.py` writes:

| File | Contents |
| --- | --- |
| `gate_zh.npy` | Chinese pooled gate activations, float16, shape `(15895, 9, 8960)` |
| `gate_en.npy` | English pooled gate activations, same shape |
| `labels.npy` | Primary emotion of each row, shared by both languages |
| `meta_zh.json`, `meta_en.json` | Token counts and runtime |

Each gate file is about 2.6 GB. The manuscript archive omitted them for that reason. They can be regenerated with `extract_baseline.py`, or requested from the corresponding author as stated in the manuscript.

The reference map is stored in `reference_map/`, not here.
