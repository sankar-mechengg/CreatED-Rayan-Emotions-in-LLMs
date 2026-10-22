# Preregistration

`PREREGISTRATION.md` is the document frozen at 2026-10-05T20:36:36Z, before any experimental activation was extracted.

`DEVIATION_1.md` was frozen at 2026-10-05T22:31:21Z. It replaces the signed-z internal measure, which failed a held-out validity check, with the mask fraction as the primary corrected measure and the absolute-z mean as the robustness measure. Hypotheses, directions, items, models, and the Holm family were left unchanged. The corrected analyses are post hoc, because the preregistered-measure results of Families 1 to 3 had already been seen.

`HASHES.txt` and `HASHES_DEVIATION_1.txt` are the SHA-256 records.

`hashed_files_as_frozen/` holds byte-identical copies of the files named in those records, laid out at the relative paths written in the hash files. The runnable scripts in `code/` differ from those copies because their paths were pointed at this repository.

Check the record from the repository root:

```bash
python code/verify_preregistration_hashes.py
```

Thirteen files match. `analyse_experiments.py` was revised after the freeze. The recorded hash is `648d1b3e78b5f456af4c3ae9d6b23305ac0d1f069f718f6b346d67d7f6dfbf16`. The copy shipped in the supplementary archive, and stored here, hashes to `784b5ad06b0e75db817d57eb3c5ac375a56664ebd8531e862c629181bf82553e`. That later file is the one that analyses the preregistered measure and both Deviation 1 measures. The pre-freeze source was not in the project files.
