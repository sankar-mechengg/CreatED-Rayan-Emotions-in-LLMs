# Published results

These files are the outputs that the Revision 5 tables and figures were drawn from. `code/report_manuscript_statistics.py` prints the principal values. The root `README.md` lists every file.

Re-running `analyse_experiments.py`, `compare_baselines.py`, `baseline_analysis.py`, or `validate_lmm.py` overwrites the matching file in this directory. Copy the directory first if you need the released bytes.

`experiments/F1.jsonl` through `F4.jsonl` are the trial-level records, including prompts, label log-probabilities, and generated text. `experiments/measures_F1.json` through `measures_F4.json` are the Deviation 1 scores for those trials.

The per-trial raw activation arrays (`raw_F1_*.npz` and the same pattern for F2, F3, and F4) are not included. `recompute_measures.py` expects them under `experiments/` after a fresh run of `run_experiments.py`.
