# Analysis code

Run every command from the repository root, not from this directory. `repo_paths.py` finds `data/`, `results/`, `stimuli/`, `reference_map/`, `cache/`, and `figures/` from the script location.

The root `README.md` maps each file to the Revision 5 section it implements. Short index:

| Script | Revision 5 section |
| --- | --- |
| `extract_baseline.py` | 3.3 Gate extraction, Chinese (`zh`) or English (`en`) |
| `baseline_analysis.py` | 3.4-3.5 and Sections 4.1, 4.2, 4.4-4.6, S4 |
| `baseline_analysis_lib.py` | Signature rule: BH-FDR q < 0.05, Cohen's h ≥ 0.2 |
| `baseline_extras.py` | S3 five annotation measures, Section 4.8 portability |
| `classifier_nested.py` | 4.2.3 and 4.7 nested classifier |
| `compare_baselines.py` | Supplementary Tables S1-S2 and Section 4.9 |
| `derive_map.py` | 3.8.2 Reference map and Family 1 sample. Overwrites frozen files |
| `run_experiments.py` | 3.8 Families 1-4 |
| `recompute_measures.py` | Deviation 1 mask and absolute-z scores |
| `analyse_experiments.py` | Section 5 and Table 3. `--measure=prereg`, `mask`, or `absz` |
| `lmm_crossed.py`, `validate_lmm.py` | Supplementary Section S6 |
| `selftest_positive.py` | Positive control for all 14 hypotheses |
| `diagnostics_measure.py` | Section 5.1 preregistered measure, held-out check |
| `diagnostics_corrected.py` | Section 5.1 corrected measures, held-out check |
| `sensitivity_h14.py`, `sensitivity_h34.py` | Exploratory within-cell analyses in Sections 5.2 and 5.4 |
| `figures_baseline.py`, `figures_classifier.py`, `figures_experiments.py` | Figures 1-8 and 12-23 |
| `supplementary_s5_synthetic_reconfiguration.py` | Supplementary Section S5 and Figure S3 |
| `report_manuscript_statistics.py` | Prints Table 2 and Table 3 from `results/` |
| `verify_preregistration_hashes.py` | Checks the frozen SHA-256 record |
| `legacy_original_pipeline/` | Section 4.3 residual stream, and the earlier uncorrected pipeline |
| `data_preparation/` | Construction of the translated corpus |

`pipeline/` is the loader imported by `common.py`: deduplication, bracket stripping, and the emotion list.
