# Emotion-associated representations in the feed-forward gates of a language model

Reproduction package for the Revision 5 manuscript by Rayan Upadhyay (The International School Bangalore) and Sankar Balasubramanian (Indian Institute of Science).

The manuscript separates three claims. Representation is supported: after Benjamini-Hochberg correction and a minimum effect size, about 4,400 feed-forward gate associations remain, and their overlap follows the circumplex model of affect. Reconfiguration is supported as a measured property of those signatures under the post hoc corrected measures. Control is not established: no experiment tests whether the identified neurons are necessary or sufficient for emotional behaviour.

Revision 5 recalibrated those claims and added the Perspectives section. It did not add an experiment or a new statistical analysis. The response to reviewers states that explicitly. The numbers in Revision 5 are the numbers produced by the scripts in this folder.

Licence: CC-BY-4.0. See `LICENSE`.

## What is in this folder

| Path | Role |
| --- | --- |
| `code/` | Runnable analysis, extraction, experiments, figures, and checks |
| `code/pipeline/` | Dataset loading used by the Revision 5 scripts |
| `code/legacy_original_pipeline/` | Original pipeline that produced the residual-stream results in Section 4.3 and the earlier uncorrected overlap matrix |
| `code/data_preparation/` | Scripts that built the bilingual corpus |
| `code/supplementary_s5_synthetic_reconfiguration.py` | Synthetic validation in Supplementary Section S5 |
| `data/emotion_dataset_translated.csv` | Analysis corpus: 16,667 rows; the loader keeps 15,895 |
| `data/overlap_matrix_earlier_version.csv` | Earlier uncorrected Jaccard matrix, the first column of Supplementary Table S1 |
| `stimuli/` | Family 1 held-out items and the Family 2-4 stimulus modules |
| `reference_map/` | English derivation-split signature map and absolute-activation scale |
| `preregistration/` | Frozen preregistration, Deviation 1, and the SHA-256 records |
| `preregistration/hashed_files_as_frozen/` | Byte-identical copies of every file named in those hash records |
| `results/` | Published baseline tables, confirmatory tests, and trial-level experiment outputs |
| `model_settings.json` | Model, layers, signature rule, label-token ids, and the three internal measures |
| `cache/` | Created when activations are extracted. Not shipped, because each language is about 2.6 GB |
| `figures/` | Created by the figure scripts |

The runnable scripts under `code/` use paths inside this repository. That changes their bytes relative to the preregistration freeze. The files that the freeze hashed are preserved, unchanged, under `preregistration/hashed_files_as_frozen/`.

## Check the released numbers without loading the model

From this folder:

```bash
pip install -r requirements.txt
python code/verify_preregistration_hashes.py
python code/report_manuscript_statistics.py
```

The report reads `results/` and prints Table 2, the sensitivity totals, and Table 3 for all three internal measures. With the files as released it prints:

- 4,444 corrected Chinese signature neurons (844, 106, 818, 739, 473, 1,464) and 3,569 English
- graded circumplex rho 0.888 (Chinese) and 0.773 (English), exact p 0.022
- partial rho given lexical similarity 0.855 (Chinese) and 0.714 (English)
- Chinese-English overlap correlation rho 0.920, exact p 0.0014
- 2, 10, and 11 of the 14 hypotheses supported after Holm correction, for the preregistered, mask, and absolute-z measures

Recomputing the 14 tests from the saved trials, still without Qwen:

```bash
python code/analyse_experiments.py --measure=prereg
python code/analyse_experiments.py --measure=mask
python code/analyse_experiments.py --measure=absz
```

`--measure=prereg` is the confirmatory test as frozen. `mask` is the primary post hoc measure (fraction of signature neurons in the top 5%). `absz` is the robustness measure (mean standardised absolute activation). Each command rewrites `results/confirmatory_<measure>_results.json`. Copy that file first if you want to keep the released bytes.

These three commands need `nrclex==4.1.0`, because Families 3 and 4 score generated text with the NRC lexicon at analysis time. They read `results/experiments/F1.jsonl` through `F4.jsonl` and `results/experiments/measures_F1.json` through `measures_F4.json`.

## Manuscript map

Emotions are ordered anger, disgust, fear, happiness, sadness, surprise. Layers are 14 to 22 inclusive. The gate activation is `act_fn(gate_proj(x)) * up_proj(x)`, 8,960 units, attention-masked mean over tokens, top 5% by absolute value.

### Corpus and model

| Manuscript | Code | Output |
| --- | --- | --- |
| Section 3.1, corpus of 15,895 texts | `code/pipeline/preprocessing.py`, called by `code/common.py` | Anger 4,792, surprise 3,095, happiness 3,010, fear 1,900, sadness 1,808, disgust 1,290 |
| Sections 3.2-3.3, gate extraction in Chinese and English | `code/extract_baseline.py` | `cache/gate_zh.npy`, `cache/gate_en.npy`, `cache/labels.npy` |
| Section 3.3.2, residual stream, not repeated in this revision | `code/legacy_original_pipeline/extraction.py` | `residual_layer_14.npy` through `residual_layer_22.npy` in that pipeline's cache |

`extract_baseline.py zh` and `extract_baseline.py en` run raw text, without a chat template, in length-sorted batches of 32, truncated at 512 tokens, stored as float16. A second argument is a resume row, a multiple of 32. The published extraction used Qwen/Qwen2.5-1.5B-Instruct in bfloat16 on CPU with 2 threads.

### Baseline, RQ1 to RQ5

`code/baseline_analysis.py zh` and `code/baseline_analysis.py en` implement Sections 3.4 and 3.5. A neuron is a corrected signature when, inside its layer and emotion, the Benjamini-Hochberg q-value is below 0.05, Cohen's h is at least 0.2, and the target rate exceeds the other rate. The same script keeps the earlier uncorrected rule (one-sided p < 0.05) so Supplementary Tables S1 and S2 can be rebuilt.

| Manuscript | What the script computes |
| --- | --- |
| 4.1.1 Consistent neurons, frequency above 0.5 | `consistent_counts` |
| 4.1.2 Chi-square, uncorrected and BH-FDR | `chi2_uncorrected`, `chi2_fdr` |
| 4.1.3 Permutation test at layer 18, 1,000 shuffles | `permutation_L18` |
| 4.1.4 Volcano data, Cohen's h | Used by `figures_baseline.py` |
| 4.2.1 Corrected signature counts | `signature_counts`, `signature_totals`, `signature_enumeration_corrected.csv` |
| 4.2.2 Split-half Jaccard at layer 18, 100 splits | `split_half_L18` |
| 4.4 Jaccard, exclusive and shared neurons, dendrogram, circumplex test over 720 relabellings | `jaccard`, `exclusive_shared`, `circumplex` |
| 4.5 Thresholds 1%, 3%, 5%, 10% | `sensitivity` |
| 4.6 Lexical TF-IDF control and partial Spearman | `circumplex[*].partial_given_lex` |
| S3, one secondary-label Jaccard inside the main script | `human_annotation` |
| S4 Triangle-inequality slack | `circumplex[*].triangle_min_slack` |

`code/baseline_analysis_lib.py` holds the shared top-5% mask, z-test, Cohen's h, BH-FDR, and signature rule. `code/derive_map.py` and the baseline script both call it.

`code/baseline_extras.py zh` and `en` add the five secondary-annotation similarities (Supplementary Table S5), cross-layer portability of the top-20 indices (Section 4.8), the within-layer top-20 backbone, pairwise intersection sizes, and the same-valence versus opposite-valence overlap ratio.

`code/classifier_nested.py zh` is Section 3.4.7 and Section 4.7: 5-fold logistic regression on the 80,640 binary gate features, with corrected signatures recomputed inside each training fold and compared with held-out confusability. The published Chinese result is rho 0.789, exact p 0.0014.

`code/compare_baselines.py` writes Supplementary Tables S1 and S2 and the Chinese-English correlation in Section 4.9. It reads `data/overlap_matrix_earlier_version.csv` for the earlier published overlaps. The earlier signature totals are the constants in that script (anger 14,165 through surprise 23,115), which are the values in the earlier manuscript.

### Reference map and the four task families

| Manuscript | Code | Released file |
| --- | --- | --- |
| Section 3.8.2, English 70/30 split, seed 42 | `code/derive_map.py` | `reference_map/reference_map_en.npz`, `results/reference_map_jaccard_en.json`, `stimuli/f1_items.csv` |
| Family 1, 480 held-out texts | `stimuli/f1_items.csv`, trials built in `run_experiments.py` | `results/experiments/F1.jsonl` |
| Family 2, 120 written scenarios | `stimuli/f2_disambiguation.py` | `results/experiments/F2.jsonl` |
| Families 3 and 4 | `stimuli/f3_f4_generation.py` | `results/experiments/F3.jsonl`, `F4.jsonl` |
| Sections 3.8.4-3.8.5 and Section 5 | `code/analyse_experiments.py` | `results/confirmatory_prereg_results.json`, `confirmatory_mask_results.json`, `confirmatory_absz_results.json` |
| Deviation 1, mask and absolute-z scores | `code/recompute_measures.py` | `results/experiments/measures_F1.json` through `measures_F4.json` |
| Section 5.1 held-out validity | `code/diagnostics_measure.py`, `code/diagnostics_corrected.py` | `results/diagnostics_measure.json`, `diagnostics_measure_corrected.json` |
| Within-cell checks of H1.4 and H3.4 | `code/sensitivity_h14.py`, `code/sensitivity_h34.py` | `results/sensitivity_h14.json`, `sensitivity_h34.json` |

`derive_map.py` overwrites the reference map, the Family 1 item list, and the reference Jaccard file. Those three are part of the preregistration freeze. Copy them before a re-run.

`run_experiments.py` takes `F1`, `F2`, `F3`, `F4`, or `PILOT`. Prompts use the chat template, instruction before stimulus. Forced choice prefixes the assistant turn with `Answer:` and reads the next-token log-probabilities of ` anger`, ` disgust`, ` fear`, ` happiness`, ` sadness`, and ` surprise` (token ids 19234, 67062, 8679, 23009, 50878, 12761). Generation is greedy, at most 80 new tokens, seed 42. Completed trials are appended and are not recomputed. Families 3 and 4 also score text with `j-hartmann/emotion-english-distilroberta-base`, with joy mapped to happiness.

The preregistered score of a token is the mean signed z-scored activation over the signature neurons. Deviation 1, hashed before the corrected scores were used, adds the mask fraction and the absolute-z mean. `recompute_measures.py` builds those two from per-trial activation arrays named `results/experiments/raw_F1_*.npz` and the same pattern for F2, F3, and F4. Those raw arrays are not in this release. The corrected scores already computed from them are in `measures_F*.json`, and `analyse_experiments.py` reads those.

### Mixed models, self-tests, and Supplementary Section S6

| Piece | Code |
| --- | --- |
| REML linear mixed model with crossed item and paraphrase intercepts | `code/lmm_crossed.py` |
| Check against statsmodels MixedLM on 720 synthetic rows | `code/validate_lmm.py` |
| Null self-test: 14 hypotheses non-significant after Holm | `python code/analyse_experiments.py --selftest` |
| Positive self-test: an injected effect on every hypothesis | `python code/selftest_positive.py` |

`validate_lmm.py` rewrites `results/validate_lmm.json`. The released file records agreement to at least six significant figures, as in Supplementary Section S6. On some statsmodels versions the variance-component vector is stored in a different order; compare the fixed effect and its standard error first.

Logistic outcomes use `BinomialBayesMixedGLM` (variational Bayes). The reported p-value is the normal tail of the posterior mean divided by the posterior standard deviation. Holm correction is applied across the 14 confirmatory tests. H4.1, the Family 4 classifier manipulation check, is outside that family.

### Figures

Run the figure scripts from this folder after the result JSON files are in place. `figures_baseline.py` also reads `cache/gate_zh.npy` for the permutation histogram and the disgust volcano plot, so those two panels need a Chinese extraction. The other panels read `results/`.

| File | Manuscript figure |
| --- | --- |
| `figures/B1_consistent_counts.png` | Figure 1 |
| `figures/B2_chi_square.png` | Figure 2 |
| `figures/B3_permutation_anger_L18.png` | Figure 3 |
| `figures/B4_volcano_disgust_L14.png` | Figure 4 |
| `figures/B5_signatures_fdr.png` | Figure 5 |
| `figures/B6_split_half.png` | Figure 6 |
| `figures/B7_classifier_f1.png` | Figure 7 |
| `figures/B8_confusion.png` | Figure 8 |
| `legacy_original_pipeline` UMAP, silhouette, purity | Figures 9, 10, and 11 |
| `figures/B12_jaccard_zh_en.png` | Figure 12 |
| `figures/B13_exclusive_shared.png` | Figure 13 |
| `figures/B14_dendrogram.png` | Figure 14 |
| `figures/B15_pairs_ranked.png` | Figure 15 |
| `figures/B17_sensitivity.png` | Figure 16 |
| `figures/B18_lexical.png` | Figure 17 |
| `figures/B19_consistency.png` | Figure 18 |
| `figures/E0_measure_validity.png` | Figure 19 |
| `figures/E1_family1.png` | Figure 20 |
| `figures/E2_family2.png` | Figure 21 |
| `figures/E3_family3.png` | Figure 22 |
| `figures/E4_family4.png` | Figure 23 |

```bash
python code/figures_classifier.py
python code/figures_experiments.py
python code/figures_baseline.py
```

Figures are written at 300 dpi. Liberation Serif is used when it is installed; otherwise the script uses Times New Roman or DejaVu Serif.

### Residual stream (Section 4.3)

Section 4.3 was not repeated on the CPU re-extraction. The published values come from the original pipeline:

- one principal component explains 99.9% of residual-stream variance
- overall silhouette score −0.090
- 15-nearest-neighbour purity from 0.177 (disgust) to 0.485 (anger)

That code is `code/legacy_original_pipeline/`. It also contains the earlier, uncorrected signature analysis (p < 0.05, no Cohen's h threshold). Do not use it to rebuild Table 2. Table 2 is rebuilt by `baseline_analysis.py`.

```bash
python code/legacy_original_pipeline/main.py --csv data/emotion_dataset_translated.csv --output-dir path/to/legacy_output
```

The original extraction used batches of 8 on a GPU. The Revision 5 gate re-extraction uses batches of 32 on CPU. The two caches are not interchangeable.

### Supplementary Section S5

`code/supplementary_s5_synthetic_reconfiguration.py` validates five reconfiguration metrics on synthetic units with a known imposed change. This is Figure S3. The paired paradigms in that file, including novice versus expert, literal versus figurative, and cooperative versus adversarial reading, were withdrawn from the experiments in Revision 5. The synthetic check does not load a model:

```bash
python code/supplementary_s5_synthetic_reconfiguration.py --selftest
```

### Section 8.3, the protocol for another model

The six steps in Section 8.3 are the corrected baseline, written so a later study can run them without new decisions:

1. Choose a gated feed-forward model and a corpus with the same emotion labels.
2. Pool the upper third of the layers and binarise at a fixed top-K percentage. The value of K is varied in Section 4.5.
3. Keep neurons with BH-FDR q < 0.05 inside each layer and emotion, Cohen's h at least 0.2, and a higher rate on the target emotion. This is `signatures()` in `baseline_analysis_lib.py`.
4. Build the pairwise Jaccard matrix over the aggregated `(layer, neuron)` sets.
5. Spearman-correlate that matrix with the graded circumplex matrix, with an exact p-value from all relabellings.
6. Repeat the correlation after partialling out a lexical similarity matrix from the same corpus.

Any score derived from the signatures should be checked on held-out texts before it is used in an experiment. That check is `diagnostics_measure.py` and `diagnostics_corrected.py`.

### Corpus construction

`code/data_preparation/` produced `data/emotion_dataset_translated.csv` (16,667 rows). The analysis loader drops duplicate pairs of Chinese text and primary label. On this file that removes 772 rows and leaves 15,895, which is the N stored in `results/baseline_zh/baseline_results.json` and the class counts in Section 3.1. The manuscript sentence that says 682 duplicates is not what this CSV and loader produce; 16,667 − 15,895 = 772.

| Script | Step |
| --- | --- |
| `emotion_classifier.py` | Primary and secondary emotion labels, GLM-4.5 |
| `filter_empty_emotions.py` | Drop rows with an empty emotion field |
| `translate_chinese.py` | English translations, GLM-4.5 |
| `check_translations.py` | Check that translations were written |
| `emotion_summary.py` | Print the class counts |

The classifier and translator read `GLM_API_KEY` from the environment and stop if it is absent. They are not required for any manuscript statistic once the translated CSV is present. No API key is stored in this folder.

## Preregistration

The freeze is 2026-10-05T20:36:36Z, before any experimental activation was extracted. Deviation 1 is 2026-10-05T22:31:21Z, after the preregistered-measure results of Families 1 to 3 had been seen and before any result had been computed with the corrected measures. The registration was internal: time-stamped and hashed, not uploaded to a public registry. The manuscript states that.

```bash
python code/verify_preregistration_hashes.py
```

Thirteen of the fourteen recorded hashes match. `analyse_experiments.py` does not. The freeze record is `648d1b3e78b5f456af4c3ae9d6b23305ac0d1f069f718f6b346d67d7f6dfbf16`. The script in the supplementary archive, which is the script that reads all three internal measures and wrote the released confirmatory JSON, hashes to `784b5ad06b0e75db817d57eb3c5ac375a56664ebd8531e862c629181bf82553e`. The pre-freeze source was not among the project files, so this package keeps the archive script and reports the difference as `LATER` rather than inventing the missing bytes. Every other recorded file matches.

`HASHES.txt` covers the preregistration, `derive_map.py`, `run_experiments.py`, `analyse_experiments.py`, `selftest_positive.py`, `common.py`, `baseline_analysis_lib.py`, the three stimulus files, `reference_map_en.npz`, and `reference_map_jaccard_en.json`.

`HASHES_DEVIATION_1.txt` covers `DEVIATION_1.md` and `diagnostics_measure.py`.

The hash of `reference_map_en.npz` was computed when that file lived in `cache/`. The same bytes are stored at `reference_map/reference_map_en.npz` and at `preregistration/hashed_files_as_frozen/cache/reference_map_en.npz`.

## Full re-run

Expect roughly 25 minutes per language for extraction, 20 minutes per language for the corrected baseline, and about four hours for the four experiment families, on the CPU configuration in `model_settings.json`. A GPU runs the same code; set the device inside the installed PyTorch build. Extraction and experiments download `Qwen/Qwen2.5-1.5B-Instruct`. Families 3 and 4 also download the emotion classifier.

```bash
python code/extract_baseline.py zh
python code/extract_baseline.py en
python code/baseline_analysis.py zh
python code/baseline_analysis.py en
python code/baseline_extras.py zh
python code/baseline_extras.py en
python code/classifier_nested.py zh
python code/compare_baselines.py
python code/derive_map.py
python code/run_experiments.py F1
python code/run_experiments.py F2
python code/run_experiments.py F3
python code/run_experiments.py F4
python code/recompute_measures.py F1
python code/recompute_measures.py F2
python code/recompute_measures.py F3
python code/recompute_measures.py F4
python code/analyse_experiments.py --measure=prereg
python code/analyse_experiments.py --measure=mask
python code/analyse_experiments.py --measure=absz
```

`recompute_measures.py` runs only after `run_experiments.py` has written `results/experiments/raw_<family>_*.npz`. Those arrays are produced by the experiment runner and are not part of the released archive.

Run the checks before treating a fresh experimental run as confirmatory:

```bash
python code/analyse_experiments.py --selftest
python code/selftest_positive.py
python code/validate_lmm.py
python code/diagnostics_measure.py
python code/diagnostics_corrected.py
```

## Result files

| File | Contents |
| --- | --- |
| `results/baseline_zh/baseline_results.json` | Chinese RQ1-RQ5, both criteria, circumplex, lexical control, sensitivity |
| `results/baseline_en/baseline_results.json` | The same for the English translations |
| `results/baseline_zh/signatures.npz` | Boolean signature masks, uncorrected and corrected, per layer and emotion |
| `results/baseline_zh/signature_enumeration_corrected.csv` | Top corrected neurons by Cohen's h |
| `results/baseline_zh/classifier_nested.json` | Nested classifier and the overlap-versus-confusion test |
| `results/baseline_zh/baseline_extras.json` | Secondary-annotation tests, portability, backbone, valence ratio |
| `results/baseline_comparison.json` | Supplementary Tables S1-S2 summary and the cross-language test |
| `results/supp_table_jaccard_original_vs_corrected.csv` | Supplementary Table S1 |
| `results/supp_table_signature_counts.csv` | Supplementary Table S2 |
| `results/lexical_matrix.json` | English TF-IDF cosine matrix |
| `results/reference_map_jaccard_en.json` | Derivation-split Jaccard matrix used by H3.2 and H3.3 |
| `results/experiments/F1.jsonl` to `F4.jsonl` | Trial prompts, log-probabilities or generated text, and preregistered scores |
| `results/experiments/measures_F1.json` to `measures_F4.json` | Mask and absolute-z scores for the same trials |
| `results/confirmatory_*_results.json` | All 14 tests, secondary analyses, and Holm p-values for one measure |
| `results/diagnostics_measure.json` | Held-out failure of the preregistered signed score |
| `results/diagnostics_measure_corrected.json` | Held-out validity of the two corrected scores |
| `results/sensitivity_h14.json`, `sensitivity_h34.json` | Exploratory within-cell analyses |
| `results/selftest_null_summary.txt`, `selftest_positive_summary.txt` | Supplementary Table S6 style checks |
| `results/validate_lmm.json` | Mixed-model software check |

## Code that was added for this package

The statistical procedures are the ones that produced the manuscript. Four pieces were added so the package runs outside the original machine and so the missing supplementary check is present:

- `code/repo_paths.py` replaces the original absolute paths, opens text as UTF-8, and finds `nrclex/data/nrc_en.json` inside the installed package.
- `code/verify_preregistration_hashes.py` checks the frozen files.
- `code/report_manuscript_statistics.py` prints the released Table 2 and Table 3 values.
- `code/supplementary_s5_synthetic_reconfiguration.py` is the synthetic validation reported in Supplementary Section S5. It was written for the earlier revision and was not inside the Revision 4 archive. Revision 5 still reports that validation, so the script is included here. Its paradigms are not the experiments in Section 5.

`common.py` imports PyTorch inside `load_model()` so the baseline and figure scripts can be imported without loading the model library. `load_model()` still sets two CPU threads, as in the published run.

## Citation

Upadhyay, R., & Balasubramanian, S. Emotion-associated representations in the feed forward gates of a language model. Revision 5.

The Family 2 to 4 stimuli and the analysis code were written with an AI system (Claude, Anthropic) under the direction of the authors, and were reviewed by them. The manuscript records the same disclosure.
