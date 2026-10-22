# Original extraction pipeline

This is the pipeline that produced Section 4.3 of the Revision 5 manuscript and the earlier uncorrected overlap matrix in `data/overlap_matrix_earlier_version.csv`.

Section 4.3 was not repeated when the gates were re-extracted. The published residual-stream values are:

- the first principal component accounts for 99.9% of variance
- overall silhouette score −0.090
- 15-nearest-neighbour purity 0.485 for anger and 0.177 for disgust

Those quantities come from `rq3_cluster_enumeration.py` on the concatenated residual stream of layers 14 to 22 (13,824 dimensions).

The signature rule in `rq2_emotion_specificity.py` is the earlier rule: a one-sided two-proportion z-test at p < 0.05, without Benjamini-Hochberg correction and without a Cohen's h threshold. The corrected rule used for Tables 2 and 3 is `../baseline_analysis_lib.py`.

Run from the repository root:

```bash
python code/legacy_original_pipeline/main.py --csv data/emotion_dataset_translated.csv --output-dir path/to/legacy_output
```

The original run used batches of 8. Outputs are written under the chosen output directory: per-layer `ffn_layer_*.npy` and `residual_layer_*.npy` caches, then `results/rq1_neuron_frequency/`, `rq2_emotion_specificity/`, `rq3_cluster_enumeration/`, `rq4_overlap_analysis/`, and `sensitivity_analysis/`.

UMAP plots require `umap-learn`.
