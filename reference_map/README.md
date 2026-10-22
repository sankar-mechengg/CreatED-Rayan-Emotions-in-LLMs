# English reference map

`reference_map_en.npz` is the map fixed before the experiments. It was derived on the English derivation split only (11,126 texts, stratified 70/30, `train_test_split` random_state 42).

Arrays:

| Key | Meaning |
| --- | --- |
| `mu`, `sd` | Per-layer, per-neuron mean and standard deviation of the pooled gate activation on the derivation split |
| `W` | Shape `(9, 8960, 6)`. Entry `1/|S_e|` on each corrected signature neuron of emotion e, else 0 |
| `der`, `held` | Row indices of the derivation and held-out splits |
| `sizes` | Aggregated signature counts: anger 567, disgust 126, fear 592, happiness 677, sadness 320, surprise 1,334 |

`reference_abs_en.npz` stores the mean and standard deviation of the absolute pooled activation on the same derivation split. The absolute-z robustness measure uses it. It was built for Deviation 1 and is not in the pre-experiment hash list.

The SHA-256 of `reference_map_en.npz` is recorded in `preregistration/HASHES.txt`. The identical bytes are also at `preregistration/hashed_files_as_frozen/cache/reference_map_en.npz`, which is the path used when the hash was recorded.

The derivation-split Jaccard matrix is `results/reference_map_jaccard_en.json`.
