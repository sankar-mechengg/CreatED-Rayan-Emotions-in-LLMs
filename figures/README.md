# Figures

The figure scripts write PNGs at 300 dpi into this directory. The root `README.md` maps each filename to the manuscript figure number.

```bash
python code/figures_classifier.py
python code/figures_experiments.py
python code/figures_baseline.py
```

`figures_baseline.py` reads `cache/gate_zh.npy` for Figure 3 and Figure 4. The other panels, and the other two scripts, read `results/` only.

Figures 9, 10, and 11 are the residual-stream projections from `code/legacy_original_pipeline/rq3_cluster_enumeration.py`.
