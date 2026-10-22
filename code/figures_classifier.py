"""Figures from the nested classifier (Chinese originals)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import (
    DATA_CSV, EARLIER_OVERLAP, CACHE, RESULTS, STIMULI, REFERENCE, FIGURES,
    PIPELINE, load_nrc_lexicon, text_open, apply_figure_style, ensure_runtime_dirs,
)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(STIMULI))
ensure_runtime_dirs()

import json, numpy as np, matplotlib, itertools
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
apply_figure_style()
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]; lab = [e.capitalize() for e in EM]
K = json.load(text_open(str(RESULTS / "baseline_zh/classifier_nested.json"))); FIG = str(FIGURES)
fig, ax = plt.subplots(figsize=(5.2, 3)); v = [K["per_class_f1"][e] for e in EM]
ax.bar(lab, v, color="#34495e"); ax.axhline(K["accuracy"], ls="--", color="k")
for i, x in enumerate(v):
    ax.text(i, x + 0.01, f"{x:.2f}", ha="center", fontsize=8)
ax.set_ylabel("F1 (5-fold CV)"); plt.savefig(f"{FIG}/B7_classifier_f1.png"); plt.close()
Cn = np.array(K["confusion_rownorm"]); fig, ax = plt.subplots(figsize=(4.6, 3.9)); im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if Cn[i, j] > 0.5 else "k")
ax.set_xticks(range(6)); ax.set_xticklabels(lab, rotation=40, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(lab)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im, ax=ax, shrink=0.8); plt.savefig(f"{FIG}/B8_confusion.png"); plt.close()
J = np.array(K["fold_mean_jaccard"]); n = np.bincount(np.repeat(np.arange(6), 1))  # placeholder
iu = np.triu_indices(6, 1)
N = np.array([4792, 1290, 1900, 3010, 1808, 3095]); predm = (N[:, None] * Cn).sum(0) / N.sum()
L = np.log(np.maximum(Cn, 1e-6) / predm[None]); S = (L + L.T) / 2
fig, ax = plt.subplots(figsize=(4.8, 3.4)); ax.scatter(J[iu], S[iu], color="#2c3e50")
for i, j in zip(*iu):
    ax.annotate(f"{lab[i][:3]}-{lab[j][:3]}", (J[i, j], S[i, j]), fontsize=7)
r = K["nested_overlap_vs_confusion"]
ax.set_xlabel("Fold-nested corrected overlap (Jaccard)"); ax.set_ylabel("Prior-corrected confusability (log lift)")
ax.set_title(f"rho = {r['rho']:.3f}, exact p = {r['p_exact']:.4f}", fontsize=9); plt.savefig(f"{FIG}/B19_consistency.png"); plt.close()
print("classifier figures done")
