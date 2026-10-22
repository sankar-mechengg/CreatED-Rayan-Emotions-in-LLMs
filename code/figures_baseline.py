"""Figures for the corrected baseline (R4). Chinese originals are primary; English shown where stated."""

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

import json, itertools, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from common import EM, LAYERS, FFN
from baseline_analysis_lib import topk_mask, ztest, cohen_h, bh
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10, "axes.titlesize": 11, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
apply_figure_style()
FIG = str(FIGURES); import os; os.makedirs(FIG, exist_ok=True)
R = str(RESULTS)
Z = json.load(text_open(f"{R}/baseline_zh/baseline_results.json")); E = json.load(text_open(f"{R}/baseline_en/baseline_results.json"))
COL = {"ANGER": "#c0392b", "DISGUST": "#7d6608", "FEAR": "#6c3483", "HAPPINESS": "#f1c40f", "SADNESS": "#2471a3", "SURPRISE": "#17a589"}
lab = [e.capitalize() for e in EM]; iu = np.triu_indices(6, 1)
QUAD = {"HAPPINESS": 0, "SURPRISE": 0, "ANGER": 1, "DISGUST": 1, "FEAR": 2, "SADNESS": 2}

def save(name):
    plt.savefig(f"{FIG}/{name}.png"); plt.close()

# B1 consistent counts
fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.13
for i, e in enumerate(EM):
    ax.bar(np.arange(9) + (i - 2.5) * w, [Z["consistent_counts"][str(l)][e] for l in LAYERS], w, color=COL[e], label=lab[i])
ax.set_xticks(range(9)); ax.set_xticklabels(LAYERS); ax.set_xlabel("Layer"); ax.set_ylabel("Consistently active neurons (freq > 0.5)")
ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper left"); save("B1_consistent_counts")

# B2 chi-square heatmaps, uncorrected versus FDR
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, k, t in zip(axs, ("chi2_uncorrected", "chi2_fdr"), ("Uncorrected p < 0.05", "BH-FDR q < 0.05")):
    A = np.array([[Z[k][str(l)][e] for l in LAYERS] for e in EM])
    im = ax.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=4500)
    for i in range(6):
        for j in range(9):
            ax.text(j, i, A[i, j], ha="center", va="center", fontsize=6.5, color="w" if A[i, j] < 2500 else "k")
    ax.set_xticks(range(9)); ax.set_xticklabels(LAYERS); ax.set_yticks(range(6)); ax.set_yticklabels(lab); ax.set_title(t); ax.set_xlabel("Layer")
fig.colorbar(im, ax=axs, shrink=0.8, label="Significant neurons"); save("B2_chi_square")

# B5 signature heatmap corrected (zh) + English totals
A = np.array([[Z["signature_counts"]["fdr"][e][str(l)] for l in LAYERS] for e in EM])
fig, ax = plt.subplots(figsize=(6.4, 3.2)); im = ax.imshow(A, cmap="magma_r", aspect="auto")
for i in range(6):
    for j in range(9):
        ax.text(j, i, A[i, j], ha="center", va="center", fontsize=7, color="w" if A[i, j] > 120 else "k")
ax.set_xticks(range(9)); ax.set_xticklabels(LAYERS); ax.set_yticks(range(6))
ax.set_yticklabels([f"{lab[i]} ({Z['signature_totals']['fdr'][e]})" for i, e in enumerate(EM)]); ax.set_xlabel("Layer")
fig.colorbar(im, ax=ax, label="Signature neurons"); save("B5_signatures_fdr")

# B6 split half corrected vs uncorrected
fig, ax = plt.subplots(figsize=(6.4, 3)); x = np.arange(6)
for k, off, c in (("unc", -0.18, "#bbbbbb"), ("fdr", 0.18, "#34495e")):
    m = [Z["split_half_L18"][k][e][0] for e in EM]; s = [Z["split_half_L18"][k][e][1] for e in EM]
    ax.bar(x + off, m, 0.36, yerr=s, color=c, label={"unc": "Uncorrected (R3 criterion)", "fdr": "Corrected (FDR + h)"}[k], capsize=2)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("Split-half Jaccard (layer 18)"); ax.legend(frameon=False, fontsize=8); save("B6_split_half")

# B12 Jaccard heatmaps zh and en
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"wspace": 0.08})
for ax, D, t in zip(axs, (Z, E), ("Chinese originals", "English translations")):
    J = np.array(D["jaccard"]["fdr"]); M = J.copy(); np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=0.075)
    for i in range(6):
        for j in range(6):
            if i != j:
                ax.text(j, i, f"{J[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(6)); ax.set_xticklabels(lab, rotation=40, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(lab if D is Z else []); ax.set_title(t)
fig.colorbar(im, ax=axs, shrink=0.8, label="Jaccard (corrected signatures)"); save("B12_jaccard_zh_en")

# B13 exclusive/shared corrected
fig, ax = plt.subplots(figsize=(6.4, 3))
ex = [Z["exclusive_shared"]["fdr"][e]["exclusive"] for e in EM]; sh = [Z["exclusive_shared"]["fdr"][e]["shared"] for e in EM]
ax.bar(lab, ex, color=[COL[e] for e in EM], label="Exclusive"); ax.bar(lab, sh, bottom=ex, color=[COL[e] for e in EM], alpha=0.35, label="Shared")
for i in range(6):
    ax.text(i, ex[i] + sh[i] + 15, f"{100 * ex[i] / (ex[i] + sh[i]):.0f}% excl.", ha="center", fontsize=7.5)
ax.set_ylabel("Corrected signature neurons"); ax.legend(frameon=False, fontsize=8); save("B13_exclusive_shared")

# B14 dendrogram
J = np.array(Z["jaccard"]["fdr"]); D = 1 - J; np.fill_diagonal(D, 0)
Lk = linkage(squareform(D, checks=False), "average")
fig, ax = plt.subplots(figsize=(5.5, 3)); dendrogram(Lk, labels=lab, ax=ax, color_threshold=0)
ax.set_ylabel("Jaccard distance"); ax.set_ylim(0.9, 1.001); save("B14_dendrogram")

# B15 ranked pairs zh with en markers
pairs = [(i, j) for i, j in zip(*iu)]; vals = [J[i, j] for i, j in pairs]; order = np.argsort(vals)[::-1]
Je = np.array(E["jaccard"]["fdr"])
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for r, o in enumerate(order):
    i, j = pairs[o]; adj = QUAD[EM[i]] == QUAD[EM[j]]
    ax.bar(r, vals[o], color="#34495e" if adj else "#bdc3c7")
    ax.plot(r, Je[i, j], "o", color="#e67e22", ms=4)
ax.set_xticks(range(15)); ax.set_xticklabels([f"{lab[pairs[o][0]][:3]}-{lab[pairs[o][1]][:3]}" for o in order], rotation=60, ha="right", fontsize=8)
c = Z["circumplex"]["fdr"]["graded"]
ax.set_ylabel("Jaccard (corrected)"); ax.set_title(f"Dark bars: circumplex-adjacent pairs. Graded test (Chinese): rho = {c[0]:.3f}, exact p = {c[1]:.3f}", fontsize=9)
ax.plot([], [], "o", color="#e67e22", label="English translations"); ax.legend(frameon=False, fontsize=8); save("B15_pairs_ranked")

# B17 sensitivity
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); th = ["1.0", "3.0", "5.0", "10.0"]
for e in EM:
    axs[0].plot([1, 3, 5, 10], [Z["sensitivity"][t]["consistent_total"][e] for t in th], "o-", color=COL[e], label=e.capitalize())
    axs[1].plot([1, 3, 5, 10], [Z["sensitivity"][t]["signature_total"][e] for t in th], "o-", color=COL[e])
axs[0].set_xlabel("Top-K threshold (%)"); axs[0].set_ylabel("Consistent neurons (all layers)"); axs[0].legend(frameon=False, fontsize=7)
axs[1].set_xlabel("Top-K threshold (%)"); axs[1].set_ylabel("Corrected signature neurons")
axs[1].set_title("Graded circumplex rho: " + ", ".join(f"{t[:-2]}%: {Z['sensitivity'][t]['graded_circumplex'][0]:.2f}" for t in th), fontsize=8)
save("B17_sensitivity")

# B18 neuron versus lexical by circumplex relationship
from sklearn.feature_extraction.text import TfidfVectorizer
from common import load_df
df = load_df()
docs = [" ".join(df.loc[df.primary_emotion == e, "english_text"].astype(str)) for e in EM]
Xt = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit_transform(docs).toarray()
Xt /= np.linalg.norm(Xt, axis=1, keepdims=True); LEX = Xt @ Xt.T
grp = []
for i, j in pairs:
    a, b = EM[i], EM[j]
    grp.append(2 if QUAD[a] == QUAD[b] else (1 if (QUAD[a] > 0) == (QUAD[b] > 0) else 0))
grp = np.array(grp)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, v, t in zip(axs, (J[iu], LEX[iu]), ("Neuron signature overlap (corrected)", "Lexical similarity (TF-IDF)")):
    zv = (v - v.mean()) / v.std()
    for g, name in zip((2, 1, 0), ("Adjacent", "Same valence", "Opposite valence")):
        ax.scatter(np.full((grp == g).sum(), 2 - g) + np.random.default_rng(g).uniform(-0.08, 0.08, (grp == g).sum()), zv[grp == g], s=22)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Adjacent", "Same valence", "Opposite valence"]); ax.set_title(t, fontsize=9)
axs[0].set_ylabel("Standardised similarity"); save("B18_lexical")
json.dump({"lexical_matrix": LEX.tolist()}, text_open(f"{R}/lexical_matrix.json", "w"))

if "--light" in sys.argv:
    print("light done"); sys.exit()
# B3 permutation null (anger, layer 18) and B4 volcano (disgust, layer 14), Chinese
X = np.load(str(CACHE / "gate_zh.npy"), mmap_mode="r"); labels = df.primary_emotion.values
Y = np.stack([labels == e for e in EM], 1).astype(np.float32); N = len(labels)
M18 = topk_mask(np.asarray(X[:, LAYERS.index(18), :], dtype=np.float32), 5.0).astype(np.float32)
rng = np.random.default_rng(42); ya = Y[:, 0]; n_a = ya.sum()
obs = int(((ya @ M18) / n_a > 0.5).sum())
null = np.array([int(((ya[rng.permutation(N)] @ M18) / n_a > 0.5).sum()) for _ in range(1000)])
fig, ax = plt.subplots(figsize=(5.5, 3)); ax.hist(null, bins=25, color="#bdc3c7"); ax.axvline(obs, ls="--", color=COL["ANGER"])
ax.set_xlabel("Consistently active neurons under shuffled labels"); ax.set_ylabel("Count")
ax.set_title(f"Anger, layer 18: observed {obs}, permutation p = {(1 + (null >= obs).sum()) / 1001:.4f}", fontsize=9); save("B3_permutation_anger_L18")
del M18
M14 = topk_mask(np.asarray(X[:, LAYERS.index(14), :], dtype=np.float32), 5.0).astype(np.float32)
yd = Y[:, 1]; act_t = yd @ M14; act_o = M14.sum(0) - act_t
pt, po, p = ztest(act_t, yd.sum(), act_o, N - yd.sum()); h = cohen_h(pt, po); q = bh(p)
sig = (q < 0.05) & (h >= 0.2) & (pt > po)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.scatter(h[~sig], -np.log10(np.maximum(p[~sig], 1e-300)), s=2, color="#bbbbbb")
ax.scatter(h[sig], -np.log10(np.maximum(p[sig], 1e-300)), s=5, color=COL["DISGUST"], label=f"Corrected signatures (n = {sig.sum()})")
ax.axvline(0.2, ls=":", color="k"); ax.set_xlabel("Cohen's h (disgust versus all other)"); ax.set_ylabel("-log10 p (one-sided)")
ax.legend(frameon=False, fontsize=8); save("B4_volcano_disgust_L14")
print("figures done")
