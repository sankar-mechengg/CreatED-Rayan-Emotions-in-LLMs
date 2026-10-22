"""Exploratory (not preregistered) checks of the internal measure, used for interpretation only.
(1) Held-out validity: emotion scores computed from the cached mean-pooled activations of the
English held-out texts (no prompt), using the reference map; six-way argmax accuracy and within-pair
discrimination (AUC) against the human labels. (2) Family 1: within-pair discrimination of the gold
label by the stimulus-span pair axis in each condition (AUC), i.e. whether the map separates the pair
inside the prompts at all."""

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

import json, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import EM
ref = np.load(str(REFERENCE / "reference_map_en.npz")); mu, sd, W = ref["mu"], ref["sd"], ref["W"]; held = ref["held"]
X = np.load(str(CACHE / "gate_en.npy"), mmap_mode="r"); lab = np.load(str(CACHE / "labels.npy"), allow_pickle=True).astype(str)
S = np.zeros((len(held), 6))
for s in range(0, len(held), 500):
    idx = held[s:s + 500]; A = np.asarray(X[idx], dtype=np.float32)
    Z = (A - mu[None]) / sd[None]; S[s:s + 500] = np.einsum("nlf,lfe->ne", Z, W)
y = np.array([EM.index(e) for e in lab[held]])
Sc = S - S.mean(0)
out = {"heldout_n": int(len(held)), "heldout_argmax_acc_centred": float((Sc.argmax(1) == y).mean()),
       "majority": float(np.bincount(y).max() / len(y))}
pairs = [("ANGER", "DISGUST"), ("FEAR", "SADNESS"), ("HAPPINESS", "SURPRISE")]
out["heldout_pair_auc"] = {}
for a, b in pairs:
    m = np.isin(y, [EM.index(a), EM.index(b)])
    out["heldout_pair_auc"][f"{a}-{b}"] = float(roc_auc_score(y[m] == EM.index(a), S[m, EM.index(a)] - S[m, EM.index(b)]))
out["heldout_one_vs_rest_auc"] = {e: float(roc_auc_score(y == i, Sc[:, i])) for i, e in enumerate(EM)}
f1 = pd.DataFrame([json.loads(l) for l in text_open(str(RESULTS / "experiments" / "F1.jsonl"))])
f1["stim"] = f1.span_scores.map(lambda d: d["stim"])
out["F1_pair_auc_by_condition"] = {}
for (pr, cond), g in f1.groupby(["pair", "cond"]):
    a, b = pr.split("-")
    ax = np.array([s[EM.index(a)] - s[EM.index(b)] for s in g.stim])
    axf = np.array([s[EM.index(a)] - s[EM.index(b)] for s in g.final])
    beh = np.array([l[EM.index(a)] - l[EM.index(b)] for l in g.lp])
    out["F1_pair_auc_by_condition"][f"{pr}|{cond}"] = {"stim_span": float(roc_auc_score(g.gold == a, ax)),
        "final_token": float(roc_auc_score(g.gold == a, axf)), "behaviour": float(roc_auc_score(g.gold == a, beh))}
json.dump(out, text_open(str(RESULTS / "diagnostics_measure.json"), "w"), indent=1); print(json.dumps(out, indent=1))
