"""Deviation 1 support: held-out validity of the corrected measures (mask fraction, absolute z) on the
English held-out texts (baseline activations, no prompt), and within-pair AUC of the corrected
stimulus-span pair axis in Family 1 by condition. Exploratory; used for interpretation."""

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
from baseline_analysis_lib import topk_mask
ref = np.load(str(REFERENCE / "reference_map_en.npz")); held = ref["held"]
Wb = (ref["W"] > 0).astype(np.float32); Wn = Wb / Wb.sum((0, 1), keepdims=True)
ab = np.load(str(REFERENCE / "reference_abs_en.npz")); MU, SD = ab["mu"], ab["sd"]
X = np.load(str(CACHE / "gate_en.npy"), mmap_mode="r"); lab = np.load(str(CACHE / "labels.npy"), allow_pickle=True).astype(str)
y = np.array([EM.index(e) for e in lab[held]])
Sm = np.zeros((len(held), 6)); Sa = np.zeros((len(held), 6))
for s in range(0, len(held), 500):
    A = np.asarray(X[held[s:s + 500]], dtype=np.float32)
    Mk = np.stack([topk_mask(A[:, li, :], 5.0) for li in range(9)], 1).astype(np.float32)
    Sm[s:s + 500] = np.einsum("nlf,lfe->ne", Mk, Wn); Sa[s:s + 500] = np.einsum("nlf,lfe->ne", (np.abs(A) - MU) / SD, Wn)
PAIRS = [("ANGER", "DISGUST"), ("FEAR", "SADNESS"), ("HAPPINESS", "SURPRISE")]
out = {}
for nm, S in (("mask", Sm), ("absz", Sa)):
    Sc = S - S.mean(0)
    out[nm] = {"argmax_acc": float((Sc.argmax(1) == y).mean()),
               "one_vs_rest": {e: float(roc_auc_score(y == i, Sc[:, i])) for i, e in enumerate(EM)},
               "pair_auc": {f"{a}-{b}": float(roc_auc_score(y[np.isin(y, [EM.index(a), EM.index(b)])] == EM.index(a),
                            (S[:, EM.index(a)] - S[:, EM.index(b)])[np.isin(y, [EM.index(a), EM.index(b)])])) for a, b in PAIRS}}
out["majority"] = float(np.bincount(y).max() / len(y))
mp = json.load(text_open(str(RESULTS / "experiments" / "measures_F1.json")))
f1 = pd.DataFrame([json.loads(l) for l in text_open(str(RESULTS / "experiments" / "F1.jsonl"))])
out["F1_pair_auc_by_condition_mask"] = {}
for (pr, cond), g in f1.groupby(["pair", "cond"]):
    a, b = pr.split("-")
    ax = np.array([mp[t]["stim_mask"][EM.index(a)] - mp[t]["stim_mask"][EM.index(b)] for t in g.trial])
    out["F1_pair_auc_by_condition_mask"][f"{pr}|{cond}"] = float(roc_auc_score(g.gold == a, ax))
json.dump(out, text_open(str(RESULTS / "diagnostics_measure_corrected.json"), "w"), indent=1); print(json.dumps(out, indent=1))
