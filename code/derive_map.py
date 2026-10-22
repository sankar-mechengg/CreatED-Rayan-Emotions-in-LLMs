"""
Builds the reference emotion map used by the reconfiguration experiments.
Re-running overwrites reference_map/reference_map_en.npz, stimuli/f1_items.csv
and results/reference_map_jaccard_en.json. Those three files are the frozen
release; keep copies before a re-run.
English baseline activations, stratified 70/30 derivation/held-out split
(seed 42). Signatures (BH-FDR q<0.05, Cohen's h>=0.2) are derived on the
derivation split only. Per-neuron mean and SD (derivation split, raw text)
define the z-scale. Held-out items for Family 1 are sampled here.
"""

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

import json, numpy as np, pandas as pd, sys
from common import EM, LAYERS, FFN, load_df
from baseline_analysis_lib import topk_mask, signatures
from sklearn.model_selection import train_test_split

X = np.load(str(CACHE / "gate_en.npy"), mmap_mode="r")
labels = np.load(str(CACHE / "labels.npy"), allow_pickle=True).astype(str); N = len(labels)
idx = np.arange(N)
der, held = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
der.sort(); held.sort()
Yd = np.stack([(labels[der] == e) for e in EM], 1).astype(np.float32); nd = Yd.sum(0)
mu = np.zeros((len(LAYERS), FFN), np.float32); sd = np.zeros_like(mu)
W = np.zeros((len(LAYERS), FFN, 6), np.float32)
sig = {}
for li, l in enumerate(LAYERS):
    A = X[der, li, :].astype(np.float32)
    mu[li] = A.mean(0); sd[li] = A.std(0) + 1e-6
    Mf = topk_mask(A, 5.0).astype(np.float32); act = Yd.T @ Mf; tot = Mf.sum(0)
    for i, e in enumerate(EM):
        sig[(l, e)] = signatures(act[i], nd[i], tot - act[i], len(der) - nd[i], True)
sizes = {e: int(sum(sig[(l, e)].sum() for l in LAYERS)) for e in EM}
for li, l in enumerate(LAYERS):
    for i, e in enumerate(EM):
        W[li, sig[(l, e)], i] = 1.0 / sizes[e]
np.savez_compressed(str(REFERENCE / "reference_map_en.npz"), mu=mu, sd=sd, W=W, der=der, held=held,
                    sizes=np.array([sizes[e] for e in EM]))
print("signature sizes (derivation, en):", sizes)

# Family 1 held-out items: 80 per member of each high-overlap pair
df = load_df()
assert (df["primary_emotion"].values == labels).all()
df["n_words"] = df["english_text"].astype(str).str.split().str.len()
rng = np.random.default_rng(42); rows = []
for a, b in [("ANGER", "DISGUST"), ("FEAR", "SADNESS"), ("HAPPINESS", "SURPRISE")]:
    for e in (a, b):
        pool = [i for i in held if labels[i] == e and 6 <= df.loc[i, "n_words"] <= 60]
        pick = rng.choice(pool, 80, replace=False)
        rows += [{"item": f"F1_{e[:3]}_{k:02d}", "row": int(i), "pair": f"{a}-{b}", "gold": e,
                  "text": str(df.loc[i, "english_text"]).strip().strip("[]").strip()} for k, i in enumerate(pick)]
pd.DataFrame(rows).to_csv(str(STIMULI / "f1_items.csv"), index=False)
print("F1 items:", len(rows), "held-out size:", len(held), "derivation:", len(der))

# derivation-split overlap matrix (Jaccard over aggregated (layer, neuron) sets): reference for H3.2/H3.3
sets = {e: set((l, int(j)) for l in LAYERS for j in np.flatnonzero(sig[(l, e)])) for e in EM}
J = [[(len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))) for b in EM] for a in EM]
json.dump({"emotions": EM, "jaccard": J, "sizes": sizes, "split": "English derivation 70%, seed 42"},
          text_open(str(RESULTS / "reference_map_jaccard_en.json"), "w"), indent=1)
print("derivation Jaccard:\n", np.round(np.array(J), 3))
