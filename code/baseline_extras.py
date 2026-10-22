"""
Supplementary baseline analyses repeated on the corrected (BH-FDR + Cohen's h) signature
sets: (i) the five human secondary-label similarity measures, (ii) portability of neuron
indices across layers, (iii) the within-layer emotion-general backbone of top-frequency
neurons, (iv) pairwise intersection sizes, (v) same-valence versus opposite-valence overlap.
Usage: python3 baseline_extras.py zh|en
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

import sys, json, re, itertools, numpy as np
from collections import Counter
from scipy.stats import spearmanr, hypergeom
from common import EM, LAYERS, FFN, load_df
from baseline_analysis_lib import topk_mask
if len(sys.argv) < 2 or sys.argv[1] not in ("zh", "en"):
    raise SystemExit("Usage: python baseline_extras.py zh|en")
LANG = sys.argv[1]
R = str(RESULTS / f"baseline_{LANG}")
S = np.load(f"{R}/signatures.npz")
B = json.load(text_open(f"{R}/baseline_results.json"))
iu = np.triu_indices(6, 1); PERMS = list(itertools.permutations(range(6)))
out = {}

def exact(Obs, ref):
    r = spearmanr(Obs[iu], ref[iu]).statistic
    null = np.array([spearmanr(Obs[np.ix_(p, p)][iu], ref[iu]).statistic for p in PERMS])
    return [float(r), float(np.mean(null >= r - 1e-12)), int(np.sum(null >= r - 1e-12))]

# (i) human secondary labels
df = load_df()
sec = df.dropna(subset=["secondary_emotions"]).copy()
sec["s"] = sec["secondary_emotions"].apply(lambda x: [w.upper() for w in re.findall(r"[A-Za-z]+", str(x))])
sec = sec[sec.s.map(len) > 0]
vocab = sorted({w for L in sec["s"] for w in L})
P = np.array([[Counter(w for L in sec[sec.primary_emotion == e]["s"] for w in L).get(v, 0) for v in vocab] for e in EM], float)
Pn = P / P.sum(1, keepdims=True); Bn = P > 0
broad = (Bn.sum(0) >= 3)
M = {}
M["cosine"] = (P @ P.T) / np.outer(np.linalg.norm(P, axis=1), np.linalg.norm(P, axis=1))
M["bhattacharyya"] = np.sqrt(Pn) @ np.sqrt(Pn).T
M["shared_label_jaccard"] = np.array([[(Bn[a] & Bn[b]).sum() / max(1, (Bn[a] | Bn[b]).sum()) for b in range(6)] for a in range(6)])
M["histogram_intersection"] = np.array([[np.minimum(Pn[a], Pn[b]).sum() for b in range(6)] for a in range(6)])
Pb = P[:, broad]; Pbn = Pb / np.maximum(1, Pb.sum(1, keepdims=True))
M["histogram_intersection_broad"] = np.array([[np.minimum(Pbn[a], Pbn[b]).sum() for b in range(6)] for a in range(6)])
out["human_annotation"] = {"n_annotated": int(len(sec)), "vocab": len(vocab), "broad_labels": int(broad.sum())}
for crit in ("fdr", "unc"):
    J = np.array(B["jaccard"][crit])
    out["human_annotation"][crit] = {k: exact(J, v) for k, v in M.items()}

# (iv), (v) intersections and valence ratio on corrected sets
sets = {e: {(l, int(j)) for l in LAYERS for j in np.flatnonzero(S[f"fdr_{l}_{e}"])} for e in EM}
inter = {f"{a}-{b}": len(sets[a] & sets[b]) for a, b in itertools.combinations(EM, 2)}
out["pairwise_intersections_fdr"] = inter
J = np.array(B["jaccard"]["fdr"])
pos = {"HAPPINESS", "SURPRISE"}
same = [J[i, j] for i, j in zip(*iu) if (EM[i] in pos) == (EM[j] in pos)]
opp = [J[i, j] for i, j in zip(*iu) if (EM[i] in pos) != (EM[j] in pos)]
out["valence_ratio_fdr"] = {"same_mean": float(np.mean(same)), "opposite_mean": float(np.mean(opp)),
                            "ratio": float(np.mean(same) / max(1e-12, np.mean(opp)))}
pred = [("FEAR", "SADNESS"), ("ANGER", "DISGUST"), ("HAPPINESS", "SURPRISE")]
pv = [J[EM.index(a), EM.index(b)] for a, b in pred]
rest = [J[i, j] for i, j in zip(*iu) if (EM[i], EM[j]) not in pred and (EM[j], EM[i]) not in pred]
out["predicted_vs_rest_fdr"] = {"predicted_mean": float(np.mean(pv)), "rest_mean": float(np.mean(rest)),
                                "ratio": float(np.mean(pv) / max(1e-12, np.mean(rest)))}

# (ii) portability across layers: top-20 corrected signatures (ranked by Cohen's h) per layer and emotion
enum = __import__("pandas").read_csv(f"{R}/signature_enumeration_corrected.csv")
top = {(int(r.layer), r.emotion): set() for r in enum.itertuples()}
for (l, e), g in enum.groupby(["layer", "emotion"]):
    top[(int(l), e)] = set(g.sort_values("rank_by_h").neuron_index.head(20).astype(int))
shared, expd, zero, n = [], [], 0, 0
for e in EM:
    for l1, l2 in itertools.combinations(LAYERS, 2):
        a, b = top.get((l1, e), set()), top.get((l2, e), set())
        if not a or not b:
            continue
        k = len(a & b); shared.append(k); expd.append(len(a) * len(b) / FFN); zero += (k == 0); n += 1
out["portability_across_layers"] = {"pairs": n, "mean_shared": float(np.mean(shared)), "expected": float(np.mean(expd)),
                                    "pairs_with_zero_shared": int(zero)}

# (iii) emotion-general backbone: top-20 most frequently active neurons per emotion within a layer
X = np.load(str(CACHE / f"gate_{LANG}.npy"), mmap_mode="r")
labels = df["primary_emotion"].values
bb = []
for li, l in enumerate(LAYERS):
    Mk = topk_mask(np.asarray(X[:, li, :], dtype=np.float32), 5.0)
    tops = {e: set(np.argsort(-Mk[labels == e].mean(0))[:20]) for e in EM}
    bb += [len(tops[a] & tops[b]) for a, b in itertools.combinations(EM, 2)]
    del Mk
out["backbone_top20_mean_overlap_across_emotion_pairs"] = float(np.mean(bb))
json.dump(out, text_open(f"{R}/baseline_extras.json", "w"), indent=1)
print(json.dumps(out, indent=1))
