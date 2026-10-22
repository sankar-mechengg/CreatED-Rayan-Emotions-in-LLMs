"""
Corrected baseline analysis (R4).
Signature selection: one-sided two-proportion z-test (as in the original
pipeline), Benjamini-Hochberg FDR q < 0.05 within each (layer, emotion)
family of 8,960 tests, plus Cohen's h >= 0.2 and p_target > p_other.
All signature-dependent analyses are rerun. Uncorrected results are kept
alongside for the supplementary comparison table.
Usage: python3 baseline_analysis.py zh|en
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

import sys, json, itertools, re, time
import numpy as np, pandas as pd
from collections import Counter
from scipy import stats
from scipy.stats import spearmanr, rankdata
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import squareform
from common import EM, LAYERS, FFN, load_df

if len(sys.argv) < 2 or sys.argv[1] not in ("zh", "en"):
    raise SystemExit("Usage: python baseline_analysis.py zh|en")
LANG = sys.argv[1]
OUT = str(RESULTS / f"baseline_{LANG}")
import os; os.makedirs(OUT, exist_ok=True)
Q, H_MIN, ALPHA = 0.05, 0.2, 0.05
RNG = np.random.default_rng(42)
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

X = np.load(str(CACHE / f"gate_{LANG}.npy"), mmap_mode="r")
labels = np.load(str(CACHE / "labels.npy"), allow_pickle=True).astype(str)
N = len(labels)
Y = np.stack([(labels == e) for e in EM], 1).astype(np.float32)  # N x 6
n_e = Y.sum(0)

def topk_mask(A, pct):
    k = max(1, int(FFN * pct / 100.0))
    a = np.abs(A)
    thr = np.partition(a, -k, axis=1)[:, -k]
    return a >= thr[:, None]

def bh(p):
    """Benjamini-Hochberg adjusted p-values (vectorised over last axis)."""
    p = np.asarray(p); m = p.shape[-1]
    o = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, o, -1)
    adj = ps * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[..., ::-1], axis=-1)[..., ::-1]
    out = np.empty_like(adj); np.put_along_axis(out, o, np.minimum(adj, 1), -1)
    return out

def ztest(act_t, n_t, act_o, n_o):
    p_t, p_o = act_t / n_t, act_o / n_o
    pp = (act_t + act_o) / (n_t + n_o)
    se = np.sqrt(pp * (1 - pp) * (1 / n_t + 1 / n_o)); se = np.where(se > 0, se, 1.0)
    z = (p_t - p_o) / se
    return p_t, p_o, stats.norm.sf(z)

def cohen_h(p_t, p_o):
    return 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_o))

def signatures(act_t, n_t, act_o, n_o, corrected=True):
    p_t, p_o, p = ztest(act_t, n_t, act_o, n_o)
    if not corrected:
        return (p < ALPHA) & (p_t > p_o)
    return (bh(p) < Q) & (cohen_h(p_t, p_o) >= H_MIN) & (p_t > p_o)

# ---------------------------------------------------------------- masks
log("building top-5% masks")
M = {}
for li, l in enumerate(LAYERS):
    M[l] = topk_mask(X[:, li, :].astype(np.float32), 5.0)
log("masks done")

R = {"lang": LANG, "N": int(N), "n_per_emotion": dict(zip(EM, n_e.astype(int).tolist())),
     "criteria": {"q": Q, "cohen_h_min": H_MIN, "alpha_uncorrected": ALPHA}}

# ---------------------------------------------------------------- RQ1
cons, chi_unc, chi_fdr = {}, {}, {}
for l in LAYERS:
    Mf = M[l].astype(np.float32)
    act = Y.T @ Mf                               # 6 x FFN active counts
    freq = act / n_e[:, None]
    cons[l] = dict(zip(EM, (freq > 0.5).sum(1).astype(int).tolist()))
    tot = Mf.sum(0)
    chi_unc[l], chi_fdr[l] = {}, {}
    for ei, e in enumerate(EM):
        a = act[ei]; b = tot - a; c = n_e[ei] - a; d = (N - n_e[ei]) - b
        num = np.maximum(np.abs(a * d - b * c) - N / 2, 0) ** 2 * N
        den = (a + b) * (c + d) * (a + c) * (b + d)
        chi = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
        p = stats.chi2.sf(chi, 1)
        chi_unc[l][e] = int((p < ALPHA).sum())
        chi_fdr[l][e] = int((bh(p) < Q).sum())
R["consistent_counts"] = cons; R["chi2_uncorrected"] = chi_unc; R["chi2_fdr"] = chi_fdr
log("RQ1 counts done")

# permutation test, layer 18 (unchanged method; 1000 label shuffles)
Mf = M[18].astype(np.float32)
obs = ((Y.T @ Mf) / n_e[:, None] > 0.5).sum(1)
null = np.zeros((1000, 6))
for b in range(1000):
    Yp = Y[RNG.permutation(N)]
    null[b] = ((Yp.T @ Mf) / n_e[:, None] > 0.5).sum(1)
perm_p = ((null >= obs[None]).sum(0) + 1) / 1001
R["permutation_L18"] = {e: {"observed": int(obs[i]), "null_mean": float(null[:, i].mean()),
                            "null_sd": float(null[:, i].std()), "p": float(perm_p[i])}
                        for i, e in enumerate(EM)}
log("permutation done")

# ---------------------------------------------------------------- signatures (full data)
SIG = {"fdr": {}, "unc": {}}
for l in LAYERS:
    Mf = M[l].astype(np.float32); act = Y.T @ Mf; tot = Mf.sum(0)
    for ei, e in enumerate(EM):
        for kind in SIG:
            SIG[kind][(l, e)] = signatures(act[ei], n_e[ei], tot - act[ei], N - n_e[ei], kind == "fdr")
R["signature_counts"] = {kind: {e: {l: int(SIG[kind][(l, e)].sum()) for l in LAYERS} for e in EM} for kind in SIG}
R["signature_totals"] = {kind: {e: int(sum(SIG[kind][(l, e)].sum() for l in LAYERS)) for e in EM} for kind in SIG}
log("signatures:", R["signature_totals"])

# save top-20 per layer x emotion (corrected) for the enumeration table
rows = []
for l in LAYERS:
    Mf = M[l].astype(np.float32); freq = (Y.T @ Mf) / n_e[:, None]
    absmean = None
    for ei, e in enumerate(EM):
        idx = np.where(SIG["fdr"][(l, e)])[0]
        if len(idx) == 0: continue
        h = cohen_h(freq[ei, idx], (Mf[:, idx].sum(0) - freq[ei, idx] * n_e[ei]) / (N - n_e[ei]))
        top = idx[np.argsort(-h)][:20]
        for r, n in enumerate(top, 1):
            rows.append((e, l, int(n), float(freq[ei, n]), r))
pd.DataFrame(rows, columns=["emotion", "layer", "neuron_index", "activation_freq", "rank_by_h"]).to_csv(f"{OUT}/signature_enumeration_corrected.csv", index=False)

# ---------------------------------------------------------------- split-half (layer 18)
def split_half(kind, reps=100):
    rng = np.random.default_rng(42); out = {e: [] for e in EM}
    Mf = M[18].astype(np.float32)
    for _ in range(reps):
        for ei, e in enumerate(EM):
            t = np.where(labels == e)[0].copy(); rng.shuffle(t)
            ha, hb = t[: len(t) // 2], t[len(t) // 2:]
            o_act = Mf[labels != e].sum(0); n_o = (labels != e).sum()
            sa = signatures(Mf[ha].sum(0), len(ha), o_act, n_o, kind == "fdr")
            sb = signatures(Mf[hb].sum(0), len(hb), o_act, n_o, kind == "fdr")
            u = (sa | sb).sum()
            out[e].append(float((sa & sb).sum() / u) if u else 0.0)
    return {e: [float(np.mean(v)), float(np.std(v))] for e, v in out.items()}
R["split_half_L18"] = {"fdr": split_half("fdr"), "unc": split_half("unc")}
log("split-half done")

# ---------------------------------------------------------------- overlap structure
def agg_sets(sig):
    return {e: {(l, int(n)) for l in LAYERS for n in np.where(sig[(l, e)])[0]} for e in EM}
def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0
def jmat(sets):
    J = np.eye(6)
    for i, j in itertools.combinations(range(6), 2):
        J[i, j] = J[j, i] = jac(sets[EM[i]], sets[EM[j]])
    return J
iu = np.triu_indices(6, 1)
SETS = {k: agg_sets(SIG[k]) for k in SIG}
J = {k: jmat(SETS[k]) for k in SIG}
R["jaccard"] = {k: J[k].tolist() for k in J}

def excl_shared(sets):
    out = {}
    for e in EM:
        others = set().union(*[sets[o] for o in EM if o != e])
        out[e] = {"exclusive": len(sets[e] - others), "shared": len(sets[e] & others), "total": len(sets[e])}
    return out
R["exclusive_shared"] = {k: excl_shared(SETS[k]) for k in SETS}

# per-layer Jaccard for the three predicted pairs
pl = {}
for l in LAYERS:
    s = {e: set(np.where(SIG["fdr"][(l, e)])[0]) for e in EM}
    pl[l] = {f"{a}-{b}": jac(s[a], s[b]) for a, b in [("FEAR", "SADNESS"), ("ANGER", "DISGUST"), ("HAPPINESS", "SURPRISE")]}
R["per_layer_pairs_fdr"] = pl

# circumplex model matrices
quad = {"HAPPINESS": "P", "SURPRISE": "P", "ANGER": "NH", "DISGUST": "NH", "FEAR": "NL", "SADNESS": "NL"}
val = {"HAPPINESS": 1, "SURPRISE": 1, "ANGER": 0, "DISGUST": 0, "FEAR": 0, "SADNESS": 0}
Mg = np.zeros((6, 6)); Mb = np.zeros((6, 6))
for i, a in enumerate(EM):
    for j, b in enumerate(EM):
        if i != j:
            Mg[i, j] = 2.0 if quad[a] == quad[b] else (1.0 if val[a] == val[b] else 0.0)
            Mb[i, j] = 1.0 if quad[a] == quad[b] else 0.0
PERMS = list(itertools.permutations(range(6)))
def exact(Obs, ref):
    r = spearmanr(Obs[iu], ref[iu]).statistic
    c = sum(spearmanr(Obs[np.ix_(p, p)][iu], ref[iu]).statistic >= r - 1e-12 for p in PERMS)
    return float(r), c / 720, int(c)
def partial(x, y, z):
    x, y, z = map(rankdata, (x, y, z))
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

# lexical similarity matrix (English text, independent of model)
from sklearn.feature_extraction.text import TfidfVectorizer
df = load_df()
docs = [" ".join(df.loc[df.primary_emotion == e, "english_text"].astype(str)) for e in EM]
Xt = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit_transform(docs).toarray()
Xt /= np.linalg.norm(Xt, axis=1, keepdims=True); LEX = Xt @ Xt.T

circ = {}
for k in J:
    Jk = J[k]; order = np.argsort(Jk[iu])[::-1]
    adj = np.array([quad[EM[i]] == quad[EM[j]] for i, j in zip(*iu)])
    pr = partial(Jk[iu], Mg[iu], LEX[iu])
    cpr = sum(partial(Jk[np.ix_(p, p)][iu], Mg[iu], LEX[np.ix_(p, p)][iu]) >= pr - 1e-12 for p in PERMS)
    D = 1 - Jk; np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    coph = cophenet(Z, squareform(D, checks=False))[0]
    slack = [D[a, b] + D[b, c] - D[a, c] for i, j, kk in itertools.combinations(range(6), 3)
             for a, b, c in [(i, j, kk), (j, kk, i), (kk, i, j)]]
    circ[k] = {"graded": exact(Jk, Mg), "binary": exact(Jk, Mb),
               "top3_are_predicted": bool(set(order[:3]) == set(np.where(adj)[0])),
               "top3_pairs": [f"{EM[iu[0][o]]}-{EM[iu[1][o]]}" for o in order[:3]],
               "mean_within": float(Jk[iu][adj].mean()), "mean_between": float(Jk[iu][~adj].mean()),
               "lexical_vs_circ": exact(LEX, Mg), "neuron_vs_lex": float(spearmanr(Jk[iu], LEX[iu]).statistic),
               "partial_given_lex": [float(pr), cpr / 720, int(cpr)],
               "cophenetic": float(coph), "dendrogram_merge_order": Z[:, :2].astype(int).tolist(),
               "triangle_min_slack": float(np.min(slack)), "triangle_violations": int(np.sum(np.array(slack) < 0))}
R["circumplex"] = circ
log("circumplex:", {k: circ[k]["graded"] for k in circ})

# human secondary-label test (independent annotations)
sec = df.dropna(subset=["secondary_emotions"]).copy()
sec["s"] = sec["secondary_emotions"].apply(lambda x: [w.upper() for w in re.findall(r"[A-Za-z]+", str(x))])
vocab = sorted({w for L in sec["s"] for w in L})
Pm = np.array([[Counter(w for L in sec[sec.primary_emotion == e]["s"] for w in L).get(v, 0) for v in vocab] for e in EM], float)
Bn = (Pm > 0)
Js = np.array([[(Bn[i] & Bn[j]).sum() / max((Bn[i] | Bn[j]).sum(), 1) for j in range(6)] for i in range(6)])
R["human_annotation"] = {k: exact(J[k], Js) for k in J}

json.dump(R, text_open(f"{OUT}/baseline_results.json", "w"), indent=1, default=str)
np.savez_compressed(f"{OUT}/signatures.npz", **{f"{k}_{l}_{e}": SIG[k][(l, e)] for k in SIG for l in LAYERS for e in EM})
log("saved core")

# ---------------------------------------------------------------- sensitivity
sens = {}
for pct in [1.0, 3.0, 5.0, 10.0]:
    cc, sc, sets = {}, {}, {e: set() for e in EM}
    for li, l in enumerate(LAYERS):
        Mm = M[l] if pct == 5.0 else topk_mask(X[:, li, :].astype(np.float32), pct)
        Mf = Mm.astype(np.float32); act = Y.T @ Mf; tot = Mf.sum(0)
        cc[l] = ((act / n_e[:, None]) > 0.5).sum(1).tolist()
        for ei, e in enumerate(EM):
            s = signatures(act[ei], n_e[ei], tot - act[ei], N - n_e[ei], True)
            sets[e] |= {(l, int(n)) for n in np.where(s)[0]}
    Jp = jmat(sets)
    sens[pct] = {"consistent_total": dict(zip(EM, np.sum([cc[l] for l in LAYERS], 0).tolist())),
                 "signature_total": {e: len(sets[e]) for e in EM},
                 "jaccard": Jp.tolist(), "rho_vs_5pct": float(spearmanr(Jp[iu], J["fdr"][iu]).statistic),
                 "graded_circumplex": exact(Jp, Mg)}
    log("sensitivity", pct, sens[pct]["graded_circumplex"])
R["sensitivity"] = sens
json.dump(R, text_open(f"{OUT}/baseline_results.json", "w"), indent=1, default=str)
log("DONE")
