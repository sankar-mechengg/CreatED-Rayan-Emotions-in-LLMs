"""
Cross-validated classifier on binary top-5% FFN masks (layers 14-22) with
signature construction nested inside each fold (R4, reviewer point on
label-dependent preprocessing). Features are label-free (per-sample top-K),
so no leakage exists in the classifier itself; the nested part is the
signature-overlap versus confusion comparison (former Section 4.7).
Usage: python3 classifier_nested.py zh|en
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

import sys, json, itertools, time
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, accuracy_score
from common import EM, LAYERS, FFN
from baseline_analysis_lib import topk_mask, signatures

if len(sys.argv) < 2 or sys.argv[1] not in ("zh", "en"):
    raise SystemExit("Usage: python classifier_nested.py zh|en")
LANG = sys.argv[1]; t0 = time.time()
X = np.load(str(CACHE / f"gate_{LANG}.npy"), mmap_mode="r")
labels = np.load(str(CACHE / "labels.npy"), allow_pickle=True).astype(str); N = len(labels)
y = np.array([EM.index(e) for e in labels])
masks = [topk_mask(X[:, li, :].astype(np.float32), 5.0) for li in range(len(LAYERS))]
F = sparse.hstack([sparse.csr_matrix(m.astype(np.float32)) for m in masks]).tocsr()
print("features", F.shape, F.nnz, f"{time.time()-t0:.0f}s", flush=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
yt, yp, fold_J, fold_acc = [], [], [], []
iu = np.triu_indices(6, 1)
for k, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
    clf = LogisticRegression(max_iter=1000, solver="saga", C=1.0, random_state=42)
    clf.fit(F[tr], y[tr]); p = clf.predict(F[te])
    yt.append(y[te]); yp.append(p); fold_acc.append(accuracy_score(y[te], p))
    # signatures from TRAINING fold only
    sets = {e: set() for e in EM}
    Ytr = np.stack([(y[tr] == i) for i in range(6)], 1).astype(np.float32); ntr = Ytr.sum(0)
    for li, l in enumerate(LAYERS):
        Mf = masks[li][tr].astype(np.float32); act = Ytr.T @ Mf; tot = Mf.sum(0)
        for i, e in enumerate(EM):
            s = signatures(act[i], ntr[i], tot - act[i], len(tr) - ntr[i], True)
            sets[e] |= {(l, int(n)) for n in np.where(s)[0]}
    Jk = np.eye(6)
    for i, j in itertools.combinations(range(6), 2):
        u = sets[EM[i]] | sets[EM[j]]
        Jk[i, j] = Jk[j, i] = len(sets[EM[i]] & sets[EM[j]]) / len(u) if u else 0
    fold_J.append(Jk)
    print(f"fold {k+1}: acc {fold_acc[-1]:.4f}  {time.time()-t0:.0f}s", flush=True)

yt = np.concatenate(yt); yp = np.concatenate(yp)
C = confusion_matrix(yt, yp, labels=range(6)).astype(float)
Cn = C / C.sum(1, keepdims=True)
n = np.bincount(yt, minlength=6); predm = (n[:, None] * Cn).sum(0) / n.sum()
L = np.log(np.maximum(Cn, 1e-6) / predm[None]); S = (L + L.T) / 2
Jm = np.mean(fold_J, 0)
PERMS = list(itertools.permutations(range(6)))
r = spearmanr(Jm[iu], S[iu]).statistic
c = sum(spearmanr(Jm[np.ix_(p, p)][iu], S[iu]).statistic >= r - 1e-12 for p in PERMS)
per_fold = [float(spearmanr(Jf[iu], S[iu]).statistic) for Jf in fold_J]
acc = accuracy_score(yt, yp); maj = n.max() / n.sum()
ci = binomtest(int((yt == yp).sum()), len(yt)).proportion_ci()
res = {"lang": LANG, "accuracy": acc, "ci95": [ci.low, ci.high], "fold_acc": fold_acc,
       "majority_baseline": maj, "uniform": 1 / 6, "prop_chance": float(((n / n.sum()) ** 2).sum()),
       "kappa": cohen_kappa_score(yt, yp), "macro_f1": f1_score(yt, yp, average="macro"),
       "per_class_f1": dict(zip(EM, f1_score(yt, yp, average=None).tolist())),
       "confusion_rownorm": Cn.tolist(),
       "nested_overlap_vs_confusion": {"rho": float(r), "p_exact": c / 720, "n": int(c), "per_fold_rho": per_fold},
       "fold_mean_jaccard": Jm.tolist(), "seconds": time.time() - t0}
json.dump(res, text_open(str(RESULTS / f"baseline_{LANG}/classifier_nested.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ["accuracy", "kappa", "macro_f1", "nested_overlap_vs_confusion"]}, indent=1))
