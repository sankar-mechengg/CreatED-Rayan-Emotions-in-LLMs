"""
Linear mixed model with crossed random intercepts, fitted by REML through the penalised
least-squares formulation of Bates, Maechler, Bolker and Walker (lme4). This fits the same
model as statsmodels MixedLM with crossed variance components (groups = one cluster), but
works with the q x q system (q = number of random-effect levels) instead of the n x n
marginal covariance, which makes models with hundreds of items tractable on a CPU.
Validated against statsmodels in validate_lmm.py.
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

import numpy as np, pandas as pd, patsy
from scipy import optimize, sparse, stats
from scipy.linalg import cho_factor, cho_solve

def _design(formula, df, vcs):
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    keep = y.index
    d = df.loc[keep]
    Zs, sizes = [], []
    for v in vcs:
        codes, levels = pd.factorize(d[v].astype(str))
        Z = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(levels)))
        Zs.append(Z); sizes.append(len(levels))
    return y.values.ravel(), X, sparse.hstack(Zs).tocsr(), sizes

def _deviance(theta, y, X, Z, sizes, ZtZ, ZtX, Zty, XtX, Xty, yty, reml=True, ret=False):
    lam = np.concatenate([np.full(s, t) for s, t in zip(sizes, theta)])
    q, p = ZtZ.shape[0], XtX.shape[0]; n = len(y)
    A11 = (lam[:, None] * ZtZ * lam[None, :]) + np.eye(q)
    A12 = lam[:, None] * ZtX
    b1 = lam * Zty
    L = np.linalg.cholesky(A11)
    RZX = np.linalg.solve(L, A12)
    cu = np.linalg.solve(L, b1)
    M = XtX - RZX.T @ RZX
    RX = np.linalg.cholesky(M)
    beta = np.linalg.solve(M, Xty - RZX.T @ cu)
    u = np.linalg.solve(L.T, cu - RZX @ beta)
    r2 = yty - 2 * beta @ Xty - 2 * u @ b1 + beta @ XtX @ beta + 2 * u @ (A12 @ beta) + u @ A11 @ u
    ldL = 2 * np.log(np.diag(L)).sum(); ldRX = 2 * np.log(np.diag(RX)).sum()
    if reml:
        df = n - p
        dev = ldL + ldRX + df * (1 + np.log(2 * np.pi * r2 / df))
    else:
        df = n
        dev = ldL + n * (1 + np.log(2 * np.pi * r2 / n))
    if ret:
        s2 = r2 / df
        covb = s2 * np.linalg.inv(M)
        return dict(beta=beta, covb=covb, sigma2=s2, vc=(np.array(theta) ** 2) * s2)
    return dev

def fit(formula, df, vcs=("item", "para"), reml=True):
    y, X, Z, sizes = _design(formula, df.reset_index(drop=True), vcs)
    Xv = X.values
    ZtZ = (Z.T @ Z).toarray(); ZtX = Z.T @ Xv; Zty = Z.T @ y
    XtX = Xv.T @ Xv; Xty = Xv.T @ y; yty = y @ y
    args = (y, Xv, Z, sizes, ZtZ, ZtX, Zty, XtX, Xty, yty, reml)
    best = None
    for start in ([1.0] * len(sizes), [0.3] * len(sizes), [0.05] * len(sizes)):
        r = optimize.minimize(_deviance, start, args=args, method="L-BFGS-B", bounds=[(0, None)] * len(sizes))
        if best is None or r.fun < best.fun:
            best = r
    out = _deviance(best.x, *args, ret=True)
    out["names"] = list(X.columns); out["theta"] = best.x; out["n"] = len(y); out["converged"] = bool(best.success)
    out["deviance"] = float(best.fun)
    return out

def lmm(formula, df, term, vcs=("item", "para")):
    r = fit(formula, df, vcs)
    i = r["names"].index(term); b = r["beta"][i]; se = np.sqrt(r["covb"][i, i])
    return {"beta": float(b), "se": float(se), "z": float(b / se), "p_one_sided": float(stats.norm.sf(b / se)),
            "n": int(r["n"]), "vc": {k: float(v) for k, v in zip(vcs, r["vc"])}, "resid_var": float(r["sigma2"]),
            "converged": r["converged"]}
