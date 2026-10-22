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

import numpy as np
from scipy import stats
from common import FFN
Q, H_MIN, ALPHA = 0.05, 0.2, 0.05
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

