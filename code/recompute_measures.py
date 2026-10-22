"""Deviation 1: recompute internal scores from the saved pooled activations with the corrected,
signature-consistent measures (mask fraction, primary; absolute z, robustness).
Usage: python3 recompute_measures.py F1|F2|F3|F4"""

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

import sys, glob, json, os, numpy as np
from baseline_analysis_lib import topk_mask
if len(sys.argv) < 2 or sys.argv[1] not in ("F1", "F2", "F3", "F4"):
    raise SystemExit("Usage: python recompute_measures.py F1|F2|F3|F4")
tag = sys.argv[1]
ref = np.load(str(REFERENCE / "reference_map_en.npz")); W = (ref["W"] > 0).astype(np.float32); Wn = W / W.sum((0, 1), keepdims=True)
if not os.path.exists(str(REFERENCE / "reference_abs_en.npz")):
    X = np.load(str(CACHE / "gate_en.npy"), mmap_mode="r"); der = ref["der"]
    mu = np.zeros((9, 8960), np.float32); sd = np.zeros_like(mu)
    for li in range(9):
        A = np.abs(np.asarray(X[der, li, :], dtype=np.float32)); mu[li] = A.mean(0); sd[li] = A.std(0) + 1e-6
    np.savez_compressed(str(REFERENCE / "reference_abs_en.npz"), mu=mu, sd=sd)
ab = np.load(str(REFERENCE / "reference_abs_en.npz")); MU, SD = ab["mu"], ab["sd"]
KEYS = {"F1": ["stim", "final"], "F2": ["final"], "F3": ["gen", "prompt_last"], "F4": ["gen", "prompt_last"]}[tag]
out = {}
for f in sorted(glob.glob(str(RESULTS / "experiments" / f"raw_{tag}_*.npz"))):
    d = np.load(f); tr = d["trial"]
    for k in KEYS:
        A = d[k].astype(np.float32)                       # n x 9 x FFN
        assert len(A) == len(tr), (f, k)
        M = np.stack([topk_mask(A[:, li, :], 5.0) for li in range(9)], 1).astype(np.float32)
        sm = np.einsum("nlf,lfe->ne", M, Wn); sa = np.einsum("nlf,lfe->ne", (np.abs(A) - MU) / SD, Wn)
        for t, x, y in zip(tr, sm, sa):
            out.setdefault(str(t), {})[k + "_mask"] = x.tolist(); out[str(t)][k + "_absz"] = y.tolist()
json.dump(out, text_open(str(RESULTS / "experiments" / f"measures_{tag}.json"), "w"))
print(tag, len(out), "trials")
