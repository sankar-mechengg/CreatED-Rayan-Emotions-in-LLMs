"""Validates lmm_crossed.fit against statsmodels MixedLM (crossed variance components) on synthetic data."""

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

import numpy as np, pandas as pd, statsmodels.api as sm, time, json
import lmm_crossed as L
rng = np.random.default_rng(1); rows = []
ie = rng.normal(0, 0.8, 60); pe = rng.normal(0, 0.4, 6)
for i in range(60):
    for p in range(6):
        for c in (0, 1):
            rows.append({"item": f"i{i}", "para": f"p{p}", "D": c, "pair": f"g{i % 3}",
                         "y": 0.3 * c + ie[i] + pe[p] + rng.normal(0, 1)})
d = pd.DataFrame(rows); d["one"] = 1
t = time.time()
r = sm.MixedLM.from_formula("y ~ D + C(pair)", groups="one", vc_formula={"item": "0+C(item)", "para": "0+C(para)"},
                            re_formula="0", data=d).fit(reml=True)
t1 = time.time() - t; t = time.time()
m = L.lmm("y ~ D + C(pair)", d, "D"); t2 = time.time() - t
out = {"statsmodels": {"beta": float(r.fe_params["D"]), "se": float(r.bse_fe["D"]), "vcomp": [float(v) for v in r.vcomp],
                       "scale": float(r.scale), "seconds": t1},
       "lmm_crossed": {"beta": m["beta"], "se": m["se"], "vc": m["vc"], "resid": m["resid_var"], "seconds": t2}}
print(json.dumps(out, indent=1)); json.dump(out, text_open(str(RESULTS / "validate_lmm.json"), "w"), indent=1)
