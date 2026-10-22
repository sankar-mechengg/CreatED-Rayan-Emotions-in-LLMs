"""Exploratory sensitivity analysis of H3.4 (not preregistered): target margin predicting classifier
correctness with the requested target emotion as a fixed effect (within-target relation)."""

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
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]
CM = {"anger": "ANGER", "disgust": "DISGUST", "fear": "FEAR", "joy": "HAPPINESS", "sadness": "SADNESS", "surprise": "SURPRISE", "neutral": "NEUTRAL"}
mp = json.load(text_open(str(RESULTS / "experiments" / "measures_F3.json"))); out = {}
for M in ("mask", "absz"):
    f = pd.DataFrame([json.loads(l) for l in text_open(str(RESULTS / "experiments" / "F3.jsonl"))])
    S = np.array([mp[t]["gen_" + M] for t in f.trial]); g = f["item"] + "|" + f.para.astype(str)
    Sc = S - pd.DataFrame(S).groupby(g.values).transform("mean").values
    ti = f.target.map(EM.index).values; t = Sc[np.arange(len(f)), ti]
    f["margin"] = t - (Sc.sum(1) - t) / 5
    f["mw"] = f.margin - f.groupby("target").margin.transform("mean"); f["zm"] = (f.mw - f.mw.mean()) / f.mw.std()
    f["correct"] = [int(CM[max(d, key=d.get)] == tg) for d, tg in zip(f.clf, f.target)]; f["para"] = f.para.astype(str)
    r = BinomialBayesMixedGLM.from_formula("correct ~ zm + C(target)", {"item": "0+C(item)", "para": "0+C(para)"}, f.reset_index(drop=True)).fit_vb()
    i = list(r.model.exog_names).index("zm")
    out[M] = {"beta": float(r.fe_mean[i]), "z": float(r.fe_mean[i] / r.fe_sd[i]), "p_one_sided": float(stats.norm.sf(r.fe_mean[i] / r.fe_sd[i]))}
json.dump(out, text_open(str(RESULTS / "sensitivity_h34.json"), "w"), indent=1); print(out)
