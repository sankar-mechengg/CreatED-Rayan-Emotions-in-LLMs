"""Exploratory sensitivity analysis of H1.4 (not preregistered): does the relation between the change in
internal separation and the change in behavioural margin hold within each pair x gold-label cell, i.e. after
removing any common shift of the whole condition towards one label?"""

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
from scipy.stats import spearmanr
import lmm_crossed
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]
out = {}
for M in ("mask", "absz"):
    mp = json.load(text_open(str(RESULTS / "experiments" / "measures_F1.json")))
    f = pd.DataFrame([json.loads(l) for l in text_open(str(RESULTS / "experiments" / "F1.jsonl"))])
    f["stim"] = f.trial.map(lambda t: mp[t]["stim_" + M])
    def sg(r):
        a, b = r.pair.split("-"); A = r.stim[EM.index(a)] - r.stim[EM.index(b)]; return A if r.gold == a else -A
    def bh(r):
        a, b = r.pair.split("-"); o = b if r.gold == a else a; return r.lp[EM.index(r.gold)] - r.lp[EM.index(o)]
    f["signed"] = f.apply(sg, axis=1); f["beh"] = f.apply(bh, axis=1)
    u = f.pivot_table(index=["item", "para", "pair", "gold"], columns="cond", values=["signed", "beh"]).reset_index()
    ch = pd.DataFrame({"item": u["item"].values.ravel(), "para": u["para"].values.ravel().astype(str), "pair": u["pair"].values.ravel(),
                       "gold": u["gold"].values.ravel(), "dS": (u[("signed", "DISCRIMINATE")] - u[("signed", "RECOGNISE")]).values,
                       "dB": (u[("beh", "DISCRIMINATE")] - u[("beh", "RECOGNISE")]).values})
    ch["cell"] = ch.pair + "|" + ch.gold
    ch["dSw"] = ch.dS - ch.groupby("cell").dS.transform("mean")
    ch["zdS"] = (ch.dSw - ch.dSw.mean()) / ch.dSw.std()
    r = lmm_crossed.lmm("dB ~ zdS + C(cell)", ch, "zdS")
    it = ch.groupby(["item", "cell"])[["dS", "dB"]].mean().reset_index()
    cells = {c: float(spearmanr(g.dS, g.dB).statistic) for c, g in it.groupby("cell")}
    out[M] = {"H1.4_within_cell": r, "item_spearman_within_cell": cells,
              "mean_dS_by_cell": ch.groupby("cell").dS.mean().to_dict(), "mean_dB_by_cell": ch.groupby("cell").dB.mean().to_dict()}
json.dump(out, text_open(str(RESULTS / "sensitivity_h14.json"), "w"), indent=1); print(json.dumps(out, indent=1))

# H1.3 within cells: does internal separation predict correctness after removing pair x gold means?
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from scipy import stats
for M in ("mask", "absz"):
    mp = json.load(text_open(str(RESULTS / "experiments" / "measures_F1.json")))
    f = pd.DataFrame([json.loads(l) for l in text_open(str(RESULTS / "experiments" / "F1.jsonl"))])
    f = f[f.cond == "DISCRIMINATE"].copy(); f["stim"] = f.trial.map(lambda t: mp[t]["stim_" + M])
    f["signed"] = f.apply(sg, axis=1); f["beh"] = f.apply(bh, axis=1); f["correct"] = (f.beh > 0).astype(int)
    f["cell"] = f.pair + "|" + f.gold; f["para"] = f.para.astype(str)
    f["w"] = f.signed - f.groupby("cell").signed.transform("mean"); f["zw"] = (f.w - f.w.mean()) / f.w.std()
    r = BinomialBayesMixedGLM.from_formula("correct ~ zw + C(cell)", {"item": "0+C(item)", "para": "0+C(para)"}, f.reset_index(drop=True)).fit_vb()
    i = list(r.model.exog_names).index("zw")
    out[M]["H1.3_within_cell"] = {"beta": float(r.fe_mean[i]), "z": float(r.fe_mean[i] / r.fe_sd[i]), "p_one_sided": float(stats.norm.sf(r.fe_mean[i] / r.fe_sd[i]))}
    out[M]["accuracy_by_cell_D"] = f.groupby("cell").correct.mean().to_dict()
json.dump(out, text_open(str(RESULTS / "sensitivity_h14.json"), "w"), indent=1)
print({M: (out[M]["H1.3_within_cell"], out[M]["accuracy_by_cell_D"]) for M in out})
