"""Figures for the reconfiguration experiments (corrected measure unless stated)."""

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

import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
apply_figure_style()
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]
D = str(RESULTS / "experiments"); FIG = str(FIGURES)
M = "mask"
mp = {t: json.load(text_open(f"{D}/measures_{t}.json")) for t in ("F1", "F2", "F3", "F4")}
load = lambda t: pd.DataFrame([json.loads(l) for l in text_open(f"{D}/{t}.jsonl")])
PAIRS = ["ANGER-DISGUST", "FEAR-SADNESS", "HAPPINESS-SURPRISE"]
PL = {"ANGER-DISGUST": "Anger vs disgust", "FEAR-SADNESS": "Fear vs sadness", "HAPPINESS-SURPRISE": "Happiness vs surprise"}

def ci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))

# E0 validity of measures on held-out texts
dg = json.load(text_open(str(RESULTS / "diagnostics_measure.json")))
dc = json.load(text_open(str(RESULTS / "diagnostics_measure_corrected.json")))
fig, ax = plt.subplots(figsize=(6.2, 3)); x = np.arange(6); w = 0.27
ax.bar(x - w, [dg["heldout_one_vs_rest_auc"][e] for e in EM], w, color="#bbbbbb", label="Preregistered (signed z)")
ax.bar(x, [dc["mask"]["one_vs_rest"][e] for e in EM], w, color="#2c3e50", label="Corrected (mask fraction)")
ax.bar(x + w, [dc["absz"]["one_vs_rest"][e] for e in EM], w, color="#7f8c8d", label="Robustness (absolute z)")
ax.axhline(0.5, ls=":", color="k"); ax.set_xticks(x); ax.set_xticklabels([e.capitalize() for e in EM]); ax.set_ylim(0.3, 0.9)
ax.set_ylabel("One-vs-rest AUC, held-out texts"); ax.legend(frameon=False, fontsize=8, loc="upper left"); plt.savefig(f"{FIG}/E0_measure_validity.png"); plt.close()

# E1 Family 1
f1 = load("F1"); f1["stim"] = f1.trial.map(lambda t: mp["F1"][t]["stim_" + M])
def signed(r):
    a, b = r.pair.split("-"); A = r.stim[EM.index(a)] - r.stim[EM.index(b)]; return A if r.gold == a else -A
def beh(r):
    a, b = r.pair.split("-"); o = b if r.gold == a else a; return r.lp[EM.index(r.gold)] - r.lp[EM.index(o)]
f1["signed"] = f1.apply(signed, axis=1); f1["beh"] = f1.apply(beh, axis=1)
u = f1.pivot_table(index=["item", "para", "pair"], columns="cond", values=["signed", "beh"]).reset_index()
u2 = pd.DataFrame({"item": u["item"].values.ravel(), "pair": u["pair"].values.ravel(),
                   "dS": (u[("signed", "DISCRIMINATE")] - u[("signed", "RECOGNISE")]).values,
                   "dB": (u[("beh", "DISCRIMINATE")] - u[("beh", "RECOGNISE")]).values})
it = u2.groupby(["item", "pair"])[["dS", "dB"]].mean().reset_index()
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
for k, pr in enumerate(PAIRS):
    g = it[it.pair == pr]; m, h = ci(g.dS); axs[0].errorbar(k, m, yerr=h, fmt="o", color="#2c3e50", capsize=4)
    axs[1].scatter(g.dS, g.dB, s=8, alpha=0.5, label=PL[pr])
axs[0].axhline(0, ls=":", color="k"); axs[0].set_xticks(range(3)); axs[0].set_xticklabels([PL[p] for p in PAIRS], fontsize=8)
axs[0].set_ylabel("Change in internal separation\n(discriminate minus recognise)"); axs[0].set_xlim(-0.5, 2.5)
axs[1].axhline(0, ls=":", color="k"); axs[1].axvline(0, ls=":", color="k")
axs[1].set_xlabel("Change in internal separation (item mean)"); axs[1].set_ylabel("Change in behavioural margin (log-prob)")
axs[1].legend(frameon=False, fontsize=7); plt.savefig(f"{FIG}/E1_family1.png"); plt.close()

# E2 Family 2
f2 = load("F2"); f2["final"] = f2.trial.map(lambda t: mp["F2"][t]["final_" + M])
f2["A"] = [r[EM.index(a)] - r[EM.index(b)] for r, a, b in zip(f2.final, f2.a, f2.b)]
f2["Beh"] = [l[EM.index(a)] - l[EM.index(b)] for l, a, b in zip(f2.lp, f2.a, f2.b)]
conds = ["EV_B", "NONE", "FILLER", "EV_A"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
for pr in PAIRS:
    g = f2[f2.pair == pr]; base = g[g.cond == "NONE"].groupby("item").A.mean()
    ms = [ci(g[g.cond == c].groupby("item").A.mean() - base) for c in conds]
    axs[0].errorbar(range(4), [m for m, _ in ms], yerr=[h for _, h in ms], fmt="o-", capsize=3, label=PL[pr])
    bs = [ci(g[g.cond == c].groupby("item").Beh.mean()) for c in conds]
    axs[1].errorbar(range(4), [m for m, _ in bs], yerr=[h for _, h in bs], fmt="o-", capsize=3, label=PL[pr])
for ax, t in zip(axs, ("Internal pair axis (a minus b), relative to NONE", "Behavioural margin, log p(a) minus log p(b)")):
    ax.set_xticks(range(4)); ax.set_xticklabels(["Evidence\nfor b", "None", "Filler", "Evidence\nfor a"]); ax.axhline(0, ls=":", color="k"); ax.set_title(t, fontsize=9)
axs[0].legend(frameon=False, fontsize=7); plt.savefig(f"{FIG}/E2_family2.png"); plt.close()

# E3 Family 3
R = json.load(text_open(str(RESULTS / "confirmatory_mask_results.json")))
Jref = np.array(json.load(text_open(str(RESULTS / "reference_map_jaccard_en.json")))["jaccard"]); iu = np.triu_indices(6, 1)
f3 = load("F3"); f3["gen"] = f3.trial.map(lambda t: mp["F3"][t]["gen_" + M])
S = np.array(f3.gen.tolist()); g = f3["item"] + "|" + f3.para.astype(str)
Sc = S - pd.DataFrame(S).groupby(g.values).transform("mean").values
ti = f3.target.map(EM.index).values; f3["margin"] = Sc[np.arange(len(f3)), ti] - (Sc.sum(1) - Sc[np.arange(len(f3)), ti]) / 5
fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
for k, e in enumerate(EM):
    m, h = ci(f3[f3.target == e].groupby("item").margin.mean()); axs[0].errorbar(k, m, yerr=h, fmt="o", color="#2c3e50", capsize=3)
axs[0].axhline(0, ls=":", color="k"); axs[0].set_xticks(range(6)); axs[0].set_xticklabels([e[:3].capitalize() for e in EM])
axs[0].set_ylabel("Target margin over generated tokens"); axs[0].set_title("Internal: target vs other five", fontsize=9)
for ax, key, t in ((axs[1], ("F3", "H3.2", "internal_separation"), "Internal separation"),
                   (axs[2], ("F3", "H3.3", "behavioural_separation"), "Classifier separation")):
    Sep = np.array(R[key[0]][key[1]][key[2]]); ax.scatter(Jref[iu], Sep[iu], s=18, color="#2c3e50")
    for i, j in zip(*iu):
        if Jref[i, j] > 0.015:
            ax.annotate(f"{EM[i][:3].capitalize()}-{EM[j][:3].capitalize()}", (Jref[i, j], Sep[i, j]), fontsize=7)
    ax.set_xlabel("Reference overlap (Jaccard)"); ax.set_title(f"{t}: rho = {R[key[0]][key[1]]['rho']:.2f}, p = {R[key[0]][key[1]]['p_one_sided_exact']:.3f}", fontsize=9)
plt.savefig(f"{FIG}/E3_family3.png"); plt.close()

# E4 Family 4
f4 = load("F4"); f4["gen"] = f4.trial.map(lambda t: mp["F4"][t]["gen_" + M]); f4["plan"] = f4.trial.map(lambda t: mp["F4"][t]["prompt_last_" + M])
f4["gen_t"] = [g[EM.index(t)] for g, t in zip(f4.gen, f4.target)]; f4["plan_t"] = [g[EM.index(t)] for g, t in zip(f4.plan, f4.target)]
f4["Pt"] = [d[t.lower()] for d, t in zip(f4.clf, f4.target)]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.1)); cs = ["EXPRESS", "SUPPRESS", "NEUTRAL"]
for ax, col, t in zip(axs, ("gen_t", "plan_t", "Pt"), ("Internal, generated text", "Internal, planning state", "Classifier probability of target")):
    for k, tg in enumerate(("ANGER", "FEAR")):
        ms = [ci(f4[(f4.target == tg) & (f4.cond == c)].groupby("item")[col].mean()) for c in cs]
        ax.errorbar(np.arange(3) + (k - 0.5) * 0.15, [m for m, _ in ms], yerr=[h for _, h in ms], fmt="o-", capsize=3, label=tg.capitalize())
    ax.set_xticks(range(3)); ax.set_xticklabels([c.capitalize() for c in cs]); ax.set_title(t, fontsize=9)
axs[0].legend(frameon=False, fontsize=8); plt.savefig(f"{FIG}/E4_family4.png"); plt.close()
print("experiment figures done")
