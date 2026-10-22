"""
Preregistered confirmatory analyses for the R4 reconfiguration experiments.
All tests are one-sided in the preregistered direction. Holm correction is
applied across the 14 confirmatory hypotheses. Usage:
    python3 analyse_experiments.py            (real data)
    python3 analyse_experiments.py --selftest (synthetic data, code check only)
"""
import sys, json, itertools, warnings
sys.path.insert(0, "/home/claude/r4")
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
warnings.filterwarnings("ignore")
EM = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]
PAIRS = [("ANGER", "DISGUST"), ("FEAR", "SADNESS"), ("HAPPINESS", "SURPRISE")]
D = "/home/claude/r4/results/experiments"
CLF_MAP = {"anger": "ANGER", "disgust": "DISGUST", "fear": "FEAR", "joy": "HAPPINESS",
           "sadness": "SADNESS", "surprise": "SURPRISE", "neutral": "NEUTRAL"}
iu = np.triu_indices(6, 1)
import re as _re
_NRC = json.load(open("/usr/local/lib/python3.13/dist-packages/nrclex/data/nrc_en.json"))
_NRC_KEYS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def nrc_scores(text):
    """Second, lexicon-based scorer (NRC Emotion Lexicon as bundled in nrclex 4.1.0): proportion of
    word tokens associated with each of the six emotions; lower-cased exact word match, no lemmatisation."""
    toks = _re.findall(r"[a-z]+", str(text).lower())
    v = np.zeros(6)
    for t in toks:
        for k in _NRC.get(t, []):
            if k in _NRC_KEYS:
                v[_NRC_KEYS.index(k)] += 1
    return v / max(1, len(toks))
PERMS = list(itertools.permutations(range(6)))

def lmm(formula, df, term, vcs=("item", "para")):
    """REML linear mixed model with crossed random intercepts (lmm_crossed; identical estimates to
    statsmodels MixedLM with crossed variance components, see validate_lmm.py, but tractable for
    hundreds of items)."""
    import lmm_crossed
    return lmm_crossed.lmm(formula, df, term, vcs)

def glmm(formula, df, term, vcs=("item", "para")):
    vc = {v: f"0+C({v})" for v in vcs}
    df = df.reset_index(drop=True)
    r = BinomialBayesMixedGLM.from_formula(formula, vc, df).fit_vb()
    names = list(r.model.exog_names); i = names.index(term)
    b, sd = r.fe_mean[i], r.fe_sd[i]
    return {"beta": float(b), "sd": float(sd), "z": float(b / sd), "p_one_sided": float(stats.norm.sf(b / sd)),
            "n": int(len(df))}

def exact_spearman(score6, ref6, direction):
    """score6, ref6: 6x6 symmetric; correlation over 15 pairs, exact null by relabelling."""
    r = spearmanr(score6[iu], ref6[iu]).statistic
    null = np.array([spearmanr(score6[np.ix_(p, p)][iu], ref6[iu]).statistic for p in PERMS])
    p = np.mean(null <= r + 1e-12) if direction < 0 else np.mean(null >= r - 1e-12)
    return {"rho": float(r), "p_one_sided_exact": float(p)}

# ----------------------------------------------------------------- Family 1
def family1(df):
    out = {}
    df = df.copy()
    def axis(row, key):
        a, b = row["pair"].split("-"); s = row[key]
        A = s[EM.index(a)] - s[EM.index(b)]
        return A if row["gold"] == a else -A
    df["signed"] = df.apply(lambda r: axis(r, "stim"), axis=1)
    df["signed_final"] = df.apply(lambda r: axis(r, "final"), axis=1)
    df["D"] = (df.cond == "DISCRIMINATE").astype(int)
    df["para"] = df.cond + df.para.astype(str)
    df["beh"] = df.apply(lambda r: r.lp[EM.index(r.gold)] - r.lp[EM.index(r.pair.replace(r.gold, "").strip("-"))], axis=1)
    df["correct"] = (df.beh > 0).astype(int)
    out["H1.1"] = lmm("signed ~ D + C(pair)", df, "D")
    out["H1.1_final_token"] = lmm("signed_final ~ D + C(pair)", df, "D")
    # H1.2: paraphrase-level d' and exact permutation over paraphrase-condition labels
    stat_per_pair, nulls, dp = {}, [], {}
    for pr, g in df.groupby("pair"):
        a = pr.split("-")[0]; vals = {}
        for para, h in g.groupby("para"):
            raw = h.signed * np.where(h.gold == a, 1, -1)       # unsigned axis value A_ab
            xa, xb = raw[h.gold == a], raw[h.gold != a]
            vals[para] = (xa.mean() - xb.mean()) / np.sqrt((xa.var() + xb.var()) / 2)
        dp[pr] = vals
        keys = sorted(vals); v = np.array([vals[k] for k in keys]); isD = np.array([k.startswith("DISC") for k in keys])
        stat_per_pair[pr] = float(v[isD].mean() - v[~isD].mean())
        nulls.append(np.array([v[list(c)].mean() - v[[i for i in range(8) if i not in c]].mean()
                               for c in itertools.combinations(range(8), 4)]))
    T = sum(stat_per_pair.values())
    tot = (nulls[0][:, None, None] + nulls[1][None, :, None] + nulls[2][None, None, :]).ravel()
    out["H1.2"] = {"T_sum_delta_dprime": T, "per_pair": stat_per_pair, "dprime": dp,
                   "p_one_sided_exact": float(np.mean(tot >= T - 1e-12)), "n_null": int(tot.size)}
    dD = df[df.D == 1].copy(); dD["zsig"] = (dD.signed - dD.signed.mean()) / dD.signed.std()
    out["H1.3"] = glmm("correct ~ zsig + C(pair)", dD, "zsig")
    # H1.4: magnitude of reconfiguration predicts behavioural gain. Unit: item x paraphrase index
    # (paraphrase k of DISCRIMINATE paired with paraphrase k of RECOGNISE).
    df["pk"] = df.para.str[-1]
    u = df.pivot_table(index=["item", "pk", "pair"], columns="D", values=["signed", "beh"]).reset_index()
    ch = pd.DataFrame({"item": u["item"], "para": u["pk"], "pair": u["pair"],
                       "dS": u[("signed", 1)] - u[("signed", 0)], "dB": u[("beh", 1)] - u[("beh", 0)]})
    ch["zdS"] = (ch.dS - ch.dS.mean()) / ch.dS.std()
    out["H1.4"] = lmm("dB ~ zdS + C(pair)", ch, "zdS")
    out["S1.per_paraphrase_H1.1"] = {k: float(g.dS.mean()) for k, g in ch.groupby("para")}
    out["S1.per_paraphrase_sign_consistent"] = bool(all(g.dS.mean() > 0 for _, g in ch.groupby("para")))
    # secondary: item-level change in internal separation vs change in behaviour
    it = df.groupby(["item", "D"])[["signed", "beh"]].mean().unstack()
    ds, db = it["signed"][1] - it["signed"][0], it["beh"][1] - it["beh"][0]
    out["S1.item_change_corr"] = {"spearman": float(spearmanr(ds, db).statistic), "p": float(spearmanr(ds, db).pvalue), "n": int(len(ds))}
    out["S1.accuracy"] = df.groupby(["pair", "cond"]).correct.mean().unstack().to_dict()
    halves = {}
    for h in (0, 1):
        sub = df[df["item"].str[-2:].astype(int) % 2 == h]
        halves[h] = lmm("signed ~ D + C(pair)", sub, "D")
    out["S1.replication_halves"] = halves
    out["S1.per_pair_H1.1"] = {pr: lmm("signed ~ D", g, "D") for pr, g in df.groupby("pair")}
    return out

# ----------------------------------------------------------------- Family 2
def family2(df):
    out = {}; df = df.copy()
    df["A"] = df.apply(lambda r: r.final[EM.index(r.a)] - r.final[EM.index(r.b)], axis=1)
    df["Beh"] = df.apply(lambda r: r.lp[EM.index(r.a)] - r.lp[EM.index(r.b)], axis=1)
    df["evdir"] = df.cond.map({"EV_A": 1, "EV_B": -1, "NONE": 0, "FILLER": 0})
    out["H2.1"] = lmm("A ~ evdir + C(pair)", df, "evdir")
    w = df.pivot_table(index=["item", "para", "pair"], columns="cond", values=["A", "Beh"]).reset_index()
    E = w[("A", "EV_A")] - w[("A", "EV_B")]; F = w[("A", "FILLER")] - w[("A", "NONE")]
    # item level: half the evidence-driven shift (averaged over paraphrases) against the
    # absolute shift produced by adding an irrelevant neutral sentence (averaged likewise)
    it = pd.DataFrame({"item": w["item"], "E": E, "F": F}).groupby("item").mean()
    per_item = it.E / 2 - it.F.abs()
    wt = stats.wilcoxon(per_item, alternative="greater")
    out["H2.2"] = {"mean_half_evidence_minus_control": float(per_item.mean()), "W": float(wt.statistic),
                   "p_one_sided": float(wt.pvalue), "n_items": int(len(per_item)),
                   "mean_half_evidence": float((it.E / 2).mean()), "mean_abs_control": float(it.F.abs().mean())}
    s = pd.DataFrame({"item": w["item"], "para": w["para"].astype(str), "pair": w["pair"], "Ishift": E,
                      "Bshift": w[("Beh", "EV_A")] - w[("Beh", "EV_B")]})
    s["zI"] = (s.Ishift - s.Ishift.mean()) / s.Ishift.std()
    out["H2.3"] = lmm("Bshift ~ zI + C(pair)", s, "zI")
    df["chose_evidenced"] = np.where(df.cond == "EV_A", df.Beh > 0, np.where(df.cond == "EV_B", df.Beh < 0, np.nan))
    out["S2.behaviour_follows_evidence"] = df[df.cond.isin(["EV_A", "EV_B"])].groupby("pair").chose_evidenced.mean().to_dict()
    out["S2.internal_follows_evidence"] = float(((E > 0)).mean())
    halves = {h: lmm("A ~ evdir + C(pair)", df[df["item"].str[-2:].astype(int) % 2 == h], "evdir") for h in (0, 1)}
    out["S2.replication_halves"] = halves
    out["S2.per_paraphrase_shift"] = {str(k): float(g.mean()) for k, g in pd.Series(E.values).groupby(w["para"].values)}
    out["S2.per_pair_H2.1"] = {pr: lmm("A ~ evdir", g, "evdir") for pr, g in df.groupby("pair")}
    return out

# ----------------------------------------------------------------- Family 3
def family3(df, Jref):
    out = {}; df = df.copy()
    S = np.array(df.gen_score.tolist())                         # n x 6
    df["grp"] = df["item"] + "|" + df.para.astype(str)
    means = pd.DataFrame(S).groupby(df["grp"].values).transform("mean").values
    Sc = S - means
    ti = df.target.map(EM.index).values
    tgt = Sc[np.arange(len(df)), ti]
    non = (Sc.sum(1) - tgt) / 5
    df["margin"] = tgt - non
    df["para"] = df.para.astype(str)
    out["H3.1"] = lmm("margin ~ 1", df, "Intercept")
    sep = np.zeros((6, 6))
    for a, b in itertools.combinations(range(6), 2):
        ra, rb = ti == a, ti == b
        v = (Sc[ra, a] - Sc[ra, b]).mean() + (Sc[rb, b] - Sc[rb, a]).mean()
        sep[a, b] = sep[b, a] = v
    out["H3.2"] = exact_spearman(sep, Jref, direction=-1); out["H3.2"]["internal_separation"] = sep.tolist()
    P = np.array([[d.get(k, 0) for k in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]] for d in df.clf])
    P6 = P / P.sum(1, keepdims=True)
    bsep = np.zeros((6, 6))
    for a, b in itertools.combinations(range(6), 2):
        ra, rb = ti == a, ti == b
        v = (P6[ra, a] - P6[ra, b]).mean() + (P6[rb, b] - P6[rb, a]).mean()
        bsep[a, b] = bsep[b, a] = v
    out["H3.3"] = exact_spearman(bsep, Jref, direction=-1); out["H3.3"]["behavioural_separation"] = bsep.tolist()
    lab = [CLF_MAP[max(d, key=d.get)] for d in df.clf]
    df["correct"] = (np.array(lab) == df.target.values).astype(int)
    df["zm"] = (df.margin - df.margin.mean()) / df.margin.std()
    out["H3.4"] = glmm("correct ~ zm", df, "zm")
    out["S3.classifier_accuracy"] = float(df.correct.mean())
    out["S3.accuracy_by_target"] = df.groupby("target").correct.mean().to_dict()
    out["S3.internal_vs_behavioural_separation"] = {"spearman": float(spearmanr(sep[iu], bsep[iu]).statistic)}
    if "output" in df:
        N6 = np.array([nrc_scores(o) for o in df.output])
        nsep = np.zeros((6, 6))
        for a, b in itertools.combinations(range(6), 2):
            ra, rb = ti == a, ti == b
            nsep[a, b] = nsep[b, a] = (N6[ra, a] - N6[ra, b]).mean() + (N6[rb, b] - N6[rb, a]).mean()
        out["S3.nrc_separation_vs_overlap"] = exact_spearman(nsep, Jref, direction=-1)
        out["S3.nrc_separation_vs_overlap"]["nrc_separation"] = nsep.tolist()
        out["S3.nrc_target_hit_rate"] = float(np.mean(N6.argmax(1)[N6.sum(1) > 0] == ti[N6.sum(1) > 0]))
        out["S3.nrc_internal_separation_corr"] = {"spearman": float(spearmanr(sep[iu], nsep[iu]).statistic)}
    out["S3.planning_state_margin"] = lmm("pm ~ 1", df.assign(pm=_margin(df, "plan_score")), "Intercept")
    return out

def _margin(df, key):
    S = np.array(df[key].tolist()); g = df["item"] + "|" + df.para.astype(str)
    Sc = S - pd.DataFrame(S).groupby(g.values).transform("mean").values
    ti = df.target.map(EM.index).values; t = Sc[np.arange(len(df)), ti]
    return t - (Sc.sum(1) - t) / 5

# ----------------------------------------------------------------- Family 4
def family4(df):
    out = {}; df = df.copy()
    ti = df.target.map(EM.index).values
    df["gen_t"] = np.array(df.gen_score.tolist())[np.arange(len(df)), ti]
    df["plan_t"] = np.array(df.plan_score.tolist())[np.arange(len(df)), ti]
    key = {"ANGER": "anger", "FEAR": "fear"}
    df["Pt"] = [d[key[t]] for d, t in zip(df.clf, df.target)]
    df["para"] = df.cond + df.para.astype(str)
    es = df[df.cond.isin(["EXPRESS", "SUPPRESS"])].copy(); es["E"] = (es.cond == "EXPRESS").astype(int)
    out["H4.1_manipulation"] = lmm("Pt ~ E + C(target)", es, "E")
    out["H4.2"] = lmm("gen_t ~ E + C(target)", es, "E")
    for k in ("gen_t", "plan_t"):
        df[k + "_z"] = (df[k] - df[k].mean()) / df[k].std()
    es = df[df.cond.isin(["EXPRESS", "SUPPRESS"])].copy(); es["E"] = (es.cond == "EXPRESS").astype(int)
    long = pd.concat([es.assign(stage=1, y=es.gen_t_z), es.assign(stage=0, y=es.plan_t_z)])
    long["trial"] = long["trial"].astype(str)
    out["H4.3"] = lmm("y ~ E * stage + C(target)", long, "E:stage", vcs=("item", "para", "trial"))
    out["S4.per_paraphrase_gen_E_minus_S"] = {str(k): float(g[g.E == 1].gen_t.mean() - g[g.E == 0].gen_t.mean())
                                              for k, g in es.groupby(es.para.str[-1])}
    out["S4.planning_E_minus_S"] = lmm("plan_t ~ E + C(target)", es, "E")
    df["zg"] = (df.gen_t - df.gen_t.mean()) / df.gen_t.std()
    out["H4.4"] = lmm("Pt ~ zg + C(cond) + C(target)", df, "zg")
    if "output" in df:
        df["Nt"] = [nrc_scores(o)[EM.index(t) if t != "HAPPINESS" else 3] for o, t in zip(df.output, df.target)]
        es2 = df[df.cond.isin(["EXPRESS", "SUPPRESS"])].copy(); es2["E"] = (es2.cond == "EXPRESS").astype(int)
        out["S4.nrc_manipulation"] = lmm("Nt ~ E + C(target)", es2, "E")
    sn = df[df.cond.isin(["SUPPRESS", "NEUTRAL"])].copy(); sn["Sup"] = (sn.cond == "SUPPRESS").astype(int)
    out["S4.suppress_vs_neutral_gen_t"] = lmm("gen_t ~ Sup + C(target)", sn, "Sup")
    out["S4.suppress_vs_neutral_plan_t"] = lmm("plan_t ~ Sup + C(target)", sn, "Sup")
    out["S4.means"] = df.groupby(["target", "cond"])[["Pt", "gen_t", "plan_t"]].mean().round(4).to_dict()
    # exploratory: neighbour shift under suppression of anger (towards disgust) and fear (towards sadness)
    nb = {"ANGER": "DISGUST", "FEAR": "SADNESS"}
    df["nb_minus_t_gen"] = [g[EM.index(nb[t])] - g[EM.index(t)] for g, t in zip(df.gen_score, df.target)]
    out["X4.neighbour_shift_by_cond"] = df.groupby(["target", "cond"]).nb_minus_t_gen.mean().round(4).to_dict()
    return out

def clean(o):
    if isinstance(o, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o

def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); adj, run = {}, 0
    for i, k in enumerate(keys):
        run = max(run, min(1, (m - i) * ps[k])); adj[k] = run
    return adj

def load(tag):
    return pd.DataFrame([json.loads(l) for l in open(f"{D}/{tag}.jsonl")])

def main(selftest=False):
    if selftest:
        f1, f2, f3, f4, Jref = synthetic()
    else:
        f1, f2, f3, f4 = load("F1"), load("F2"), load("F3"), load("F4")
        f1["stim"] = f1.span_scores.map(lambda d: d["stim"])
        if MEASURE != "prereg":          # Deviation 1: signature-consistent measures from saved activations
            mp = {t: json.load(open(f"{D}/measures_{t}.json")) for t in ("F1", "F2", "F3", "F4")}
            sfx = "_" + MEASURE
            f1["stim"] = f1.trial.map(lambda t: mp["F1"][t]["stim" + sfx]); f1["final"] = f1.trial.map(lambda t: mp["F1"][t]["final" + sfx])
            f2["final"] = f2.trial.map(lambda t: mp["F2"][t]["final" + sfx])
            for nm, f in (("F3", f3), ("F4", f4)):
                f["gen_score"] = f.trial.map(lambda t: mp[nm][t]["gen" + sfx]); f["plan_score"] = f.trial.map(lambda t: mp[nm][t]["prompt_last" + sfx])
        excl = {}
        for nm, f in (("F3", f3), ("F4", f4)):      # preregistered exclusion: empty generations
            bad = f.gen_score.isna() | (f.output.str.strip() == "")
            excl[nm] = int(bad.sum())
        f3 = f3[~(f3.gen_score.isna() | (f3.output.str.strip() == ""))]
        f4 = f4[~(f4.gen_score.isna() | (f4.output.str.strip() == ""))]
        Jref = np.array(json.load(open("/home/claude/r4/results/reference_map_jaccard_en.json"))["jaccard"])
    if selftest:
        excl = {"F3": 0, "F4": 0}
    R = {"excluded_empty_generations": excl, "F1": family1(f1), "F2": family2(f2), "F3": family3(f3, Jref), "F4": family4(f4)}
    conf = {"H1.1": R["F1"]["H1.1"]["p_one_sided"], "H1.2": R["F1"]["H1.2"]["p_one_sided_exact"],
            "H1.3": R["F1"]["H1.3"]["p_one_sided"], "H1.4": R["F1"]["H1.4"]["p_one_sided"], "H2.1": R["F2"]["H2.1"]["p_one_sided"],
            "H2.2": R["F2"]["H2.2"]["p_one_sided"], "H2.3": R["F2"]["H2.3"]["p_one_sided"],
            "H3.1": R["F3"]["H3.1"]["p_one_sided"], "H3.2": R["F3"]["H3.2"]["p_one_sided_exact"],
            "H3.3": R["F3"]["H3.3"]["p_one_sided_exact"], "H3.4": R["F3"]["H3.4"]["p_one_sided"],
            "H4.2": R["F4"]["H4.2"]["p_one_sided"], "H4.3": R["F4"]["H4.3"]["p_one_sided"],
            "H4.4": R["F4"]["H4.4"]["p_one_sided"]}
    R["confirmatory_raw_p"] = conf; R["confirmatory_holm_p"] = holm(conf)
    tag = "selftest" if selftest else f"confirmatory_{MEASURE}"
    json.dump(clean(R), open(f"/home/claude/r4/results/{tag}_results.json", "w"), indent=1, default=str)
    for k in conf:
        print(f"{k}: raw p={conf[k]:.4g}  Holm p={R['confirmatory_holm_p'][k]:.4g}")

def synthetic():
    rng = np.random.default_rng(0); rows = []
    for pr in PAIRS:
        for g in pr:
            for k in range(20):
                for cond in ("RECOGNISE", "DISCRIMINATE"):
                    for p in range(4):
                        s = rng.normal(0, 1, 6); lp = rng.normal(-3, 1, 6)
                        rows.append({"trial": f"{g}{k}|{cond}|{p}", "item": f"F1_{g[:3]}_{k:02d}", "pair": "-".join(pr),
                                     "gold": g, "cond": cond, "para": p, "lp": lp.tolist(), "final": s.tolist(), "stim": s.tolist()})
    f1 = pd.DataFrame(rows); rows = []
    for a, b in PAIRS:
        for k in range(10):
            for cond in ("NONE", "FILLER", "EV_A", "EV_B"):
                for p in range(4):
                    rows.append({"trial": f"{a}{k}|{cond}|{p}", "item": f"F2_{a[:3]}_{k:02d}", "pair": f"{a}-{b}", "a": a, "b": b,
                                 "cond": cond, "para": p, "lp": rng.normal(-3, 1, 6).tolist(), "final": rng.normal(0, 1, 6).tolist()})
    f2 = pd.DataFrame(rows); rows = []
    labs = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    for k in range(10):
        for e in EM:
            for p in range(3):
                pr = rng.dirichlet(np.ones(7))
                rows.append({"trial": f"{k}|{e}|{p}", "item": f"F3_{k:02d}", "target": e, "para": p,
                             "gen_score": rng.normal(0, 1, 6).tolist(), "plan_score": rng.normal(0, 1, 6).tolist(),
                             "clf": dict(zip(labs, pr.tolist()))})
    f3 = pd.DataFrame(rows); rows = []
    for e in ("ANGER", "FEAR"):
        for k in range(10):
            for cond in ("EXPRESS", "SUPPRESS", "NEUTRAL"):
                for p in range(3):
                    pr = rng.dirichlet(np.ones(7))
                    rows.append({"trial": f"{e}{k}|{cond}|{p}", "item": f"F4_{e[:3]}_{k:02d}", "target": e, "cond": cond, "para": p,
                                 "gen_score": rng.normal(0, 1, 6).tolist(), "plan_score": rng.normal(0, 1, 6).tolist(),
                                 "clf": dict(zip(labs, pr.tolist()))})
    f4 = pd.DataFrame(rows)
    J = rng.uniform(0, 0.2, (6, 6)); J = (J + J.T) / 2; np.fill_diagonal(J, 1)
    return f1, f2, f3, f4, J

MEASURE = next((a.split("=")[1] for a in sys.argv if a.startswith("--measure=")), "prereg")

if __name__ == "__main__":
    main(selftest="--selftest" in sys.argv)
